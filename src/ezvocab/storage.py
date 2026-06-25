from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ezvocab.models import (
    CardRecord,
    CardStatus,
    LearningCard,
    NewCard,
    SuggestionRecord,
    SuggestionStatus,
    WordSuggestion,
    learning_cards_from_json,
    learning_cards_to_json,
    make_definition_card,
)


SCHEMA_VERSION = 3


def default_db_path() -> Path:
    return Path.home() / ".ezvocab" / "ezvocab.sqlite3"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def encode_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def decode_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.migrate()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def migrate(self) -> None:
        with self.transaction() as conn:
            current_version = conn.execute("PRAGMA user_version").fetchone()[0]

            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    example_sentence TEXT NOT NULL DEFAULT '',
                    part_of_speech TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'manual',
                    status TEXT NOT NULL DEFAULT 'active',
                    fsrs_card TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    embedding BLOB DEFAULT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cards_due_at
                    ON cards(status, due_at);

                CREATE TABLE IF NOT EXISTS review_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
                    rating TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    next_due_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    example_sentence TEXT NOT NULL DEFAULT '',
                    part_of_speech TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

            if current_version < 2:
                try:
                    conn.execute("ALTER TABLE cards ADD COLUMN embedding BLOB DEFAULT NULL")
                except sqlite3.OperationalError:
                    pass

            if current_version < 3:
                try:
                    conn.execute("ALTER TABLE cards ADD COLUMN learning_cards TEXT NOT NULL DEFAULT '[]'")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE suggestions ADD COLUMN learning_cards TEXT NOT NULL DEFAULT '[]'")
                except sqlite3.OperationalError:
                    pass
                conn.execute(
                    """
                    UPDATE cards SET learning_cards = json_array(
                        json_object(
                            'card_type', 'definition',
                            'front', json_object('prompt', word, 'answer', part_of_speech),
                            'back', json_object('prompt', definition, 'answer', example_sentence)
                        )
                    ) WHERE learning_cards = '[]'
                    """
                )

            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._ensure_default_settings(conn)

    def _ensure_default_settings(self, conn: sqlite3.Connection) -> None:
        defaults = {
            "provider": "fallback",
            "gemini_api_key": "",
            "gemini_model": "gemini-2.5-flash",
            "ollama_endpoint": "http://localhost:11434",
            "ollama_model": "llama3.2",
            "daily_suggestion_count": "5",
            "learner_level": "B1",
        }
        conn.executemany(
            "INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)",
            defaults.items(),
        )


class CardRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, card: NewCard, fsrs_card: str, due_at: datetime, embedding: bytes | None = None) -> CardRecord:
        now = encode_datetime(utc_now())
        cards_json = learning_cards_to_json(list(card.learning_cards)) if card.learning_cards else "[]"
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO cards(
                    word, definition, example_sentence, part_of_speech, source,
                    status, fsrs_card, due_at, created_at, updated_at, embedding,
                    learning_cards
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card.word.strip(),
                    card.definition.strip(),
                    card.example_sentence.strip(),
                    card.part_of_speech.strip(),
                    card.source.strip() or "manual",
                    CardStatus.ACTIVE.value,
                    fsrs_card,
                    encode_datetime(due_at),
                    now,
                    now,
                    embedding,
                    cards_json,
                ),
            )
            card_id = int(cursor.lastrowid)
        return self.get(card_id)

    def get(self, card_id: int) -> CardRecord:
        row = self.db.conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        if row is None:
            raise KeyError(f"Card {card_id} does not exist")
        return _card_from_row(row)

    def list_due(self, now: datetime, limit: int = 20) -> list[CardRecord]:
        rows = self.db.conn.execute(
            """
            SELECT * FROM cards
            WHERE status = ? AND due_at <= ?
            ORDER BY due_at ASC, id ASC
            LIMIT ?
            """,
            (CardStatus.ACTIVE.value, encode_datetime(now), limit),
        ).fetchall()
        return [_card_from_row(row) for row in rows]

    def list_all(self) -> list[CardRecord]:
        rows = self.db.conn.execute("SELECT * FROM cards ORDER BY word COLLATE NOCASE").fetchall()
        return [_card_from_row(row) for row in rows]

    def known_words(self) -> list[str]:
        rows = self.db.conn.execute("SELECT word FROM cards ORDER BY word COLLATE NOCASE").fetchall()
        return [str(row["word"]) for row in rows]

    def update_embedding(self, card_id: int, embedding: bytes) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE cards SET embedding = ?, updated_at = ? WHERE id = ?",
                (embedding, encode_datetime(utc_now()), card_id),
            )

    def list_with_embeddings(self) -> list[CardRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM cards WHERE embedding IS NOT NULL ORDER BY word COLLATE NOCASE"
        ).fetchall()
        return [_card_from_row(row) for row in rows]

    def list_without_embeddings(self) -> list[CardRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM cards WHERE embedding IS NULL ORDER BY word COLLATE NOCASE"
        ).fetchall()
        return [_card_from_row(row) for row in rows]

    def update_schedule(self, card_id: int, fsrs_card: str, due_at: datetime) -> CardRecord:
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE cards
                SET fsrs_card = ?, due_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (fsrs_card, encode_datetime(due_at), encode_datetime(utc_now()), card_id),
            )
        return self.get(card_id)

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS count FROM cards").fetchone()
        return int(row["count"])


class ReviewLogRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        card_id: int,
        rating: str,
        reviewed_at: datetime,
        previous_state: str,
        next_due_at: datetime,
    ) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO review_logs(card_id, rating, reviewed_at, previous_state, next_due_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    card_id,
                    rating,
                    encode_datetime(reviewed_at),
                    previous_state,
                    encode_datetime(next_due_at),
                ),
            )

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS count FROM review_logs").fetchone()
        return int(row["count"])


class SuggestionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add_many(self, provider: str, suggestions: list[WordSuggestion]) -> list[SuggestionRecord]:
        created_at = encode_datetime(utc_now())
        ids: list[int] = []
        with self.db.transaction() as conn:
            for suggestion in suggestions:
                cards_json = learning_cards_to_json(list(suggestion.learning_cards)) if suggestion.learning_cards else "[]"
                cursor = conn.execute(
                    """
                    INSERT INTO suggestions(
                        word, definition, example_sentence, part_of_speech,
                        provider, status, created_at, learning_cards
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        suggestion.word.strip(),
                        suggestion.definition.strip(),
                        suggestion.example_sentence.strip(),
                        suggestion.part_of_speech.strip(),
                        provider,
                        SuggestionStatus.PENDING.value,
                        created_at,
                        cards_json,
                    ),
                )
                ids.append(int(cursor.lastrowid))
        return [self.get(suggestion_id) for suggestion_id in ids]

    def get(self, suggestion_id: int) -> SuggestionRecord:
        row = self.db.conn.execute(
            "SELECT * FROM suggestions WHERE id = ?",
            (suggestion_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Suggestion {suggestion_id} does not exist")
        return _suggestion_from_row(row)

    def list_pending(self) -> list[SuggestionRecord]:
        rows = self.db.conn.execute(
            """
            SELECT * FROM suggestions
            WHERE status = ?
            ORDER BY created_at DESC, id DESC
            """,
            (SuggestionStatus.PENDING.value,),
        ).fetchall()
        return [_suggestion_from_row(row) for row in rows]

    def mark(self, suggestion_id: int, status: SuggestionStatus) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE suggestions SET status = ? WHERE id = ?",
                (status.value, suggestion_id),
            )


class SettingsRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, key: str, default: str = "") -> str:
        row = self.db.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return default if row is None else str(row["value"])

    def set(self, key: str, value: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO settings(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def as_dict(self) -> dict[str, str]:
        rows = self.db.conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}


def _card_from_row(row: sqlite3.Row) -> CardRecord:
    embedding_raw = row["embedding"]
    embedding = bytes(embedding_raw) if embedding_raw is not None else None
    raw_cards = str(row["learning_cards"]) if "learning_cards" in row.keys() else "[]"
    learning_cards = tuple(learning_cards_from_json(raw_cards))
    if not learning_cards:
        learning_cards = (make_definition_card(
            str(row["word"]),
            str(row["definition"]),
            str(row["part_of_speech"]),
            str(row["example_sentence"]),
        ),)
    return CardRecord(
        id=int(row["id"]),
        word=str(row["word"]),
        definition=str(row["definition"]),
        example_sentence=str(row["example_sentence"]),
        part_of_speech=str(row["part_of_speech"]),
        source=str(row["source"]),
        status=str(row["status"]),
        fsrs_card=str(row["fsrs_card"]),
        due_at=decode_datetime(str(row["due_at"])),
        created_at=decode_datetime(str(row["created_at"])),
        updated_at=decode_datetime(str(row["updated_at"])),
        embedding=embedding,
        learning_cards=learning_cards,
    )


def _suggestion_from_row(row: sqlite3.Row) -> SuggestionRecord:
    raw_cards = str(row["learning_cards"]) if "learning_cards" in row.keys() else "[]"
    learning_cards = tuple(learning_cards_from_json(raw_cards))
    if not learning_cards:
        learning_cards = (make_definition_card(
            str(row["word"]),
            str(row["definition"]),
            str(row["part_of_speech"]),
            str(row["example_sentence"]),
        ),)
    return SuggestionRecord(
        id=int(row["id"]),
        word=str(row["word"]),
        definition=str(row["definition"]),
        example_sentence=str(row["example_sentence"]),
        part_of_speech=str(row["part_of_speech"]),
        provider=str(row["provider"]),
        status=str(row["status"]),
        created_at=decode_datetime(str(row["created_at"])),
        learning_cards=learning_cards,
    )
