from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Optional


class ReviewRating(StrEnum):
    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"


class CardStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class CEFRLevel(StrEnum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class CardType(StrEnum):
    TRANSLATION = "translation"
    CLOZE = "cloze"
    EXAMPLE = "example"
    DEFINITION = "definition"
    SYNONYM = "synonym"
    PRONUNCIATION = "pronunciation"
    LISTENING = "listening"
    IMAGE = "image"


CEFR_DESCRIPTIONS: dict[CEFRLevel, str] = {
    CEFRLevel.A1: "Beginner – basic everyday expressions, simple phrases, introduce yourself",
    CEFRLevel.A2: "Elementary – frequently used expressions, routine tasks, simple information exchange",
    CEFRLevel.B1: "Intermediate – main points on familiar topics, travel, experiences, opinions",
    CEFRLevel.B2: "Upper Intermediate – complex texts, abstract topics, technical discussions in your field",
    CEFRLevel.C1: "Advanced – wide range of demanding texts, fluent and spontaneous expression",
    CEFRLevel.C2: "Proficient – virtually everything heard or read, precise nuanced expression",
}


class SuggestionStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CardRecord:
    id: int
    word: str
    definition: str
    example_sentence: str
    part_of_speech: str
    source: str
    status: str
    fsrs_card: str
    due_at: datetime
    created_at: datetime
    updated_at: datetime
    embedding: Optional[bytes] = None
    learning_cards: tuple[LearningCard, ...] = ()


@dataclass(frozen=True)
class NewCard:
    word: str
    definition: str
    example_sentence: str
    part_of_speech: str = ""
    source: str = "manual"
    learning_cards: tuple[LearningCard, ...] = ()


@dataclass(frozen=True)
class ReviewLogRecord:
    id: int
    card_id: int
    rating: str
    reviewed_at: datetime
    previous_state: str
    next_due_at: datetime


@dataclass(frozen=True)
class CardFace:
    prompt: str
    answer: str


@dataclass(frozen=True)
class LearningCard:
    card_type: str
    front: CardFace
    back: CardFace


def learning_cards_to_json(cards: list[LearningCard]) -> str:
    return json.dumps([asdict(c) for c in cards])


def learning_cards_from_json(raw: str) -> list[LearningCard]:
    if not raw or raw == "[]":
        return []
    data = json.loads(raw)
    return [
        LearningCard(
            card_type=item["card_type"],
            front=CardFace(**item["front"]),
            back=CardFace(**item["back"]),
        )
        for item in data
    ]


def make_definition_card(word: str, definition: str, part_of_speech: str, example_sentence: str) -> LearningCard:
    return LearningCard(
        card_type=CardType.DEFINITION,
        front=CardFace(prompt=word, answer=part_of_speech),
        back=CardFace(prompt=definition, answer=example_sentence),
    )


@dataclass(frozen=True)
class WordSuggestion:
    word: str
    definition: str
    example_sentence: str
    part_of_speech: str = ""
    learning_cards: tuple[LearningCard, ...] = ()


@dataclass(frozen=True)
class SuggestionRecord:
    id: int
    word: str
    definition: str
    example_sentence: str
    part_of_speech: str
    provider: str
    status: str
    created_at: datetime
    learning_cards: tuple[LearningCard, ...] = ()
