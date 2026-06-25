from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class NewCard:
    word: str
    definition: str
    example_sentence: str
    part_of_speech: str = ""
    source: str = "manual"


@dataclass(frozen=True)
class ReviewLogRecord:
    id: int
    card_id: int
    rating: str
    reviewed_at: datetime
    previous_state: str
    next_due_at: datetime


@dataclass(frozen=True)
class WordSuggestion:
    word: str
    definition: str
    example_sentence: str
    part_of_speech: str = ""


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
