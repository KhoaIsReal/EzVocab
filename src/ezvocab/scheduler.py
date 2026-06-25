from __future__ import annotations

import base64
import pickle
from datetime import datetime, timezone
from typing import Any

from fsrs import Card, Rating, Scheduler

from ezvocab.models import CardRecord, NewCard, ReviewRating
from ezvocab.storage import CardRepository, ReviewLogRepository


class SchedulerService:
    def __init__(
        self,
        cards: CardRepository,
        review_logs: ReviewLogRepository,
        scheduler: Scheduler | None = None,
    ) -> None:
        self.cards = cards
        self.review_logs = review_logs
        self.scheduler = scheduler or Scheduler()

    def create_card(self, card: NewCard) -> CardRecord:
        fsrs_card = Card()
        return self.cards.add(card, serialize_card(fsrs_card), _due_at(fsrs_card))

    def get_due_cards(self, now: datetime | None = None, limit: int = 20) -> list[CardRecord]:
        return self.cards.list_due(_aware(now), limit)

    def review(
        self,
        card_id: int,
        rating: ReviewRating | str,
        reviewed_at: datetime | None = None,
    ) -> CardRecord:
        reviewed_at = _aware(reviewed_at)
        card_record = self.cards.get(card_id)
        fsrs_card = deserialize_card(card_record.fsrs_card)
        previous_state = str(getattr(fsrs_card, "state", "unknown"))
        reviewed_card, _review_log = self.scheduler.review_card(
            fsrs_card,
            _to_fsrs_rating(rating),
            review_datetime=reviewed_at,
        )
        next_due = _due_at(reviewed_card)
        updated = self.cards.update_schedule(card_id, serialize_card(reviewed_card), next_due)
        self.review_logs.add(
            card_id=card_id,
            rating=str(ReviewRating(rating).value if isinstance(rating, str) else rating.value),
            reviewed_at=reviewed_at,
            previous_state=previous_state,
            next_due_at=next_due,
        )
        return updated


def serialize_card(card: Card) -> str:
    payload = pickle.dumps(card)
    return base64.b64encode(payload).decode("ascii")


def deserialize_card(payload: str) -> Card:
    card = pickle.loads(base64.b64decode(payload.encode("ascii")))
    if not isinstance(card, Card):
        raise TypeError("Stored FSRS payload did not contain an fsrs.Card")
    return card


def _to_fsrs_rating(rating: ReviewRating | str) -> Rating:
    normalized = ReviewRating(rating)
    mapping = {
        ReviewRating.AGAIN: Rating.Again,
        ReviewRating.HARD: Rating.Hard,
        ReviewRating.GOOD: Rating.Good,
        ReviewRating.EASY: Rating.Easy,
    }
    return mapping[normalized]


def _due_at(card: Any) -> datetime:
    due = getattr(card, "due", None)
    if due is None:
        return datetime.now(timezone.utc)
    return _aware(due)


def _aware(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
