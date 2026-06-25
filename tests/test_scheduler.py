from ezvocab.models import NewCard, ReviewRating
from ezvocab.scheduler import SchedulerService
from ezvocab.storage import CardRepository, Database, ReviewLogRepository, utc_now


def test_new_card_is_due_immediately():
    db = Database(":memory:")
    service = SchedulerService(CardRepository(db), ReviewLogRepository(db))

    created = service.create_card(NewCard("scarce", "Not enough", "Food was scarce."))
    due = service.get_due_cards(utc_now(), limit=10)

    assert created.id in {card.id for card in due}


def test_review_updates_due_date_and_logs_review():
    db = Database(":memory:")
    cards = CardRepository(db)
    logs = ReviewLogRepository(db)
    service = SchedulerService(cards, logs)
    created = service.create_card(NewCard("evaluate", "Judge quality", "Evaluate the answer."))

    updated = service.review(created.id, ReviewRating.GOOD, utc_now())

    assert updated.due_at >= created.due_at
    assert logs.count() == 1


def test_again_rating_records_relearning_attempt():
    db = Database(":memory:")
    cards = CardRepository(db)
    logs = ReviewLogRepository(db)
    service = SchedulerService(cards, logs)
    created = service.create_card(NewCard("concise", "Short and clear", "Keep it concise."))

    updated = service.review(created.id, ReviewRating.AGAIN, utc_now())

    assert updated.id == created.id
    assert logs.count() == 1
