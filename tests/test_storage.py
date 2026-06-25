from ezvocab.models import NewCard, WordSuggestion
from ezvocab.scheduler import serialize_card
from ezvocab.storage import CardRepository, Database, SettingsRepository, SuggestionRepository, utc_now

from fsrs import Card


def test_database_migration_and_card_crud():
    db = Database(":memory:")
    cards = CardRepository(db)

    created = cards.add(
        NewCard("clarify", "Make easier to understand", "Please clarify this."),
        serialize_card(Card()),
        utc_now(),
    )

    assert created.id > 0
    assert cards.get(created.id).word == "clarify"
    assert cards.count() == 1
    assert cards.known_words() == ["clarify"]


def test_settings_defaults_and_update():
    db = Database(":memory:")
    settings = SettingsRepository(db)

    assert settings.get("provider") == "fallback"
    settings.set("provider", "ollama")
    assert settings.get("provider") == "ollama"


def test_suggestion_repository_pending_flow():
    db = Database(":memory:")
    suggestions = SuggestionRepository(db)

    created = suggestions.add_many(
        "fallback",
        [WordSuggestion("resilient", "Able to recover", "She is resilient.", "adjective")],
    )

    assert len(created) == 1
    assert suggestions.list_pending()[0].word == "resilient"
