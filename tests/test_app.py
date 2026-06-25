from ezvocab.app import AppServices, suggestions_view
from ezvocab.models import WordSuggestion


def test_app_services_can_start_with_in_memory_database():
    services = AppServices.create(":memory:")

    assert services.cards.count() == 0
    assert services.settings.get("provider") == "fallback"


def test_suggestions_view_can_build_before_mounting():
    services = AppServices.create(":memory:")

    control = suggestions_view(services, lambda message: None, lambda index: None)

    assert control is not None


def test_suggestions_view_can_build_pending_suggestion_tiles():
    services = AppServices.create(":memory:")
    services.suggestions.add_many(
        "fallback",
        [WordSuggestion("meticulous", "Very careful", "She is meticulous.", "adjective")],
    )

    control = suggestions_view(services, lambda message: None, lambda index: None)

    assert control is not None
