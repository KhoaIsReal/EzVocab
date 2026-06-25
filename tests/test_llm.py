import json

import pytest

from ezvocab.llm import (
    FallbackSuggestionProvider,
    SuggestionProviderError,
    parse_suggestions,
    suggestions_to_json,
)


def test_parse_suggestions_accepts_json_list():
    parsed = parse_suggestions(
        json.dumps(
            [
                {
                    "word": "meticulous",
                    "definition": "Careful and precise",
                    "example_sentence": "She is meticulous.",
                    "part_of_speech": "adjective",
                }
            ]
        )
    )

    assert parsed[0].word == "meticulous"
    assert len(parsed[0].learning_cards) == 1
    assert parsed[0].learning_cards[0].card_type == "definition"


def test_parse_suggestions_with_cards():
    parsed = parse_suggestions(
        json.dumps(
            [
                {
                    "word": "abandon",
                    "definition": "To leave permanently",
                    "example_sentence": "They abandoned the project.",
                    "part_of_speech": "verb",
                    "cards": [
                        {
                            "type": "translation",
                            "front": {"prompt": "từ bỏ", "answer": ""},
                            "back": {"prompt": "abandon", "answer": ""},
                        },
                        {
                            "type": "cloze",
                            "front": {"prompt": "He had to _____ the car.", "answer": ""},
                            "back": {"prompt": "abandon", "answer": ""},
                        },
                    ],
                }
            ]
        )
    )

    assert parsed[0].word == "abandon"
    assert len(parsed[0].learning_cards) == 2
    assert parsed[0].learning_cards[0].card_type == "translation"
    assert parsed[0].learning_cards[0].front.prompt == "từ bỏ"
    assert parsed[0].learning_cards[1].card_type == "cloze"


def test_parse_suggestions_rejects_invalid_json():
    with pytest.raises(SuggestionProviderError):
        parse_suggestions("not json")


def test_fallback_provider_skips_known_words():
    provider = FallbackSuggestionProvider()

    suggestions = provider.suggest_words("intermediate", 3, ["meticulous"])

    assert len(suggestions) == 3
    assert "meticulous" not in {suggestion.word for suggestion in suggestions}


def test_fallback_provider_has_learning_cards():
    provider = FallbackSuggestionProvider()

    suggestions = provider.suggest_words("intermediate", 1, [])

    assert len(suggestions[0].learning_cards) >= 4
    types = {c.card_type for c in suggestions[0].learning_cards}
    assert "translation" in types
    assert "cloze" in types
    assert "definition" in types
    assert "example" in types


def test_suggestions_to_json_outputs_list():
    provider = FallbackSuggestionProvider()
    raw = suggestions_to_json(provider.suggest_words("intermediate", 1, []))
    data = json.loads(raw)

    assert data[0]["word"]
    assert "cards" in data[0]
    assert len(data[0]["cards"]) >= 4
