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


def test_parse_suggestions_rejects_invalid_json():
    with pytest.raises(SuggestionProviderError):
        parse_suggestions("not json")


def test_fallback_provider_skips_known_words():
    provider = FallbackSuggestionProvider()

    suggestions = provider.suggest_words("intermediate", 3, ["meticulous"])

    assert len(suggestions) == 3
    assert "meticulous" not in {suggestion.word for suggestion in suggestions}


def test_suggestions_to_json_outputs_list():
    provider = FallbackSuggestionProvider()
    raw = suggestions_to_json(provider.suggest_words("intermediate", 1, []))

    assert json.loads(raw)[0]["word"]
