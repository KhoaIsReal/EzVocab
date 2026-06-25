from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from ezvocab.models import (
    CEFR_DESCRIPTIONS,
    CardFace,
    CardType,
    LearningCard,
    WordSuggestion,
    make_definition_card,
)


class SuggestionProviderError(RuntimeError):
    pass


class SuggestionProvider(ABC):
    name: str

    @abstractmethod
    def suggest_words(
        self,
        level: str,
        count: int,
        known_words: list[str],
    ) -> list[WordSuggestion]:
        raise NotImplementedError


class FallbackSuggestionProvider(SuggestionProvider):
    name = "fallback"

    _WORDS = [
        WordSuggestion(
            "meticulous",
            "Very careful and precise.",
            "She kept meticulous notes during the lecture.",
            "adjective",
            (
                LearningCard(CardType.TRANSLATION, CardFace("cẩn thận", ""), CardFace("meticulous", "")),
                LearningCard(CardType.CLOZE, CardFace("She kept _____ notes during the lecture.", ""), CardFace("meticulous", "")),
                LearningCard(CardType.DEFINITION, CardFace("meticulous", "adjective"), CardFace("Very careful and precise.", "She kept meticulous notes during the lecture.")),
                LearningCard(CardType.EXAMPLE, CardFace("She kept meticulous notes during the lecture.", ""), CardFace("Cô ấy ghi chép cẩn thận trong bài giảng.", "")),
            ),
        ),
        WordSuggestion(
            "resilient",
            "Able to recover quickly after difficulty.",
            "The team stayed resilient after the first failure.",
            "adjective",
            (
                LearningCard(CardType.TRANSLATION, CardFace("kiên cường", ""), CardFace("resilient", "")),
                LearningCard(CardType.CLOZE, CardFace("The team stayed _____ after the first failure.", ""), CardFace("resilient", "")),
                LearningCard(CardType.DEFINITION, CardFace("resilient", "adjective"), CardFace("Able to recover quickly after difficulty.", "The team stayed resilient after the first failure.")),
                LearningCard(CardType.EXAMPLE, CardFace("The team stayed resilient after the first failure.", ""), CardFace("Đội đã kiên cường sau thất bại đầu tiên.", "")),
            ),
        ),
        WordSuggestion(
            "clarify",
            "To make something easier to understand.",
            "Could you clarify the last instruction?",
            "verb",
            (
                LearningCard(CardType.TRANSLATION, CardFace("làm rõ", ""), CardFace("clarify", "")),
                LearningCard(CardType.CLOZE, CardFace("Could you _____ the last instruction?", ""), CardFace("clarify", "")),
                LearningCard(CardType.DEFINITION, CardFace("clarify", "verb"), CardFace("To make something easier to understand.", "Could you clarify the last instruction?")),
                LearningCard(CardType.EXAMPLE, CardFace("Could you clarify the last instruction?", ""), CardFace("Bạn có thể làm rõ hướng dẫn cuối cùng không?", "")),
            ),
        ),
        WordSuggestion(
            "scarce",
            "Not easy to find or get.",
            "Fresh water is scarce in some regions.",
            "adjective",
            (
                LearningCard(CardType.TRANSLATION, CardFace("khan hiếm", ""), CardFace("scarce", "")),
                LearningCard(CardType.CLOZE, CardFace("Fresh water is _____ in some regions.", ""), CardFace("scarce", "")),
                LearningCard(CardType.DEFINITION, CardFace("scarce", "adjective"), CardFace("Not easy to find or get.", "Fresh water is scarce in some regions.")),
                LearningCard(CardType.EXAMPLE, CardFace("Fresh water is scarce in some regions.", ""), CardFace("Nước ngọt khan hiếm ở một số vùng.", "")),
            ),
        ),
        WordSuggestion(
            "evaluate",
            "To judge or calculate the value or quality of something.",
            "We need to evaluate each answer carefully.",
            "verb",
            (
                LearningCard(CardType.TRANSLATION, CardFace("đánh giá", ""), CardFace("evaluate", "")),
                LearningCard(CardType.CLOZE, CardFace("We need to _____ each answer carefully.", ""), CardFace("evaluate", "")),
                LearningCard(CardType.DEFINITION, CardFace("evaluate", "verb"), CardFace("To judge or calculate the value or quality of something.", "We need to evaluate each answer carefully.")),
                LearningCard(CardType.EXAMPLE, CardFace("We need to evaluate each answer carefully.", ""), CardFace("Chúng ta cần đánh giá mỗi câu trả lời cẩn thận.", "")),
            ),
        ),
        WordSuggestion(
            "concise",
            "Short and clear, without extra words.",
            "Her summary was concise and useful.",
            "adjective",
            (
                LearningCard(CardType.TRANSLATION, CardFace("súc tích", ""), CardFace("concise", "")),
                LearningCard(CardType.CLOZE, CardFace("Her summary was _____ and useful.", ""), CardFace("concise", "")),
                LearningCard(CardType.DEFINITION, CardFace("concise", "adjective"), CardFace("Short and clear, without extra words.", "Her summary was concise and useful.")),
                LearningCard(CardType.EXAMPLE, CardFace("Her summary was concise and useful.", ""), CardFace("Tóm tắt của cô ấy súc tích và hữu ích.", "")),
            ),
        ),
    ]

    def suggest_words(self, level: str, count: int, known_words: list[str]) -> list[WordSuggestion]:
        known = {word.casefold() for word in known_words}
        suggestions = [word for word in self._WORDS if word.word.casefold() not in known]
        return suggestions[: max(0, count)]


class GeminiSuggestionProvider(SuggestionProvider):
    name = "gemini"

    def __init__(self, api_key: str = "", model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model

    def suggest_words(self, level: str, count: int, known_words: list[str]) -> list[WordSuggestion]:
        if not self.api_key:
            raise SuggestionProviderError("Gemini API key is missing.")
        try:
            from google import genai
        except ImportError as exc:
            raise SuggestionProviderError("google-genai is not installed.") from exc

        try:
            client = genai.Client(api_key=self.api_key)
            prompt = _suggestion_prompt(level, count, known_words)
            response = client.models.generate_content(model=self.model, contents=prompt)
            return parse_suggestions(getattr(response, "text", ""))
        except SuggestionProviderError:
            raise
        except Exception as exc:
            raise SuggestionProviderError(f"Gemini API error: {exc}") from exc


class OllamaSuggestionProvider(SuggestionProvider):
    name = "ollama"

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_seconds: int = 60,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def suggest_words(self, level: str, count: int, known_words: list[str]) -> list[WordSuggestion]:
        payload = {
            "model": self.model,
            "prompt": _suggestion_prompt(level, count, known_words),
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SuggestionProviderError(f"Ollama request failed: {exc}") from exc
        return parse_suggestions(str(body.get("response", "")))


def provider_from_settings(settings: dict[str, str]) -> SuggestionProvider:
    provider = settings.get("provider", "fallback")
    if provider == "gemini":
        return GeminiSuggestionProvider(
            api_key=settings.get("gemini_api_key", ""),
            model=settings.get("gemini_model", "gemini-2.5-flash"),
        )
    if provider == "ollama":
        return OllamaSuggestionProvider(
            endpoint=settings.get("ollama_endpoint", "http://localhost:11434"),
            model=settings.get("ollama_model", "llama3.2"),
        )
    return FallbackSuggestionProvider()


def parse_suggestions(raw_text: str) -> list[WordSuggestion]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SuggestionProviderError("LLM response was not valid JSON.") from exc

    if isinstance(data, dict):
        data = data.get("suggestions", [])
    if not isinstance(data, list):
        raise SuggestionProviderError("LLM response must be a list of suggestions.")

    suggestions: list[WordSuggestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip()
        definition = str(item.get("definition", "")).strip()
        example = str(item.get("example_sentence") or item.get("example") or "").strip()
        part_of_speech = str(item.get("part_of_speech", "")).strip()
        if not word or not definition:
            continue

        learning_cards = _parse_learning_cards(item.get("cards"), word, definition, part_of_speech, example)
        suggestions.append(WordSuggestion(word, definition, example, part_of_speech, tuple(learning_cards)))

    if not suggestions:
        raise SuggestionProviderError("LLM response did not contain usable suggestions.")
    return suggestions


def _parse_learning_cards(
    raw_cards: Any,
    word: str,
    definition: str,
    part_of_speech: str,
    example: str,
) -> list[LearningCard]:
    if isinstance(raw_cards, list) and raw_cards:
        cards: list[LearningCard] = []
        for item in raw_cards:
            if not isinstance(item, dict):
                continue
            card_type = str(item.get("type", "")).strip()
            front_data = item.get("front", {})
            back_data = item.get("back", {})
            if not card_type or not front_data or not back_data:
                continue
            front = CardFace(
                prompt=str(front_data.get("prompt", "")).strip(),
                answer=str(front_data.get("answer", "")).strip(),
            )
            back = CardFace(
                prompt=str(back_data.get("prompt", "")).strip(),
                answer=str(back_data.get("answer", "")).strip(),
            )
            cards.append(LearningCard(card_type=card_type, front=front, back=back))
        if cards:
            return cards

    return [make_definition_card(word, definition, part_of_speech, example)]


def suggestions_to_json(suggestions: list[WordSuggestion]) -> str:
    return json.dumps(
        [
            {
                "word": s.word,
                "definition": s.definition,
                "example_sentence": s.example_sentence,
                "part_of_speech": s.part_of_speech,
                "cards": [asdict(c) for c in s.learning_cards],
            }
            for s in suggestions
        ],
        indent=2,
    )


def _suggestion_prompt(level: str, count: int, known_words: list[str]) -> str:
    known = ", ".join(known_words[:100]) or "none"
    level_desc = CEFR_DESCRIPTIONS.get(level, f"Level: {level}")
    return (
        "Suggest English vocabulary words for a learner.\n"
        f"CEFR Level: {level} – {level_desc}\n"
        f"Count: {count}.\n"
        f"Already known words: {known}.\n\n"
        "For each word, generate 4 learning cards with different types to support active recall.\n"
        "Card types: translation, cloze, definition, example.\n\n"
        "Return only JSON in this exact shape:\n"
        "[\n"
        "  {\n"
        '    "word": "abandon",\n'
        '    "definition": "To leave something permanently.",\n'
        '    "example_sentence": "They abandoned the project.",\n'
        '    "part_of_speech": "verb",\n'
        '    "cards": [\n'
        '      {"type": "translation", "front": {"prompt": "từ bỏ", "answer": ""}, "back": {"prompt": "abandon", "answer": ""}},\n'
        '      {"type": "cloze", "front": {"prompt": "He had to _____ the car after the accident.", "answer": ""}, "back": {"prompt": "abandon", "answer": ""}},\n'
        '      {"type": "definition", "front": {"prompt": "To leave something permanently.", "answer": ""}, "back": {"prompt": "abandon", "answer": ""}},\n'
        '      {"type": "example", "front": {"prompt": "They abandoned the project.", "answer": ""}, "back": {"prompt": "Họ đã từ bỏ dự án.", "answer": ""}}\n'
        "    ]\n"
        "  }\n"
        "]\n\n"
        "The translation card should translate the word to the learner's native language (Vietnamese).\n"
        "The cloze card should have a blank (_____) where the word goes.\n"
        "The definition card front should be the definition, back should be the word.\n"
        "The example card should show an English sentence on front, Vietnamese translation on back.\n"
    )
