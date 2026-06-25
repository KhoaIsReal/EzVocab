from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from ezvocab.models import CEFR_DESCRIPTIONS, WordSuggestion


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
        ),
        WordSuggestion(
            "resilient",
            "Able to recover quickly after difficulty.",
            "The team stayed resilient after the first failure.",
            "adjective",
        ),
        WordSuggestion(
            "clarify",
            "To make something easier to understand.",
            "Could you clarify the last instruction?",
            "verb",
        ),
        WordSuggestion(
            "scarce",
            "Not easy to find or get.",
            "Fresh water is scarce in some regions.",
            "adjective",
        ),
        WordSuggestion(
            "evaluate",
            "To judge or calculate the value or quality of something.",
            "We need to evaluate each answer carefully.",
            "verb",
        ),
        WordSuggestion(
            "concise",
            "Short and clear, without extra words.",
            "Her summary was concise and useful.",
            "adjective",
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
        if word and definition:
            suggestions.append(WordSuggestion(word, definition, example, part_of_speech))
    if not suggestions:
        raise SuggestionProviderError("LLM response did not contain usable suggestions.")
    return suggestions


def suggestions_to_json(suggestions: list[WordSuggestion]) -> str:
    return json.dumps([asdict(suggestion) for suggestion in suggestions], indent=2)


def _suggestion_prompt(level: str, count: int, known_words: list[str]) -> str:
    known = ", ".join(known_words[:100]) or "none"
    level_desc = CEFR_DESCRIPTIONS.get(level, f"Level: {level}")
    return (
        "Suggest English vocabulary words for a learner.\n"
        f"CEFR Level: {level} – {level_desc}\n"
        f"Count: {count}.\n"
        f"Already known words: {known}.\n"
        "Return only JSON in this exact shape: "
        "[{\"word\":\"...\",\"definition\":\"...\",\"example_sentence\":\"...\","
        "\"part_of_speech\":\"...\"}]."
    )
