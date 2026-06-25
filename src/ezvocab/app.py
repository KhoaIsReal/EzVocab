from __future__ import annotations

from dataclasses import dataclass

import flet as ft
import numpy as np

from ezvocab.embedding import GeminiEmbedder, embedding_to_bytes, bytes_to_embedding, cosine_similarity
from ezvocab.llm import SuggestionProviderError, provider_from_settings
from ezvocab.models import CEFRLevel, NewCard, ReviewRating, SuggestionStatus
from ezvocab.scheduler import SchedulerService
from ezvocab.storage import (
    CardRepository,
    Database,
    ReviewLogRepository,
    SettingsRepository,
    SuggestionRepository,
)


@dataclass
class AppServices:
    db: Database
    cards: CardRepository
    reviews: ReviewLogRepository
    suggestions: SuggestionRepository
    settings: SettingsRepository
    scheduler: SchedulerService

    @classmethod
    def create(cls, db_path: str | None = None) -> "AppServices":
        db = Database(db_path)
        cards = CardRepository(db)
        reviews = ReviewLogRepository(db)
        suggestions = SuggestionRepository(db)
        settings = SettingsRepository(db)
        scheduler = SchedulerService(cards, reviews)
        return cls(db, cards, reviews, suggestions, settings, scheduler)


def run() -> None:
    ft.run(main)


def main(page: ft.Page) -> None:
    services = AppServices.create()
    page.title = "EzVocab"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.min_width = 900
    page.window.min_height = 640

    selected_index = 0
    content = ft.Container(expand=True, padding=24)
    status = ft.SnackBar(content=ft.Text(""))
    page.overlay.append(status)

    def toast(message: str) -> None:
        status.content = ft.Text(message)
        status.open = True
        page.update()

    def route(index: int) -> None:
        nonlocal selected_index
        selected_index = index
        nav.selected_index = selected_index
        builders = [review_view, add_card_view, suggestions_view, settings_view]
        content.content = builders[selected_index](services, toast, route)
        page.update()

    nav = ft.NavigationRail(
        selected_index=selected_index,
        label_type=ft.NavigationRailLabelType.ALL,
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.SCHOOL_OUTLINED, label="Review"),
            ft.NavigationRailDestination(icon=ft.Icons.ADD_CARD_OUTLINED, label="Add"),
            ft.NavigationRailDestination(icon=ft.Icons.AUTO_AWESOME_OUTLINED, label="Suggest"),
            ft.NavigationRailDestination(icon=ft.Icons.SETTINGS_OUTLINED, label="Settings"),
        ],
        on_change=lambda e: route(int(e.control.selected_index)),
    )

    page.add(
        ft.Row(
            expand=True,
            controls=[
                ft.Container(nav, padding=ft.Padding(0, 12, 0, 0)),
                ft.VerticalDivider(width=1),
                content,
            ],
        )
    )
    route(0)


def outline_border(color: ft.Colors = ft.Colors.BLUE_GREY_100, width: int = 1) -> ft.Border:
    side = ft.BorderSide(width, color)
    return ft.Border(side, side, side, side)


def review_view(services: AppServices, toast, route) -> ft.Control:
    due_cards = services.scheduler.get_due_cards(limit=1)
    total_due = len(services.scheduler.get_due_cards(limit=100))
    total_cards = services.cards.count()

    header = ft.Row(
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        controls=[
            ft.Column(
                spacing=4,
                controls=[
                    ft.Text("Review", size=28, weight=ft.FontWeight.BOLD),
                    ft.Text(f"{total_due} due today · {total_cards} total cards"),
                ],
            ),
            ft.IconButton(icon=ft.Icons.REFRESH, tooltip="Refresh", on_click=lambda _: route(0)),
        ],
    )

    if not due_cards:
        return ft.Column(
            expand=True,
            spacing=24,
            controls=[
                header,
                ft.Container(
                    expand=True,
                    alignment=ft.Alignment(0, 0),
                    content=ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, size=64, color=ft.Colors.GREEN_600),
                            ft.Text("No cards due", size=22, weight=ft.FontWeight.W_600),
                            ft.FilledButton("Add a card", icon=ft.Icons.ADD, on_click=lambda _: route(1)),
                        ],
                    ),
                ),
            ],
        )

    card = due_cards[0]
    answer = ft.Column(
        visible=False,
        spacing=12,
        controls=[
            ft.Divider(),
            ft.Text(card.definition, size=18),
            ft.Text(card.example_sentence, italic=True, color=ft.Colors.BLUE_GREY_700),
            ft.Text(card.part_of_speech, color=ft.Colors.BLUE_GREY_500),
        ],
    )
    reveal_button = ft.FilledButton("Reveal", icon=ft.Icons.VISIBILITY, on_click=lambda _: reveal())
    rating_row = ft.Row(visible=False, wrap=True, spacing=10)

    def reveal() -> None:
        answer.visible = True
        reveal_button.visible = False
        rating_row.visible = True
        reveal_button.page.update()

    def rate(rating: ReviewRating) -> None:
        services.scheduler.review(card.id, rating)
        toast(f"Reviewed '{card.word}' as {rating.value}.")
        route(0)

    rating_row.controls = [
        ft.OutlinedButton("Again", icon=ft.Icons.REPLAY, on_click=lambda _: rate(ReviewRating.AGAIN)),
        ft.OutlinedButton("Hard", icon=ft.Icons.TRENDING_UP, on_click=lambda _: rate(ReviewRating.HARD)),
        ft.FilledButton("Good", icon=ft.Icons.THUMB_UP_OUTLINED, on_click=lambda _: rate(ReviewRating.GOOD)),
        ft.OutlinedButton("Easy", icon=ft.Icons.STAR_OUTLINE, on_click=lambda _: rate(ReviewRating.EASY)),
    ]

    return ft.Column(
        expand=True,
        spacing=24,
        controls=[
            header,
            ft.Container(
                padding=28,
                border=outline_border(),
                border_radius=8,
                content=ft.Column(
                    spacing=18,
                    controls=[
                        ft.Text(card.word, size=42, weight=ft.FontWeight.BOLD),
                        ft.Text(f"Source: {card.source}"),
                        answer,
                        ft.Row(controls=[reveal_button]),
                        rating_row,
                    ],
                ),
            ),
        ],
    )


def add_card_view(services: AppServices, toast, route) -> ft.Control:
    word = ft.TextField(label="Word", autofocus=True)
    part_of_speech = ft.TextField(label="Part of speech")
    definition = ft.TextField(label="Definition", multiline=True, min_lines=2)
    example = ft.TextField(label="Example sentence", multiline=True, min_lines=2)

    def save(_: ft.ControlEvent) -> None:
        if not word.value or not definition.value:
            toast("Word and definition are required.")
            return
        services.scheduler.create_card(
            NewCard(
                word=word.value,
                definition=definition.value,
                example_sentence=example.value or "",
                part_of_speech=part_of_speech.value or "",
            )
        )
        toast(f"Added '{word.value}'.")
        route(0)

    return ft.Column(
        expand=True,
        spacing=18,
        controls=[
            ft.Text("Add Card", size=28, weight=ft.FontWeight.BOLD),
            ft.ResponsiveRow(
                controls=[
                    ft.Container(word, col={"sm": 12, "md": 8}),
                    ft.Container(part_of_speech, col={"sm": 12, "md": 4}),
                ]
            ),
            definition,
            example,
            ft.Row(
                controls=[
                    ft.FilledButton("Save", icon=ft.Icons.SAVE_OUTLINED, on_click=save),
                    ft.OutlinedButton("Review", icon=ft.Icons.SCHOOL_OUTLINED, on_click=lambda _: route(0)),
                ]
            ),
        ],
    )


def _ensure_card_embeddings(services: AppServices, embedder: GeminiEmbedder) -> None:
    cards_without = services.cards.list_without_embeddings()
    if not cards_without:
        return
    texts = [c.word for c in cards_without]
    if not texts:
        return
    embeddings = embedder.embed_texts(texts)
    if len(embeddings) != len(cards_without):
        return
    for card, emb in zip(cards_without, embeddings):
        services.cards.update_embedding(card.id, embedding_to_bytes(emb))


def _filter_by_similarity(
    services: AppServices,
    embedder: GeminiEmbedder,
    suggestions: list,
    threshold: float = 0.85,
) -> list:
    cards_with_emb = services.cards.list_with_embeddings()
    if not cards_with_emb:
        return suggestions
    known_embeddings = [
        (c.id, bytes_to_embedding(c.embedding))
        for c in cards_with_emb
        if c.embedding
    ]
    if not known_embeddings:
        return suggestions

    suggestion_texts = [s.word for s in suggestions]
    suggestion_embs = embedder.embed_texts(suggestion_texts)

    filtered = []
    for suggestion, s_emb in zip(suggestions, suggestion_embs):
        is_duplicate = False
        s_arr = np.array(s_emb, dtype=np.float64)
        for _, k_emb in known_embeddings:
            if cosine_similarity(s_arr, np.array(k_emb, dtype=np.float64)) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(suggestion)
    return filtered


def suggestions_view(services: AppServices, toast, route) -> ft.Control:
    settings = services.settings.as_dict()
    pending_column = ft.Column(spacing=12, scroll=ft.ScrollMode.AUTO, expand=True)

    def refresh_pending() -> None:
        pending_column.controls.clear()
        for suggestion in services.suggestions.list_pending():
            pending_column.controls.append(_suggestion_tile(services, suggestion, toast, refresh_pending))
        if not pending_column.controls:
            pending_column.controls.append(ft.Text("No pending suggestions."))
        try:
            pending_page = pending_column.page
        except RuntimeError:
            return
        pending_page.update()

    def generate(_: ft.ControlEvent) -> None:
        settings_dict = services.settings.as_dict()
        provider = provider_from_settings(settings_dict)
        count = int(services.settings.get("daily_suggestion_count", "5") or "5")
        level = services.settings.get("learner_level", "B1")

        known = services.cards.known_words()

        try:
            words = provider.suggest_words(level, count, known)
        except Exception as exc:
            toast(f"Suggestion error: {exc}")
            return

        if not words:
            toast("No new suggestions available.")
            return

        if provider.name == "gemini" and settings_dict.get("gemini_api_key"):
            try:
                embedder = GeminiEmbedder(api_key=settings_dict["gemini_api_key"])
                _ensure_card_embeddings(services, embedder)
                words = _filter_by_similarity(services, embedder, words)
            except Exception:
                pass

        if not words:
            toast("All suggestions were duplicates of existing cards.")
            return
        services.suggestions.add_many(provider.name, words)
        toast(f"Generated {len(words)} suggestions with {provider.name}.")
        refresh_pending()

    refresh_pending()
    return ft.Column(
        expand=True,
        spacing=18,
        controls=[
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Column(
                        spacing=4,
                        controls=[
                            ft.Text("Daily Suggestions", size=28, weight=ft.FontWeight.BOLD),
                            ft.Text(f"Provider: {settings.get('provider', 'fallback')}"),
                        ],
                    ),
                    ft.FilledButton("Generate", icon=ft.Icons.AUTO_AWESOME, on_click=generate),
                ],
            ),
            pending_column,
        ],
    )


def _suggestion_tile(
    services: AppServices,
    suggestion,
    toast,
    refresh_pending,
) -> ft.Control:
    def accept(_: ft.ControlEvent) -> None:
        services.scheduler.create_card(
            NewCard(
                word=suggestion.word,
                definition=suggestion.definition,
                example_sentence=suggestion.example_sentence,
                part_of_speech=suggestion.part_of_speech,
                source=f"llm:{suggestion.provider}",
            )
        )
        services.suggestions.mark(suggestion.id, SuggestionStatus.ACCEPTED)
        toast(f"Accepted '{suggestion.word}'.")
        refresh_pending()

    def reject(_: ft.ControlEvent) -> None:
        services.suggestions.mark(suggestion.id, SuggestionStatus.REJECTED)
        toast(f"Rejected '{suggestion.word}'.")
        refresh_pending()

    return ft.Container(
        padding=16,
        border=outline_border(),
        border_radius=8,
        content=ft.Column(
            spacing=8,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(suggestion.word, size=20, weight=ft.FontWeight.W_600),
                        ft.Text(suggestion.part_of_speech, color=ft.Colors.BLUE_GREY_500),
                    ],
                ),
                ft.Text(suggestion.definition),
                ft.Text(suggestion.example_sentence, italic=True, color=ft.Colors.BLUE_GREY_700),
                ft.Row(
                    controls=[
                        ft.FilledButton("Accept", icon=ft.Icons.CHECK, on_click=accept),
                        ft.OutlinedButton("Reject", icon=ft.Icons.CLOSE, on_click=reject),
                    ]
                ),
            ],
        ),
    )


def settings_view(services: AppServices, toast, route) -> ft.Control:
    settings = services.settings.as_dict()
    provider = ft.Dropdown(
        label="Provider",
        value=settings.get("provider", "fallback"),
        options=[
            ft.dropdown.Option("fallback", "Offline fallback"),
            ft.dropdown.Option("gemini", "Gemini"),
            ft.dropdown.Option("ollama", "Ollama"),
        ],
    )
    gemini_key = ft.TextField(label="Gemini API key", password=True, value=settings.get("gemini_api_key", ""))
    gemini_model = ft.TextField(label="Gemini model", value=settings.get("gemini_model", "gemini-2.5-flash"))
    ollama_endpoint = ft.TextField(label="Ollama endpoint", value=settings.get("ollama_endpoint", "http://localhost:11434"))
    ollama_model = ft.TextField(label="Ollama model", value=settings.get("ollama_model", "llama3.2"))
    daily_count = ft.TextField(label="Daily suggestion count", value=settings.get("daily_suggestion_count", "5"))
    learner_level = ft.Dropdown(
        label="CEFR Level",
        value=settings.get("learner_level", "B1"),
        options=[
            ft.dropdown.Option("A1", "A1 - Beginner"),
            ft.dropdown.Option("A2", "A2 - Elementary"),
            ft.dropdown.Option("B1", "B1 - Intermediate"),
            ft.dropdown.Option("B2", "B2 - Upper Intermediate"),
            ft.dropdown.Option("C1", "C1 - Advanced"),
            ft.dropdown.Option("C2", "C2 - Proficient"),
        ],
    )

    def save(_: ft.ControlEvent) -> None:
        for key, control in {
            "provider": provider,
            "gemini_api_key": gemini_key,
            "gemini_model": gemini_model,
            "ollama_endpoint": ollama_endpoint,
            "ollama_model": ollama_model,
            "daily_suggestion_count": daily_count,
            "learner_level": learner_level,
        }.items():
            services.settings.set(key, str(control.value or ""))
        toast("Settings saved.")
        route(3)

    return ft.Column(
        expand=True,
        spacing=18,
        scroll=ft.ScrollMode.AUTO,
        controls=[
            ft.Text("Settings", size=28, weight=ft.FontWeight.BOLD),
            provider,
            ft.Divider(),
            ft.Text("Gemini", size=18, weight=ft.FontWeight.W_600),
            gemini_key,
            gemini_model,
            ft.Divider(),
            ft.Text("Ollama", size=18, weight=ft.FontWeight.W_600),
            ollama_endpoint,
            ollama_model,
            ft.Divider(),
            daily_count,
            learner_level,
            ft.Row(controls=[ft.FilledButton("Save", icon=ft.Icons.SAVE_OUTLINED, on_click=save)]),
        ],
    )
