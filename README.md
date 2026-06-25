# EzVocab

A cross-platform English vocabulary learning app with FSRS spaced repetition scheduling and AI-powered word suggestions.

## Features

- **FSRS Spaced Repetition** — scientifically optimized review scheduling
- **AI Suggestions** — get new vocabulary from Gemini or Ollama, filtered by your CEFR level (A1–C2)
- **Gemini Embeddings** — deduplicates suggestions against your existing cards using semantic similarity
- **Offline Fallback** — works without an API key using a built-in word list
- **Local Storage** — all data stored in SQLite at `~/.ezvocab/ezvocab.sqlite3`

## Quick Start

```bash
pip install -e ".[dev]"
python main.py
```

## Settings

| Setting | Default | Description |
|---|---|---|
| Provider | `fallback` | `gemini`, `ollama`, or `fallback` |
| Gemini API key | — | Get from [Google AI Studio](https://aistudio.google.com/apikey) |
| Gemini model | `gemini-2.5-flash` | Model for text generation |
| CEFR Level | `B1` | A1–C2, controls word difficulty |
| Daily suggestion count | `5` | Words per generation |

## Build Executables

```bash
pip install pyinstaller
pyinstaller --onefile --name EzVocab --add-data "src/ezvocab:ezvocab" main.py
```

Cross-platform builds (Linux, Windows, macOS) are automated via GitHub Actions — push a tag or trigger manually from the Actions tab.

## Test

```bash
pytest
```

## Project Structure

```
src/ezvocab/
  app.py          — Flet UI (Review, Add, Suggest, Settings views)
  models.py       — Data models and enums
  storage.py      — SQLite database and repositories
  scheduler.py    — FSRS spaced repetition service
  llm.py          — LLM suggestion providers (Gemini, Ollama, fallback)
  embedding.py    — Gemini embedding client and cosine similarity
```
