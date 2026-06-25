from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class EmbeddingError(RuntimeError):
    pass


class GeminiEmbedder:
    def __init__(self, api_key: str = "", model: str = "text-embedding-004") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model

    def _get_client(self):
        if not self.api_key:
            raise EmbeddingError("Gemini API key is missing.")
        try:
            from google import genai
        except ImportError as exc:
            raise EmbeddingError("google-genai is not installed.") from exc
        return genai.Client(api_key=self.api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        try:
            result = client.models.embed_content(
                model=self.model,
                contents=texts,
            )
        except Exception as exc:
            raise EmbeddingError(f"Gemini embedding request failed: {exc}") from exc
        return [list(e.values) for e in result.embeddings]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def most_similar(
    query_embedding: list[float],
    candidate_embeddings: list[tuple[int, list[float]]],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    if not candidate_embeddings:
        return []
    query = np.array(query_embedding, dtype=np.float64)
    results: list[tuple[int, float]] = []
    for idx, emb in candidate_embeddings:
        score = cosine_similarity(query, np.array(emb, dtype=np.float64))
        results.append((idx, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def embedding_to_bytes(emb: list[float]) -> bytes:
    return np.array(emb, dtype=np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> list[float]:
    return np.frombuffer(data, dtype=np.float32).tolist()
