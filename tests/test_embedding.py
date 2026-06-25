import numpy as np
import pytest

from ezvocab.embedding import (
    bytes_to_embedding,
    cosine_similarity,
    embedding_to_bytes,
    most_similar,
)


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_most_similar_returns_top_k():
    query = [1.0, 0.0]
    candidates = [
        (10, [0.9, 0.1]),
        (20, [0.1, 0.9]),
        (30, [1.0, 0.0]),
        (40, [-1.0, 0.0]),
    ]
    results = most_similar(query, candidates, top_k=2)
    assert len(results) == 2
    assert results[0][0] == 30
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][0] == 10


def test_most_similar_empty_candidates():
    results = most_similar([1.0, 0.0], [], top_k=3)
    assert results == []


def test_embedding_roundtrip():
    original = [0.1, 0.2, 0.3, 0.4]
    data = embedding_to_bytes(original)
    restored = bytes_to_embedding(data)
    assert restored == pytest.approx(original, abs=1e-6)
