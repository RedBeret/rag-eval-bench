"""Sentence-transformers embedding wrapper."""

from __future__ import annotations

import hashlib
from typing import Protocol


class Embedder(Protocol):
    """Protocol for embedding functions."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Embed text using a sentence-transformers model.

    Default model: all-MiniLM-L6-v2 (23MB, fast, good quality for retrieval).
    Downloads on first use and caches locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


class MockEmbedder:
    """Deterministic mock embedder for tests (no model download required)."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return stable deterministic vectors based on SHA-256 bytes."""
        result = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = (digest * ((self._dim // len(digest)) + 1))[: self._dim]
            vec = [byte / 255.0 for byte in raw]
            result.append(vec)
        return result
