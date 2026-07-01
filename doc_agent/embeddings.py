"""Pluggable embeddings for the RAG pipeline.

The default :class:`HashingEmbeddings` needs no extra dependencies, no network
and no API key — ideal for demos, tests and CI. For real semantic retrieval,
set ``EMBEDDINGS=fastembed`` (after ``pip install fastembed``) or inject any
LangChain ``Embeddings`` implementation (OpenAI, Voyage, HuggingFace, ...).
"""
from __future__ import annotations

import hashlib
import math
import os

from langchain_core.embeddings import Embeddings


class HashingEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embedding.

    Not semantic — it's a lightweight, reproducible stand-in so the RAG
    architecture (splitting, vector store, retrieval) runs anywhere. Swap in a
    real embedding model for production quality.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbeddings(Embeddings):
    """Semantic embeddings via the optional ``fastembed`` library (local, ONNX)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, next(iter(self._model.embed([text])))))


def get_embeddings() -> Embeddings:
    """Return the embeddings selected by the ``EMBEDDINGS`` env var (default: hashing)."""
    if os.getenv("EMBEDDINGS", "hashing").lower() == "fastembed":
        return FastEmbedEmbeddings()
    return HashingEmbeddings()
