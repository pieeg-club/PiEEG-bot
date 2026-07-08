"""Retrieval: load ``index.json`` and find the most relevant chunks.

No vector database — the whole index is a NumPy matrix in memory and retrieval
is a single cosine-similarity dot product. This is more than fast enough for a
corpus of this size and keeps deployment to "copy one JSON file".
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import numpy as np

from . import config, embeddings


@dataclass
class Retrieved:
    text: str
    url: str
    source: str
    title: str
    score: float


class Retriever:
    """In-memory cosine-similarity index over documentation chunks."""

    def __init__(self, chunks: list[dict], matrix: np.ndarray) -> None:
        self._chunks = chunks
        # Pre-normalise so retrieval is a plain dot product.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    @classmethod
    def load(cls) -> "Retriever":
        if not config.INDEX_PATH.exists():
            raise SystemExit(
                f"Index not found at {config.INDEX_PATH}. "
                "Run `pieeg-bot ingest` first."
            )
        data = json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))
        matrix = np.asarray(data["vectors"], dtype=np.float32)
        return cls(data["chunks"], matrix)

    async def search(self, query: str) -> list[Retrieved]:
        """Return up to ``TOP_K`` chunks above ``MIN_SCORE``, best first."""
        # Embedding is CPU-bound; run it off the event loop.
        vector = await asyncio.to_thread(embeddings.embed_query, query)
        q = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(q) or 1.0
        scores = self._matrix @ (q / norm)

        order = np.argsort(scores)[::-1][: config.TOP_K]
        results: list[Retrieved] = []
        for idx in order:
            score = float(scores[idx])
            if score < config.MIN_SCORE:
                break
            chunk = self._chunks[idx]
            results.append(
                Retrieved(
                    text=chunk["text"],
                    url=chunk["url"],
                    source=chunk["source"],
                    title=chunk["title"],
                    score=score,
                )
            )
        return results
