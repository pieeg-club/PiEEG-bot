"""Local embeddings via fastembed.

Anthropic doesn't offer an embeddings endpoint, so we embed on-device with
fastembed (ONNX runtime, CPU). For a corpus this small this is free, keyless
and fast, and it means ingestion needs no API credentials at all.

The model is loaded lazily and cached: the first call downloads/initialises the
ONNX model (a few seconds), every call after that is milliseconds.
"""

from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so `--help` and unit tests don't pay the import cost.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=config.EMBED_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed documentation chunks (used at ingestion time)."""
    return [vec.tolist() for vec in _model().embed(list(texts))]


def embed_query(text: str) -> list[float]:
    """Embed a single search query.

    bge-style models expect a short instruction prefix on queries only;
    fastembed's ``query_embed`` adds it, which improves retrieval quality.
    """
    return list(next(iter(_model().query_embed([text]))).tolist())


def warm() -> None:
    """Force model initialisation up front (e.g. at bot startup)."""
    _model()
