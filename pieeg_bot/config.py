"""Configuration for PiEEG-bot.

All tunables live here so the rest of the code stays declarative. Secrets come
from environment variables (see ``.env.example``); the documentation sources are
declared as plain data so adding a new repo is a one-line change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "index.json"
# Working directory where source repos are cloned during ingestion.
CLONE_DIR = DATA_DIR / "sources"

# --- Models ----------------------------------------------------------------

# Chat runs on Anthropic. Claude Haiku is the cheapest current model
# ($1 / $5 per MTok) and the fastest — ideal for short, grounded doc answers.
CHAT_MODEL = os.environ.get("PIEEG_CHAT_MODEL", "claude-haiku-4-5")
# Cap the answer length (Discord messages top out at 2000 chars anyway).
MAX_TOKENS = int(os.environ.get("PIEEG_MAX_TOKENS", "1024"))

# Anthropic has no embeddings API, so embeddings run locally via fastembed
# (ONNX on CPU — free, no API key, and plenty fast for a tiny corpus).
EMBED_MODEL = os.environ.get("PIEEG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Retrieval -------------------------------------------------------------

TOP_K = int(os.environ.get("PIEEG_TOP_K", "5"))
# Minimum cosine similarity for a chunk to be considered relevant at all.
# Tuned for bge-small (relevant passages typically score ~0.6+).
MIN_SCORE = float(os.environ.get("PIEEG_MIN_SCORE", "0.30"))

# --- Community --------------------------------------------------------------

DISCORD_INVITE = "https://discord.gg/neJ45FR6Sv"


# --- Documentation sources --------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A documentation repository to index.

    Attributes:
        name: Human-readable label, shown in citations.
        repo: Git URL to clone.
        content_dir: Sub-directory within the repo that holds the markdown.
        url_base: Public base URL that ``content_dir`` maps to. A file at
            ``<content_dir>/software/getting-started/quick-start.mdx`` becomes
            ``<url_base>/software/getting-started/quick-start``.
    """

    name: str
    repo: str
    content_dir: str
    url_base: str


SOURCES: list[Source] = [
    Source(
        name="PiEEG Docs",
        repo="https://github.com/pieeg-club/PiEEG-docs.git",
        content_dir="content",
        url_base="https://pieeg.com/docs",
    ),
    Source(
        name="PiEEG News",
        repo="https://github.com/pieeg-club/PiEEG-com.git",
        content_dir="content/news",
        url_base="https://pieeg.com/news",
    ),
]


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable {name!r}. "
            "See .env.example for the full list."
        )
    return value
