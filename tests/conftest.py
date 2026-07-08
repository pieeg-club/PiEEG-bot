"""Shared test fixtures.

Loads ``PiEEG-bot/.env`` (if present) into the environment so the live
Anthropic test can pick up ``ANTHROPIC_API_KEY`` without exporting it by hand.
No dependency on python-dotenv — the file format we need is trivial.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_env() -> None:
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()
