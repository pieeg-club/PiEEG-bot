"""Anthropic chat client.

Embeddings run locally (see ``embeddings.py``); the only remote call is the
grounded chat completion, which we make with the official ``anthropic`` SDK's
async client. The SDK manages its own HTTP session, retries and timeouts.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from . import config


def build_client(api_key: str) -> AsyncAnthropic:
    """Create the async Anthropic client used for the lifetime of the bot."""
    return AsyncAnthropic(api_key=api_key)


async def chat(client: AsyncAnthropic, system: str, user: str) -> str:
    """Run a single grounded completion and return the assistant text."""
    message = await client.messages.create(
        model=config.CHAT_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Concatenate any text blocks (there is normally exactly one).
    parts = [
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts).strip()
