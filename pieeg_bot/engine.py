"""The Discord-free question-answering engine.

The retrieval-augmented pipeline lives here so it can be driven from anywhere.
Both the Discord bot (:mod:`pieeg_bot.bot`) and the command-line REPL
(``pieeg-bot chat``) are thin wrappers around :class:`Engine` — which means you
can exercise the full pipeline end-to-end without a Discord token.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from . import config, embeddings, llm
from .retriever import Retrieved, Retriever

SYSTEM_PROMPT = (
    "You are PiEEG-bot, the documentation companion for the PiEEG "
    "brain-computer-interface community. Answer the user's question using ONLY "
    "the documentation excerpts provided in CONTEXT. Be concise, friendly and "
    "practical. If the context does not contain the answer, say so plainly and "
    "suggest rephrasing the question or asking the team for help — do not invent "
    "details, part numbers, pin-outs or API names. Do not add source links "
    "yourself; they are appended automatically."
)

MAX_REPLY = 1900  # Stay under Discord's 2000-character message limit.


def _build_context(results: list[Retrieved]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] Source: {r.source} ({r.url})\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def format_sources(results: list[Retrieved]) -> str:
    """Render a de-duplicated Markdown list of the cited source links."""
    seen: dict[str, str] = {}
    for r in results:
        seen.setdefault(r.url, r.title or r.source)
    return "\n".join(f"• [{title}](<{url}>)" for url, title in seen.items())


@dataclass
class Answer:
    """A grounded answer plus the chunks it was built from.

    ``reply`` is the raw model text; ``results`` are the retrieved chunks (empty
    when nothing relevant was found). Front-ends decide how to render it.
    """

    reply: str
    results: list[Retrieved] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return bool(self.results)

    @property
    def sources(self) -> str:
        return format_sources(self.results)

    def to_discord(self, limit: int = MAX_REPLY) -> str:
        """Format the answer as a single Discord message with source links."""
        if not self.results:
            return self.reply
        body = f"{self.reply}\n\n**Sources**\n{self.sources}"
        if len(body) > limit:
            body = body[:limit].rsplit("\n", 1)[0] + "\n…"
        return body


class Engine:
    """Turns a question into a grounded, source-cited :class:`Answer`."""

    def __init__(self, api_key: str) -> None:
        self._anthropic = llm.build_client(api_key)
        self._retriever = Retriever.load()

    @classmethod
    def create(cls) -> "Engine":
        """Build an engine from the environment (needs ``ANTHROPIC_API_KEY``)."""
        return cls(config.require_env("ANTHROPIC_API_KEY"))

    async def warm(self) -> None:
        """Load the embedding model up front so the first query isn't slow."""
        await asyncio.to_thread(embeddings.warm)

    async def close(self) -> None:
        await self._anthropic.close()

    async def answer(self, question: str) -> Answer:
        """Retrieve, generate and return a grounded answer for ``question``."""
        question = question.strip()
        if not question:
            return Answer(
                "Ask me anything about PiEEG hardware, software or setup."
            )

        results = await self._retriever.search(question)
        if not results:
            return Answer(
                "I couldn't find that in the PiEEG documentation. "
                "Try rephrasing your question or ask the team here for help."
            )

        context = _build_context(results)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        reply = await llm.chat(self._anthropic, SYSTEM_PROMPT, user)
        return Answer(reply, results)
