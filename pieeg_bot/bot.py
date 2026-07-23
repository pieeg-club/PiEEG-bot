"""The Discord bot: answers PiEEG questions from the pre-built index.

Two ways to ask:

* ``/ask <question>`` — a slash command (discoverable, works anywhere).
* Mention the bot (``@PiEEG-bot how do I ...``) in any channel it can see.

Answers are grounded strictly in the retrieved documentation and always end
with source links back to the live docs/site. If the docs don't cover the
question, the bot says so and points to the community Discord rather than
guessing.
"""

from __future__ import annotations

import asyncio

import discord
from discord import app_commands

from . import config, embeddings, llm
from .retriever import Retriever

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


def _build_context(results) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] Source: {r.source} ({r.url})\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def _format_sources(results) -> str:
    seen: dict[str, str] = {}
    for r in results:
        seen.setdefault(r.url, r.title or r.source)
    lines = [f"• [{title}](<{url}>)" for url, title in seen.items()]
    return "\n".join(lines)


class PiEEGBot(discord.Client):
    def __init__(self, api_key: str) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self._anthropic = llm.build_client(api_key)
        self._retriever = Retriever.load()

    async def setup_hook(self) -> None:
        # Load the embedding model up front so the first query isn't slow.
        await asyncio.to_thread(embeddings.warm)
        await self.tree.sync()

    async def close(self) -> None:
        await self._anthropic.close()
        await super().close()

    async def answer(self, question: str) -> str:
        """Retrieve, generate and format a grounded answer for ``question``."""
        question = question.strip()
        if not question:
            return "Ask me anything about PiEEG hardware, software or setup."

        results = await self._retriever.search(question)
        if not results:
            return (
                "I couldn't find that in the PiEEG documentation. "
                "Try rephrasing your question or ask the team here for help."
            )

        context = _build_context(results)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        reply = await llm.chat(self._anthropic, SYSTEM_PROMPT, user)

        sources = _format_sources(results)
        body = f"{reply}\n\n**Sources**\n{sources}"
        if len(body) > MAX_REPLY:
            body = body[:MAX_REPLY].rsplit("\n", 1)[0] + "\n…"
        return body

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or self.user is None:
            return
        if self.user not in message.mentions:
            return
        question = message.clean_content.replace(f"@{self.user.display_name}", "")
        async with message.channel.typing():
            reply = await self.answer(question)
        await message.reply(reply, mention_author=False)


def build_client() -> PiEEGBot:
    api_key = config.require_env("ANTHROPIC_API_KEY")
    client = PiEEGBot(api_key)

    @client.tree.command(name="ask", description="Ask a question about PiEEG.")
    @app_commands.describe(question="What do you want to know?")
    async def ask(interaction: discord.Interaction, question: str) -> None:
        await interaction.response.defer(thinking=True)
        reply = await client.answer(question)
        await interaction.followup.send(reply)

    return client


def main() -> None:
    token = config.require_env("DISCORD_TOKEN")
    client = build_client()
    client.run(token)


if __name__ == "__main__":
    main()
