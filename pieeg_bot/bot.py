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

import discord
from discord import app_commands

from . import config
from .engine import Engine


class PiEEGBot(discord.Client):
    def __init__(self, api_key: str) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self._engine = Engine(api_key)

    async def setup_hook(self) -> None:
        # Load the embedding model up front so the first query isn't slow.
        await self._engine.warm()
        await self.tree.sync()

    async def close(self) -> None:
        await self._engine.close()
        await super().close()

    async def answer(self, question: str) -> str:
        """Retrieve, generate and format a grounded answer for ``question``."""
        answer = await self._engine.answer(question)
        return answer.to_discord()

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
