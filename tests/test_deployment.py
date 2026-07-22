"""Test the deployed bot on Fly.io.

This script verifies the production Discord bot is online and responding
correctly. It requires the DISCORD_TOKEN from your .env file.

Usage:
    python -m pytest tests/test_deployment.py -v
    # or run directly:
    python tests/test_deployment.py

What it tests:
    1. Bot application is reachable via Discord API
    2. Bot user exists and has correct name/avatar
    3. Bot responds to a test question (requires TEST_CHANNEL_ID)

Setup:
    1. Create a test Discord server or use an existing one
    2. Invite the bot using the OAuth2 URL from Discord Developer Portal:
       https://discord.com/developers/applications
       - Go to OAuth2 → URL Generator
       - Select scopes: bot, applications.commands
       - Select permissions: Send Messages, Use Slash Commands
       - Copy the generated URL and open it in a browser
    3. Copy a channel ID from your test server:
       - Enable Developer Mode (Settings → Advanced → Developer Mode)
       - Right-click a channel → Copy ID
    4. Set TEST_CHANNEL_ID in your .env file or environment

Environment variables:
    DISCORD_TOKEN: Bot token (required, from Discord Developer Portal)
    TEST_CHANNEL_ID: Channel ID for live message test (optional)
"""

from __future__ import annotations

import asyncio
import os

import discord
import pytest


@pytest.fixture
def token() -> str:
    """Discord bot token from .env."""
    token = os.environ.get("DISCORD_TOKEN")
    if not token or token == "...":
        pytest.skip("DISCORD_TOKEN not set — skipping deployment tests")
    return token


@pytest.fixture
def test_channel_id() -> int | None:
    """Optional test channel ID for live message verification."""
    channel_id = os.environ.get("TEST_CHANNEL_ID")
    return int(channel_id) if channel_id else None


@pytest.mark.asyncio
async def test_bot_is_reachable(token: str) -> None:
    """Verify the bot application is reachable via Discord API."""
    client = discord.Client(intents=discord.Intents.default())
    try:
        await client.login(token)
        app_info = await client.application_info()
        assert app_info.bot_public, "Bot should be public"
        print(f"\n✓ Bot application: {app_info.name} (ID: {app_info.id})")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_bot_user_exists(token: str) -> None:
    """Verify the bot user exists and has expected properties."""
    client = discord.Client(intents=discord.Intents.default())
    try:
        await client.login(token)
        user = await client.fetch_user(client.application_id)
        assert user.bot, "User should be marked as a bot"
        assert "pieeg" in user.name.lower(), f"Expected 'PiEEG' in name, got {user.name!r}"
        print(f"\n✓ Bot user: {user.name}#{user.discriminator} (ID: {user.id})")
        if user.avatar:
            print(f"  Avatar: {user.avatar.url}")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_bot_responds_to_mention(token: str, test_channel_id: int | None) -> None:
    """Send a test message and verify the bot responds.
    
    Requires TEST_CHANNEL_ID to be set. Skips if not configured.
    """
    if not test_channel_id:
        pytest.skip("TEST_CHANNEL_ID not set — skipping live message test")

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    try:
        await client.login(token)
        await client.connect()
        
        # Wait for bot to be ready
        await client.wait_until_ready()
        
        channel = client.get_channel(test_channel_id)
        if not isinstance(channel, discord.TextChannel):
            pytest.skip(f"Channel {test_channel_id} not found or not a text channel")

        # Get bot user
        bot_app = await client.application_info()
        bot_user = await client.fetch_user(bot_app.id)
        
        # Send test question mentioning the bot
        test_question = f"<@{bot_user.id}> What is PiEEG?"
        sent_msg = await channel.send(test_question)
        print(f"\n✓ Sent test message: {test_question}")

        # Wait for bot response (timeout after 30 seconds)
        def check(msg: discord.Message) -> bool:
            return (
                msg.author.id == bot_user.id
                and msg.reference is not None
                and msg.reference.message_id == sent_msg.id
            )

        try:
            response = await client.wait_for("message", check=check, timeout=30.0)
            assert response.content, "Bot response should not be empty"
            assert "**Sources**" in response.content, "Response should include source citations"
            print(f"✓ Bot responded: {response.content[:100]}...")
        except asyncio.TimeoutError:
            pytest.fail("Bot did not respond within 30 seconds")

    finally:
        await client.close()


@pytest.mark.asyncio
async def test_slash_command_exists(token: str) -> None:
    """Verify the /ask slash command is registered."""
    client = discord.Client(intents=discord.Intents.default())
    try:
        await client.login(token)
        commands = await client.http.get_global_commands(client.application_id)
        command_names = [cmd["name"] for cmd in commands]
        assert "ask" in command_names, f"Expected /ask command, got {command_names}"
        
        ask_cmd = next(cmd for cmd in commands if cmd["name"] == "ask")
        assert ask_cmd["description"] == "Ask a question about PiEEG."
        assert len(ask_cmd["options"]) == 1
        assert ask_cmd["options"][0]["name"] == "question"
        
        print(f"\n✓ Slash command /ask is registered")
        print(f"  Description: {ask_cmd['description']}")
    finally:
        await client.close()


async def _main() -> None:
    """Run all tests directly (without pytest)."""
    from conftest import _load_env
    _load_env()
    
    token = os.environ.get("DISCORD_TOKEN")
    if not token or token == "...":
        print("❌ DISCORD_TOKEN not set in .env")
        print("   Add your bot token to .env and try again.")
        return
    
    test_channel_id = os.environ.get("TEST_CHANNEL_ID")
    channel_id = int(test_channel_id) if test_channel_id else None
    
    print("Testing deployed PiEEG bot...\n")
    
    try:
        await test_bot_is_reachable(token)
        await test_bot_user_exists(token)
        await test_slash_command_exists(token)
        
        if channel_id:
            await test_bot_responds_to_mention(token, channel_id)
        else:
            print("\n⚠ TEST_CHANNEL_ID not set — skipping live message test")
            print("  To test bot responses:")
            print("  1. Create a test Discord server")
            print("  2. Invite the bot (see script docstring)")
            print("  3. Add TEST_CHANNEL_ID=<channel_id> to .env")
        
        print("\n✅ All deployment tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(_main())
