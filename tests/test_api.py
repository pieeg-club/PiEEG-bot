from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from pieeg_bot import api as api_module
from pieeg_bot.api import create_app
from pieeg_bot.engine import Answer
from pieeg_bot.retriever import Retrieved


class FakeEngine:
    def __init__(self) -> None:
        self.warm_calls = 0
        self.closed = False

    async def warm(self) -> None:
        self.warm_calls += 1

    async def close(self) -> None:
        self.closed = True

    async def answer(self, question: str) -> Answer:
        return Answer(
            reply=f"Answer to: {question}",
            results=[
                Retrieved(
                    text="PiEEG documentation",
                    url="https://docs.pieeg.com/hardware",
                    source="PiEEG Docs",
                    title="Hardware",
                    score=0.9,
                )
            ],
        )


def build_test_client() -> TestClient:
    app = create_app(FakeEngine(), api_key="test-secret", run_discord=False)
    return TestClient(app)


def test_health_is_public() -> None:
    with build_test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_requires_api_key() -> None:
    with build_test_client() as client:
        response = client.post("/ask", json={"question": "What is PiEEG?"})

    assert response.status_code == 401


def test_ask_returns_structured_answer() -> None:
    with build_test_client() as client:
        response = client.post(
            "/ask",
            headers={"X-API-Key": "test-secret"},
            json={"question": "What is PiEEG?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer to: What is PiEEG?",
        "grounded": True,
        "sources": [
            {
                "title": "Hardware",
                "url": "https://docs.pieeg.com/hardware",
            }
        ],
    }


def test_rest_and_discord_share_engine(monkeypatch) -> None:
    engine = FakeEngine()

    class FakeBot:
        def __init__(self) -> None:
            self.engine = None
            self.started_with = None
            self.closed = False
            self._stop: asyncio.Event | None = None

        async def start(self, token: str) -> None:
            self.started_with = token
            self._stop = asyncio.Event()
            await self._stop.wait()

        async def close(self) -> None:
            self.closed = True
            if self._stop is not None:
                self._stop.set()

    bot = FakeBot()

    def build_client(shared_engine):
        bot.engine = shared_engine
        return bot

    monkeypatch.setattr(api_module, "build_client", build_client)
    monkeypatch.setattr(api_module.config, "require_env", lambda name: "discord-token")

    app = create_app(engine, api_key="test-secret", run_discord=True)
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            headers={"X-API-Key": "test-secret"},
            json={"question": "What is PiEEG?"},
        )

        assert response.status_code == 200
        assert bot.engine is engine
        assert bot.started_with == "discord-token"
        assert engine.warm_calls == 1

    assert bot.closed
    assert engine.closed