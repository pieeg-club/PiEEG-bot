"""HTTP API and Discord bot running over one shared answer engine."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from . import config
from .bot import build_client
from .engine import Engine


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceResponse(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceResponse]


def create_app(
    engine: Engine | None = None,
    *,
    api_key: str | None = None,
    run_discord: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        shared_engine = engine or Engine.create()
        app.state.engine = shared_engine
        app.state.api_key = api_key or config.require_env("PIEEG_API_KEY")
        await shared_engine.warm()

        bot = None
        bot_task = None
        if run_discord:
            bot = build_client(shared_engine)
            bot_task = asyncio.create_task(
                bot.start(config.require_env("DISCORD_TOKEN")),
                name="pieeg-discord-bot",
            )

        try:
            yield
        finally:
            if bot is not None:
                await bot.close()
            if bot_task is not None:
                await asyncio.gather(bot_task, return_exceptions=True)
            await shared_engine.close()

    app = FastAPI(title="PiEEG API", version="1.0.0", lifespan=lifespan)

    def require_api_key(
        request: Request,
        supplied_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        expected_key = request.app.state.api_key
        if supplied_key is None or not secrets.compare_digest(supplied_key, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    async def ask(
        payload: AskRequest,
        request: Request,
        _: None = Depends(require_api_key),
    ) -> AskResponse:
        answer = await request.app.state.engine.answer(payload.question)
        seen: set[str] = set()
        sources = []
        for result in answer.results:
            if result.url in seen:
                continue
            seen.add(result.url)
            sources.append(
                SourceResponse(title=result.title or result.source, url=result.url)
            )
        return AskResponse(
            answer=answer.reply,
            grounded=answer.grounded,
            sources=sources,
        )

    return app


app = create_app()