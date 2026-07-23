"""Command-line entry point: ``pieeg-bot {ingest,run,ask,chat}``.

``ask`` and ``chat`` exercise the full RAG pipeline locally — no Discord token
required — which makes them the fastest way to sanity-check the index, tune
retrieval, or demo the bot.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from . import config


def _render(answer, show_scores: bool) -> str:
    """Format an :class:`~pieeg_bot.engine.Answer` for the terminal."""
    lines = [answer.reply]
    if answer.results:
        lines.append("")
        lines.append("Sources:")
        seen: set[str] = set()
        for r in answer.results:
            if r.url in seen:
                continue
            seen.add(r.url)
            title = r.title or r.source
            if show_scores:
                lines.append(f"  [{r.score:.2f}] {title} — {r.url}")
            else:
                lines.append(f"  • {title} — {r.url}")
    return "\n".join(lines)


async def _ask(question: str, show_scores: bool) -> None:
    from .engine import Engine

    engine = Engine.create()
    try:
        await engine.warm()
        answer = await engine.answer(question)
        print(_render(answer, show_scores))
    finally:
        await engine.close()


async def _chat(show_scores: bool) -> None:
    from .engine import Engine

    print("PiEEG-bot — ask me about PiEEG. Ctrl-D or 'exit' to quit.")
    print("Loading model…", flush=True)
    engine = Engine.create()
    try:
        await engine.warm()
        while True:
            try:
                question = input("\nyou › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", ":q"}:
                break
            start = time.perf_counter()
            answer = await engine.answer(question)
            elapsed = time.perf_counter() - start
            print(f"\nbot › {_render(answer, show_scores)}")
            print(f"\n({elapsed:.1f}s)")
    finally:
        await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="pieeg-bot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Clone the docs, chunk, embed and build the index.")
    sub.add_parser("run", help="Start the Discord bot using the built index.")

    p_ask = sub.add_parser(
        "ask", help="Ask one question and print the answer (no Discord)."
    )
    p_ask.add_argument("question", nargs="+", help="The question to ask.")
    p_ask.add_argument(
        "--scores", action="store_true", help="Show retrieval scores for each source."
    )

    p_chat = sub.add_parser(
        "chat", help="Interactive REPL to chat with the bot (no Discord)."
    )
    p_chat.add_argument(
        "--scores", action="store_true", help="Show retrieval scores for each source."
    )

    args = parser.parse_args()
    # Pick up ANTHROPIC_API_KEY etc. from a local .env for every command.
    config.load_env()

    if args.command == "ingest":
        from .ingest import main as ingest_main

        ingest_main()
    elif args.command == "run":
        from .bot import main as bot_main

        bot_main()
    elif args.command == "ask":
        asyncio.run(_ask(" ".join(args.question), args.scores))
    elif args.command == "chat":
        asyncio.run(_chat(args.scores))


if __name__ == "__main__":
    main()
