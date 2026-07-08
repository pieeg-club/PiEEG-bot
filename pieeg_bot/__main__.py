"""Command-line entry point: ``pieeg-bot {ingest,run}``."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="pieeg-bot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Clone the docs, chunk, embed and build the index.")
    sub.add_parser("run", help="Start the Discord bot using the built index.")

    args = parser.parse_args()
    if args.command == "ingest":
        from .ingest import main as ingest_main

        ingest_main()
    elif args.command == "run":
        from .bot import main as bot_main

        bot_main()


if __name__ == "__main__":
    main()
