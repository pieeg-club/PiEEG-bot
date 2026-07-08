"""PiEEG-bot: a Discord documentation companion for the PiEEG / BCI community.

The package is intentionally small and has two entry points:

* ``pieeg-bot ingest`` — clone the documentation repositories, chunk the
  markdown, embed it and write ``data/index.json``.
* ``pieeg-bot run`` — start the Discord bot, which answers questions from the
  pre-built index.

Everything is deliberately kept simple: there is no vector database. The corpus
is a few dozen markdown files, so the index lives in a single JSON file that is
loaded into memory and searched with plain cosine similarity.
"""

__version__ = "0.1.0"
