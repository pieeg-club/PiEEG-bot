"""End-to-end smoke tests for the retrieval + chat pipeline.

Two layers:

* ``test_retrieval_*`` run everywhere — they embed locally with fastembed and
  exercise the in-memory cosine retriever. No API key, no network.
* ``test_live_answer_is_grounded`` runs only when ``ANTHROPIC_API_KEY`` is set.
  It drives the *real* production prompt (``bot.SYSTEM_PROMPT`` +
  ``bot._build_context``) through Claude and checks the answer is grounded in
  the supplied context.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pieeg_bot import bot, embeddings, llm
from pieeg_bot.retriever import Retriever

# A tiny stand-in corpus so tests don't depend on a built index.json.
FAKE_CHUNKS = [
    {
        "text": (
            "The PiEEG board samples 8 EEG channels at 250 samples per second "
            "using the ADS1299 analog front-end."
        ),
        "url": "https://pieeg.com/docs/hardware/devices/pieeg-8",
        "source": "PiEEG Docs",
        "title": "PiEEG-8 device",
    },
    {
        "text": (
            "To connect electrodes, snap the gold-cup leads onto the header and "
            "reference the ear clip to reduce common-mode noise."
        ),
        "url": "https://pieeg.com/docs/hardware/signal-quality/electrodes",
        "source": "PiEEG Docs",
        "title": "Electrode setup",
    },
    {
        "text": (
            "The Mediterranean diet emphasizes fish, olive oil and vegetables "
            "and is unrelated to brain-computer interfaces."
        ),
        "url": "https://example.com/diet",
        "source": "Off-topic",
        "title": "Diet",
    },
]


def _build_retriever(chunks: list[dict]) -> Retriever:
    vectors = embeddings.embed_passages([c["text"] for c in chunks])
    matrix = np.asarray(vectors, dtype=np.float32)
    return Retriever(chunks, matrix)


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return _build_retriever(FAKE_CHUNKS)


@pytest.mark.asyncio
async def test_retrieval_finds_relevant_chunk(retriever: Retriever) -> None:
    results = await retriever.search("How many EEG channels and what sample rate?")
    assert results, "expected at least one hit"
    # The channels/sample-rate chunk should rank first.
    assert "ADS1299" in results[0].text
    assert results[0].url.endswith("/pieeg-8")


@pytest.mark.asyncio
async def test_retrieval_ranks_electrodes_for_setup(retriever: Retriever) -> None:
    results = await retriever.search("how do I attach the electrodes?")
    assert results
    assert "electrodes" in results[0].url


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live Claude call",
)
async def test_live_answer_is_grounded(retriever: Retriever) -> None:
    question = "What sample rate does the PiEEG board use?"
    results = await retriever.search(question)
    context = bot._build_context(results)
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    client = llm.build_client(os.environ["ANTHROPIC_API_KEY"])
    try:
        answer = await llm.chat(client, bot.SYSTEM_PROMPT, user)
    finally:
        await client.close()

    assert answer.strip(), "expected a non-empty answer"
    # The only sample rate in context is 250 SPS — a grounded answer must use it.
    assert "250" in answer
