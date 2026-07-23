FROM python:3.12-slim

WORKDIR /app

# git is needed by `pieeg-bot ingest`; harmless for `run`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
# Install dependencies only (not the package itself yet)
RUN pip install --no-cache-dir discord.py>=2.3 numpy>=1.24 anthropic>=0.40 fastembed>=0.3

COPY pieeg_bot/ pieeg_bot/

# Build the docs index at image-build time. It is a derived artifact, not
# source: every build re-ingests the live docs, so each deploy ships fresh
# documentation with no committed JSON to keep in sync. Ingest is self-contained
# (git + local fastembed, no API keys), and warming the model here also makes
# the first query at runtime instant.
RUN python -c "from pieeg_bot import embeddings; embeddings.warm()" \
    && python -m pieeg_bot ingest

CMD ["python", "-m", "pieeg_bot", "run"]
