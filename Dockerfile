FROM python:3.12-slim

WORKDIR /app

# git is needed by `pieeg-bot ingest`; harmless for `run`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
# Install dependencies only (not the package itself yet)
RUN pip install --no-cache-dir discord.py>=2.3 numpy>=1.24 anthropic>=0.40 fastembed>=0.3 fastapi>=0.115 uvicorn>=0.30

COPY pieeg_bot/ pieeg_bot/
# The index is committed and baked into the image.
COPY data/ data/

# Pre-download the fastembed model so the first query at runtime is instant.
RUN python -c "from pieeg_bot import embeddings; embeddings.warm()"

CMD ["uvicorn", "pieeg_bot.api:app", "--host", "0.0.0.0", "--port", "8080"]
