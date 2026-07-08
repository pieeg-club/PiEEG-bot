FROM python:3.12-slim

WORKDIR /app

# git is needed by `pieeg-bot ingest`; harmless for `run`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY pieeg_bot/ pieeg_bot/
# The index is committed by the nightly GitHub Action and baked into the image.
COPY data/ data/

RUN pip install --no-cache-dir .

# Pre-download the fastembed model so the first query at runtime is instant.
RUN python -c "from pieeg_bot import embeddings; embeddings.warm()"

CMD ["pieeg-bot", "run"]
