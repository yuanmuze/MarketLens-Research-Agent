# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM python:3.12-slim AS builder
WORKDIR /app
COPY --from=uv /uv /bin/uv

ARG EMBEDDING_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ARG RERANKER_MODEL_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a
ENV HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Resolve runtime packages exclusively from the committed lockfile. The
# embeddings extra pins CPU-only PyTorch through the official index declared in
# pyproject.toml, avoiding CUDA dependencies in the API image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra embeddings --no-install-project

# Cache only the PyTorch model artifacts used by the supported retrieval path.
RUN python -c "from huggingface_hub import snapshot_download; repo='sentence-transformers/all-MiniLM-L6-v2'; rev='${EMBEDDING_MODEL_REVISION}'; snapshot_download(repo_id=repo, revision=rev, allow_patterns=['config.json','config_sentence_transformers.json','modules.json','sentence_bert_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors','1_Pooling/config.json'])" \
    && mkdir -p /opt/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs \
    && printf '%s' "$EMBEDDING_MODEL_REVISION" > /opt/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main

RUN python -c "from huggingface_hub import snapshot_download; repo='cross-encoder/ms-marco-MiniLM-L-6-v2'; rev='${RERANKER_MODEL_REVISION}'; snapshot_download(repo_id=repo, revision=rev, allow_patterns=['config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors'])" \
    && mkdir -p /opt/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/refs \
    && printf '%s' "$RERANKER_MODEL_REVISION" > /opt/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/refs/main

# Install the frequently changing application only after dependencies and model
# snapshots, so source edits do not invalidate those expensive layers.
COPY README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra embeddings --no-editable

FROM python:3.12-slim
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /opt/huggingface /opt/huggingface
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser scripts/ scripts/

ENV HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

USER appuser
EXPOSE 8000
CMD ["uvicorn", "marketlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
