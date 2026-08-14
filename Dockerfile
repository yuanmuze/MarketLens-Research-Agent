# MarketLens API image (Python 3.12, non-root).
FROM python:3.12-slim

# Non-root user for security.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

ARG EMBEDDING_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ARG RERANKER_MODEL_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a
ENV HF_HOME=/opt/huggingface

# Keep dependency layers tied to dependency metadata, not application source.
COPY pyproject.toml uv.lock ./

# Install runtime dependencies directly (avoids building the project wheel,
# which references the tests/ directory excluded from the image).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    fastapi==0.141.1 uvicorn==0.34.3 pydantic==2.11.5 pydantic-settings==2.14.2 \
    sqlalchemy==2.0.41 alembic==1.19.1 psycopg2-binary==2.9.12 pgvector==0.5.0 \
    langchain-core==1.4.8 langchain==1.3.9 langgraph==1.2.9 httpx==0.28.1 \
    numpy==2.3.0

# PyPI's Linux torch 2.13.0 wheel pulls the CUDA 13 dependency family. Install
# the same upstream release from PyTorch's official CPU wheel index first.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install sentence-transformers==5.7.0

# Freeze the exact public model snapshots used by the existing 384-dim cache
# and quality reranker. Only PyTorch runtime files are included; the repositories
# also contain unused ONNX, OpenVINO, TensorFlow, Rust, and Flax weights.
RUN python -c "from huggingface_hub import snapshot_download; repo='sentence-transformers/all-MiniLM-L6-v2'; rev='${EMBEDDING_MODEL_REVISION}'; print(f'Downloading {repo}@{rev} to {__import__(\"os\").environ[\"HF_HOME\"]}', flush=True); path=snapshot_download(repo_id=repo, revision=rev, allow_patterns=['config.json','config_sentence_transformers.json','modules.json','sentence_bert_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors','1_Pooling/config.json']); print(f'Completed {repo}: {path}', flush=True)" \
    && mkdir -p /opt/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs \
    && printf '%s' "$EMBEDDING_MODEL_REVISION" > /opt/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main \
    && chown -R appuser:appuser /opt/huggingface

RUN python -c "from huggingface_hub import snapshot_download; repo='cross-encoder/ms-marco-MiniLM-L-6-v2'; rev='${RERANKER_MODEL_REVISION}'; print(f'Downloading {repo}@{rev} to {__import__(\"os\").environ[\"HF_HOME\"]}', flush=True); path=snapshot_download(repo_id=repo, revision=rev, allow_patterns=['config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors']); print(f'Completed {repo}: {path}', flush=True)" \
    && mkdir -p /opt/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/refs \
    && printf '%s' "$RERANKER_MODEL_REVISION" > /opt/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/refs/main \
    && chown -R appuser:appuser /opt/huggingface

# Copy frequently changing application source only after dependencies and models.
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./
COPY scripts/ scripts/

# marketlens package lives under src/
ENV PYTHONPATH=/app/src
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

# Switch to non-root
USER appuser

EXPOSE 8000

CMD ["uvicorn", "marketlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
