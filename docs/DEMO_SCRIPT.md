# MarketLens demo script

This walkthrough uses Python 3.12 and the deterministic Fake LLM. It requires
no provider API key and makes no external LLM call.

## 1. Clone and install

```bash
git clone https://github.com/yuanmuze/MarketLens-Research-Agent.git
cd MarketLens-Research-Agent
uv sync --python 3.12 --locked --extra dev --extra db --extra embeddings
```

## 2. Verify the offline application

```bash
uv run pytest -m "not postgres" -q -rs
uv run ruff check .
uv run mypy src scripts
```

Phase 8's frozen release evidence is 437 passed with one expected skip. Test
counts may grow after hardening; an unexpected skip or reduced coverage is not
accepted.

## 3. Start the local API

```bash
uv run uvicorn marketlens.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. The OpenAPI application exposes 10 routes:
health, liveness, readiness, search, synchronous research, async job creation,
job status, job report, agent recommendation, and feedback.

## 4. Search and health checks

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
curl "http://127.0.0.1:8000/search?q=wireless+headphones&strategy=hybrid&top_k=5"
```

Point out the request ID, ranked products, score source, and readiness backend.

## 5. Run the offline agent

```bash
curl -X POST http://127.0.0.1:8000/agent/recommend \
  -H "Content-Type: application/json" \
  -d '{"message":"best wireless headphones under $200","mode":"balanced","max_results":3}'
```

The default `MARKETLENS_USE_FAKE_LLM=true` uses a deterministic local client.
It demonstrates orchestration and persistence without presenting Fake LLM text
as real-provider quality evidence.

## 6. Run the Compose stack

```bash
docker compose -f compose.yaml config
docker compose -f compose.yaml up -d --build
docker compose -f compose.yaml ps
```

The Compose profile uses PostgreSQL/pgvector, a real local embedding model, and
the Fake LLM. Before pgvector readiness, import the catalog and build the index
as described in the README.

## 7. Evidence to present

- 437 passed and 1 expected skip in the Phase 8 frozen offline gate.
- 33 PostgreSQL integration tests passed.
- WANDS memory/pgvector parity: 96/96 queries.
- ESCI memory/pgvector parity: 100/100 queries.
- Local Docker load matrix: 800/800 successful requests.
- External LLM calls: 0.

Historical Phase 6 fixture benchmarks remain in their phase reports and must
not be substituted for these Phase 8 frozen results.

## Talking points

1. Hard constraints and evidence validation stay in deterministic code.
2. BM25 and semantic results are fused through reciprocal-rank fusion.
3. pgvector readiness fails explicitly instead of silently switching backend.
4. API errors and stored failures are sanitized before crossing trust boundaries.
5. Open Deep Research attribution is retained while MarketLens has a distinct,
   tested FastAPI entry point.
