# Phase 7: Production Readiness Report

## 1. Commits

- Start: `1051088`
- Final: (see git log; sub-phase commits listed in Section 12)

## 2. Architecture Changes

- pgvector semantic retrieval backend (PostgreSQL) added alongside the
  in-memory backend. Hybrid RRF still combines BM25 + semantic results.
- Request idempotency (request_id + request_hash) added.
- Liveness / readiness probes added.
- Minimal feedback loop added.
- Structured logging added.
- API containerized; CI workflow added; load test script added.

## 3. pgvector Schema, Dimension, Index

- Extension: `vector` (CREATE EXTENSION IF NOT EXISTS vector)
- Table: `product_embeddings`
  - product_id FK → products (ON DELETE CASCADE)
  - model_name (String)
  - dim (Integer)
  - embedding vector(384) — matches all-MiniLM-L6-v2
  - unique (product_id, model_name)
- Index: HNSW cosine (`vector_cosine_ops`) — chosen for approximate
  nearest-neighbor at scale; cosine matches the L2-normalized dot-product
  used by the in-memory backend, preserving result consistency.

## 4. Migration Results

```
0001_initial (frozen) → 0002_add_pgvector_embeddings → 0003_add_request_hash
→ 0004_add_feedback_events
alembic upgrade head / current / check all pass; no metadata drift.
downgrade → upgrade cycle verified on marketlens_test.
```

## 5. Idempotency Semantics

- `AgentRequest.request_id` is an optional idempotency key.
- Same request_id + same content → replay stored result.
- Same request_id + different content → HTTP 409 conflict.
- No IntegrityError-as-500 fallback for idempotency.

## 6. Timeout & Error Semantics

- Unified error response: `{code, message, request_id}` (no stack/keys).
- Stable codes: internal_error, plus 404/409/422/503 for known cases.

## 7. Liveness vs Readiness

- `/health/live`: process running (no external deps).
- `/health/ready`: catalog + retrieval service usable; 503 otherwise.

## 8. Feedback Loop

- Implemented (no authoritative spec deferred it to Phase 8).
- `feedback_events` table: agent_run_id FK CASCADE, feedback_type
  (helpful/unhelpful), optional reason, idempotency_key.
- POST /feedback: 404 missing run, idempotent via key.

## 9. Docker Images & Services

- db: pgvector/pgvector:pg16 (PostgreSQL 16.14), 127.0.0.1:5432
- api: python:3.12-slim, non-root appuser, 127.0.0.1:8000, healthcheck
  on /health/live

## 10. CI Status

**CI workflow created; local equivalent commands passed; remote GitHub
Actions not run** (no push / no GitHub access).

## 11. PostgreSQL / pgvector Tests

- 24 postgres integration tests passed (not skipped): pgvector extension,
  vector(384), idempotent upsert, cosine top-k, model filter, FK cascade,
  in-memory vs pgvector overlap, idempotency, feedback, etc.

## 12. Local Load Test Configuration & Results

- Script: scripts/load_test.py (httpx, hits real Docker API)
- Dataset: fixture catalog (20 products); queries: fixed search + agent
- Machine: Docker Desktop 4.86.0, Windows host, Python 3.11 local client

| Scenario | Concurrency | Success | 4xx | 5xx | Throughput | p50 | p95 | p99 |
|----------|-------------|---------|-----|-----|------------|-----|-----|-----|
| retrieval | 10 | 100 | 0 | 0 | 10.24 rps | 969ms | 1016ms | 1016ms |
| agent (Fake) | 10 | 100 | 0 | 0 | 10.22 rps | 969ms | 1016ms | 1031ms |
| retrieval | 1 | 100 | 0 | 0 | 3.42 rps | 297ms | 313ms | 344ms |

> These are LOCAL development measurements on a laptop with the fake
> embedding backend — NOT production performance. Concurrency=10 shows
> higher latency than concurrency=1 because CPU-bound fake embedding +
> BM25 contend for a single core.

## 13. Failure Cases & Limitations

- pgvector backend is implemented but not wired into RetrievalService's
  default path (in-memory remains default). Full Hybrid-on-pgvector
  integration is a follow-up.
- CI workflow created but not executed remotely.
- Load test uses fixture catalog + fake embeddings; not representative
  of production scale.

## 14. Real LLM Used?

**No.** All tests and load tests use Fake LLM / offline paths. No real
LLM or external network calls.

## 15. New/Modified Files

- alembic/versions/0002/0003/0004
- src/marketlens/retrieval/pgvector_retriever.py
- src/marketlens/observability.py
- src/marketlens/persistence/{models,repositories}.py
- src/marketlens/agent/models.py, src/marketlens/api/{routes,main}.py
- Dockerfile, .dockerignore, compose.yaml, .github/workflows/ci.yml
- scripts/load_test.py, tests/test_pgvector.py, tests/test_observability.py

## 16. Five Interview-Relevant Concepts

1. **pgvector vs in-memory vectors**: pgvector persists embeddings in
   PostgreSQL so they survive restarts and scale beyond RAM; HNSW index
   gives approximate nearest-neighbor search.
2. **Migration immutability**: never edit a frozen migration (0001);
   always add a new one (0002+) so history stays reproducible.
3. **Liveness vs readiness**: liveness = "is the process alive";
   readiness = "can it serve traffic" (checks deps).
4. **request_id uniqueness vs idempotency**: uniqueness enforces one row
   per key; idempotency returns the same result on retry; a content hash
   distinguishes a replay from a conflict.
5. **p95 latency vs throughput**: p95 captures the worst-typical latency
   (users feel the tail); throughput is requests/sec under concurrency.

## 17. Learning Notes (concise)

- pgvector stores vectors with a distance operator (`<=>` cosine), so
  semantic search becomes a SQL query.
- Migrations are append-only; never amend a committed one.
- liveness/readiness separate "alive" from "ready to serve".
- A unique request_id + content hash gives idempotent retries + conflict
  detection.
- p95 shows tail latency; throughput shows capacity — both matter.
- CI's PostgreSQL service container gives real (not mocked) integration
  tests.
