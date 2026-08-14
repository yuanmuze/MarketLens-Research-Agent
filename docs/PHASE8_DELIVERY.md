# Phase 8 Delivery: Real pgvector, Frozen Evaluation, and Local Load Evidence

## Outcome

Phase 8 closes the gap between an in-memory retrieval demo and a reproducible
PostgreSQL/pgvector system. The API now selects semantic storage explicitly,
fails readiness when the configured backend is unavailable or incomplete, and
uses the same embedding model and retrieval contract across memory and
pgvector. Migrations 0001-0004 remain unchanged.

This is an engineering portfolio result, not a production certification. The
load evidence is local Docker/CPU evidence; no distributed deployment,
autoscaling, production traffic, or external LLM was tested.

## Runtime path

```text
HTTP request
  -> FastAPI validation / request ID
  -> RetrievalService
     -> BM25
     -> memory embeddings OR PostgreSQL exact cosine (pgvector)
     -> Reciprocal Rank Fusion
     -> optional CPU CrossEncoder rerank
  -> evidence-backed response
  -> PostgreSQL observability records
```

The Docker runtime uses a non-root user, CPU-only PyTorch, pinned offline model
snapshots, a read-only data mount, and a separate writable reproducible cache
mount. `/health/ready` reports the actual catalog backend, semantic backend,
embedding model/dimension, and indexed coverage.

## Frozen retrieval evidence

Two independent public e-commerce datasets were used without mixing labels or
query splits:

| Dataset | Fixed test | Corpus | Best nDCG@10 | Backend parity | Failures |
|---|---:|---:|---:|---:|---:|
| WANDS | 96 queries | 42,994 products | 0.72505 (quality pgvector) | 96/96 exact | 0 |
| ESCI US reduced subset | 100 queries | 10,346 products | 0.71306 (quality pgvector) | 100/100 exact | 0 |

WANDS had been evaluated in an earlier phase, so its Phase 8 split is frozen
but not a historically untouched holdout. ESCI used the official train/test
boundary and independently selected 300 train, 100 validation, and 100 test
queries with seed `20260814`. Test qrels were opened only after the evaluation
configuration was frozen. ESCI numbers describe this fixed subset, not the full
official benchmark.

The exact memory/pgvector ranking matches demonstrate implementation parity for
the evaluated corpora and queries. Migration 0002 does define HNSW, but a
post-run `EXPLAIN` audit on both frozen databases proved that the active
deterministic two-key query used `Seq Scan + Sort`; these measurements are exact.
An approximate HNSW path was not exposed or evaluated separately.

## Local Docker load evidence

The final image ID is
`sha256:0e27929407bf23f45ad744c2fae48edc526510295e8381309b6ce3c19e762e00`
(612,818,126 bytes). The first formal matrix sent 800 measured requests plus
separate warmups to the real Docker HTTP API:

- memory Hybrid and pgvector Hybrid at concurrency 1 and 10;
- pgvector quality/rerank at concurrency 1 and 10;
- deterministic Fake-Agent with pgvector Hybrid at concurrency 1 and 10.

All 800 requests returned HTTP 200 and passed response-level checks: non-empty
retrieval, expected strategy/mode, non-degraded Agent, and at least one tool
call. There were no 4xx, 5xx, transport failures, invalid payloads, memory
fallbacks, or external LLM calls.

At concurrency 10, pgvector Hybrid sustained 32.88 requests/second with p95
329.25 ms; Fake-Agent sustained 19.78 requests/second with p95 666.91 ms. CPU
quality reranking sustained 0.88 requests/second with p95 12.49 seconds,
identifying the CrossEncoder as the clear local throughput bottleneck. These
first results were preserved without tuning or retesting.

## Reproduction boundaries

Tracked manifests preserve official source commit, file hashes, subset hashes,
model revisions, split policy, and evaluator configuration. Raw parquet,
derived query/qrel data, model caches, embedding arrays, local settings, and
detailed per-query runs remain ignored. The download helper accepts only the
pinned official files, verifies LFS size/SHA/schema, resumes through `.part`,
and atomically publishes validated files.

See `reports/phase8_progress.md` for detailed metrics and hashes,
`reports/phase8_load_test_results.json` for the HTTP matrix, and the two Phase
8 run manifests for frozen evaluation configuration.

## Interview-ready summary

- Built an explicit memory/pgvector semantic backend boundary and injected it
  through retrieval, Hybrid RRF, reranking, API startup, and readiness.
- Implemented atomic, idempotent 384-dimensional indexing with finite/shape
  validation and float32-safe change detection.
- Proved exact ranking parity between memory and PostgreSQL cosine retrieval on
  196 frozen queries across two independent datasets.
- Designed leakage-aware, hash-manifested evaluation workflows and preserved
  first-run results, failures, limitations, and non-applicable claims.
- Reduced the CPU Docker image to about 613 MB, pinned offline models, ran as
  non-root, and measured 800 validated HTTP requests without an external LLM.

The main trade-off is deliberate: the active exact pgvector plan maximizes
parity and auditability at these corpus sizes, while CPU CrossEncoder improves
nDCG but sharply limits concurrent throughput. A later production design would
benchmark approximate indexes and decouple/batch reranking under a new,
explicit migration and evaluation plan.

The final local quality gate completed with 437 tests passed and one expected
skip; Ruff, mypy across 59 source/script files, and `git diff --check` passed.
