# MarketLens Phase 8 Final Frozen Report

## Scope and repository state

- Actual starting commit: `d5e69ae24d6073be7b9b9fcc51b045108c769ad4`.
- Phase 8 implementation-freeze commit before release audit:
  `49e86b48e88eacc063a5ac424d093d15e2215a6c`.
- Frozen migrations 0001-0004 were not modified and no migration was added.
- Real LLM/model fine-tuning: not used. Agent integration and load used a
  deterministic Fake LLM; external LLM calls were zero.
- Test-only PostgreSQL databases: `marketlens_repo_test`, `marketlens_test`,
  `marketlens_wands_test`, and `marketlens_esci_test`.

## Modified delivery files

Runtime/retrieval:

- `src/marketlens/config.py`
- `src/marketlens/retrieval/semantic.py`
- `src/marketlens/retrieval/embedding.py`
- `src/marketlens/retrieval/pgvector_retriever.py`
- `src/marketlens/retrieval/service.py`
- `src/marketlens/persistence/engine.py`
- `src/marketlens/persistence/repositories.py`
- `src/marketlens/api/database.py`
- `src/marketlens/api/main.py`
- `src/marketlens/api/routes.py`
- `alembic/env.py`
- `scripts/verify_retrieval_core.py`

Evaluation/index/load and tests:

- `scripts/download_esci.py`
- `scripts/prepare_esci_subset.py`
- `scripts/index_product_embeddings.py`
- `scripts/index_wands_embeddings.py`
- `scripts/index_esci_embeddings.py`
- `scripts/evaluate_phase8_wands.py`
- `scripts/evaluate_phase8_esci.py`
- `scripts/load_test_phase8.py`
- `tests/test_pgvector.py`
- `tests/test_embedding_indexer.py`
- `tests/test_semantic_backends.py`
- `tests/test_phase8_api.py`
- `tests/test_phase8_postgres.py`
- `tests/test_phase8_wands.py`
- `tests/test_phase8_esci.py`

Build/configuration:

- `.python-version`
- `.gitignore`
- `.dockerignore`
- `.env.example`
- `pyproject.toml`
- `Dockerfile`
- `compose.yaml`

Manifests/reports/documentation:

- `data/manifests/esci_source.json`
- `data/manifests/esci_subset.json`
- `reports/phase8_run_manifest.json`
- `reports/phase8_esci_run_manifest.json`
- `reports/phase8_load_test_results.json`
- `reports/phase8_progress.md`
- `reports/phase8_final_report.md`
- `docs/PHASE8_DELIVERY.md`
- `README.md`
- Published small pre-existing aggregate reports:
  `reports/eval_candidate_summary.md` and `reports/load_test_results.json`.

## ESCI source and fixed subset

Official source: Amazon Science
`https://github.com/amazon-science/esci-data`, Apache-2.0, commit
`7916cdf6ab75a462e77f20ab40428a10923998d5`.

| Official file | Bytes | SHA-256 | Rows |
|---|---:|---|---:|
| examples parquet | 51,286,808 | `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a` | 2,621,288 |
| products parquet | 1,108,857,465 | `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265` | 1,814,924 |

Only English-US rows with `small_version==1` were eligible. Seed `20260814`
ranked query IDs by SHA-256 within the official split: 300 train queries (5,954
judgments, 5,939 unique products), 100 validation queries from official train
(2,147 judgments, 2,143 products), and 100 queries from official test (2,315
judgments, 2,310 products). Query groups are pairwise disjoint. There were zero
missing product joins, duplicate query/product pairs, or empty titles. The
union catalog contains 10,346 products. Detailed queries/qrels remain local.

## Frozen evaluation results

WANDS first/only Phase 8 test: 96 queries, 42,994 products, seven strategies,
672 runs, zero failures. WANDS had been evaluated in Phase 4, so this split was
not historically untouched.

| WANDS strategy | Recall@10 | relevant MRR@10 | exact MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.00044 | 0.02431 | 0.00000 | 0.00789 | 0.00 | 0.00 |
| BM25 | 0.05461 | 0.82517 | 0.43051 | 0.61309 | 50.14 | 103.13 |
| Semantic memory | 0.06285 | 0.88218 | 0.43116 | 0.66662 | 58.64 | 68.24 |
| Semantic pgvector | 0.06285 | 0.88218 | 0.43116 | 0.66662 | 256.29 | 271.71 |
| Hybrid memory | 0.06001 | 0.85965 | 0.44641 | 0.66366 | 112.77 | 180.15 |
| Hybrid pgvector | 0.06001 | 0.85965 | 0.44641 | 0.66366 | 311.36 | 367.85 |
| Quality pgvector | 0.06449 | 0.90972 | 0.54203 | 0.72505 | 2171.77 | 5441.87 |

ESCI first/only official-test subset run: 100 queries, 10,346 products, seven
strategies, 700 runs, zero failures. E/S are relevant for Recall/MRR; E alone
is exact; nDCG grades E=3, S=2, C=1, I=0.

| ESCI strategy | Recall@10 | relevant MRR@10 | exact MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.00250 | 0.01000 | 0.01000 | 0.00390 | 0.00 | 0.00 |
| BM25 | 0.40173 | 0.83133 | 0.67435 | 0.62460 | 36.58 | 80.24 |
| Semantic memory | 0.43677 | 0.91093 | 0.73254 | 0.65920 | 81.95 | 91.85 |
| Semantic pgvector | 0.43677 | 0.91093 | 0.73254 | 0.65920 | 186.83 | 233.72 |
| Hybrid memory | 0.43436 | 0.88278 | 0.71836 | 0.66477 | 108.74 | 154.38 |
| Hybrid pgvector | 0.43436 | 0.88278 | 0.71836 | 0.66477 | 127.93 | 155.47 |
| Quality pgvector | 0.46589 | 0.90350 | 0.78875 | 0.71306 | 3856.74 | 4667.29 |

Semantic and Hybrid memory/pgvector rankings matched exactly on 96/96 WANDS
and 100/100 ESCI queries. Migration 0002 defines HNSW, but post-run read-only
`EXPLAIN` on both frozen databases showed the active deterministic distance +
product-ID query used `Seq Scan + Sort`; recorded results are exact.
Approximate HNSW was not run separately. This corrects the frozen ESCI
manifest's original, inaccurate reason text without changing its frozen config,
metrics, or running the test again.

ESCI's five lowest predefined quality cases were `didlos women clearance`,
`0rgans only not piano for sale`, `shelack nail polish`, `women on women toys`,
and `scotch lip balm`. They expose typo/noisy-token, negation, spelling, and
brand/category ambiguity rather than execution failures.

## Docker and load evidence

Final image:
`sha256:0e27929407bf23f45ad744c2fae48edc526510295e8381309b6ce3c19e762e00`,
612,818,126 bytes, non-root `appuser`, Torch 2.13.0+cpu, two pinned models
verified offline. The formal local matrix returned 800/800 valid HTTP 200
responses, zero 4xx/5xx/transport/payload/fallback failures, and zero external
LLM calls.

| Backend/path | Concurrency | RPS | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| memory Hybrid | 1 | 8.55 | 118.08 | 129.51 | 135.05 |
| memory Hybrid | 10 | 8.31 | 1193.22 | 1249.29 | 1493.62 |
| pgvector Hybrid | 1 | 31.77 | 30.85 | 34.96 | 40.44 |
| pgvector Hybrid | 10 | 32.88 | 291.47 | 329.25 | 331.67 |
| pgvector quality | 1 | 0.87 | 1070.58 | 1433.98 | 1526.96 |
| pgvector quality | 10 | 0.88 | 11276.00 | 12487.00 | 12495.51 |
| pgvector Fake-Agent | 1 | 17.35 | 53.72 | 71.23 | 93.93 |
| pgvector Fake-Agent | 10 | 19.78 | 473.32 | 666.91 | 670.36 |

This is a local Docker CPU measurement, not a production SLA. CPU reranking is
the observed throughput bottleneck; no result-driven tuning/retest occurred.

## Acceptance, security, and large files

- `pytest`: 437 passed, 1 expected skip.
- PostgreSQL marker: 33 passed.
- Ruff: passed. mypy: passed for 59 source/script files.
- Alembic: current at 0004; check reports no new operations after registering
  both existing declarative metadata collections in `alembic/env.py`.
- Docker Compose config/health and actual pgvector Hybrid were verified locally.
- `git diff --check`: passed; only Windows LF/CRLF notices were emitted.
- No dedicated secret scanner was installed. Read-only fallback scans found no
  high-confidence secret in delivery candidates. Historical `sk-` matches in
  upstream benchmark text were format-checked as non-secret identifiers.
- The initial object-store scan surfaced blob
  `9e4f7962d0a9cf5e454ce35719bf1d5a30bb7a1d`, 66,038,912 bytes, associated
  with local cache path `data/cache/embeddings_fea3dff39a7bb8cc.npy`.
  Repository-external backup and subsequent forensic checks proved it was
  unreachable: it appears in no commit, branch, remote ref, bundle ref, or path
  log. A normal branch push cannot transfer it.
- Because the target was not in reachable history, running `git-filter-repo`
  would have rewritten safe commits without removing anything. No history
  filter was run and the Phase 0-8 old-to-new commit map is identity.
- The public candidate history has no raw dataset, embedding/cache array,
  model weight, or credential. It has zero blobs over 50 MiB; its largest blob
  is 2,105,328 bytes.
- Verified recovery bundle:
  `<repository-parent>/MarketLens-Research-Agent-pre-public-20260814-163714.bundle`,
  7,143,843 bytes, SHA-256
  `ABF079043E00AF5ED5363308DE1F582DAFCAA78B55C0A5D792F844286E8FDD21`.
- Local-artifact backup directory:
  `<repository-parent>/MarketLens-Research-Agent-local-artifacts-20260814-163714`.
  JSON SHA-256 is
  `33AE916524B47CE7D9BAFB1AFB0BE3D08D6EB68EAC5EA1C5997FE86896318B01`;
  Markdown SHA-256 is
  `5AF9CB67B56A677431745B48CA39BCE38A073FD4AA2E6AD129C3C7F1D9CE36BC`.

Local ignored/unpublished large data and cache artifacts:

| Path | Bytes | Recommendation |
|---|---:|---|
| `data/raw/meta_Electronics.jsonl.gz` | 1,312,900,427 | retain locally; raw source, never commit |
| `data/raw/esci/...products...corrupt....part` | 1,143,973,106 | failed-download evidence; safe manual cleanup later, not automatic |
| `data/raw/esci/...products.parquet` | 1,108,857,465 | retain locally; reproducible official raw source |
| `data/external/wands/product.csv` | 90,621,131 | retain locally; benchmark raw data |
| `data/cache/embeddings_fea3dff39a7bb8cc.npy` | 66,038,912 | reproducible cache; never add again |
| `data/raw/esci/...examples.parquet` | 51,286,808 | retain locally; reproducible official raw source |
| `data/cache/embeddings_ae626a3b60be3624.npy` | 15,891,584 | reproducible ESCI cache; never commit |
| `data/cache/embeddings_949e8c4ecc6e32bb.npy` | 3,072,128 | reproducible cache; never commit |

`.claude/settings.json` is local tool configuration; `data/eval/eval_review.csv`
is preserved local human/query-detail data. Neither is published. No raw data,
derived query/qrels, model weights, `.part`, local database, cache, or absolute
personal path is in the Phase 8 candidate set.

`reports/data_quality_report.json` and `reports/data_quality_report.md` are also
preserved locally but excluded because they contain example-level product
details; they were not deleted or silently ignored.

## Publication plan and limitations

Recommended grouped commits:

1. `feat(phase-8): complete pgvector hybrid retrieval`
2. `test(phase-8): add reproducible WANDS and ESCI evaluation`
3. `build(phase-8): harden containerized model and indexing workflow`
4. `docs(phase-8): freeze evaluation and delivery report`

Remote CI status at report freeze: not yet run; it must not be described as
green until the audited commits are pushed and the actual workflow succeeds.

Known limits: fixed subsets rather than full leaderboard protocols; WANDS is
not a historically untouched holdout; active exact sequential pgvector plan
does not demonstrate HNSW recall/performance; CPU rerank concurrency is poor;
no real LLM, token/cost, authentication, distributed deployment, or production
capacity/SLA evidence exists.
