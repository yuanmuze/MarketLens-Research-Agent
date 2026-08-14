# MarketLens Phase 8 Progress

## Current sub-phase

Final delivery audit after completing the real pgvector/API/Docker loop, frozen
WANDS and ESCI evaluations, and the first formal local Docker HTTP load matrix.

## Completed

- Completed takeover audit at baseline `d5e69ae24d6073be7b9b9fcc51b045108c769ad4`.
- Confirmed the existing `.venv` references an unavailable Python 3.11 installation.
- Confirmed the default global uv cache is not accessible in this environment.
- Added precise Git ignore rules for the project `.venv` and Phase 8 uv cache.
- Installed project-local CPython 3.12.13 under an ignored directory without
  changing the system Python, PATH, or registry.
- Rebuilt only the broken repository `.venv`.
- Synced the existing lock file with the development, database, and embedding
  extras; `uv.lock` was not regenerated.
- Added a minimal `SemanticRetriever` protocol and structured backend status.
- Routed the active `RetrievalService` embedding, Hybrid RRF, and rerank paths
  through the selected memory or pgvector semantic retriever.
- Added startup validation for backend selection, PostgreSQL availability,
  model name, 384-dimensional pgvector queries, and complete catalog coverage.
- Pushed structured-filter candidate IDs into both memory and pgvector semantic
  search and added deterministic product-ID tie breaking.
- Preserved the active BM25 and RRF implementation and memory default behavior.
- Fixed fake embedding injection so rerank does not attempt to download/load a
  real CrossEncoder during offline tests.
- Added and verified the explicit cache-to-pgvector index command against the
  real 2,000 x 384 cache.
- Added API/configuration coverage (15 focused tests passed).
- Added real PostgreSQL/pgvector service coverage (4 focused tests passed),
  including SQL cosine retrieval, memory/pgvector fixture parity, the real
  CrossEncoder quality path, and a Fake LLM agent run with persisted AgentRun
  and ToolCall records.
- Imported 2,000 products and indexed 2,000 embeddings in `marketlens_test`.
  The repeat full index is idempotent (`unchanged=2000`, total remains 2,000).
- Changed embedding upsert comparison to shape/finite-value validation plus
  float32-tolerant comparison (`rtol=1e-6`, `atol=1e-7`). Tiny serialization
  noise is unchanged, a material `1e-4` change updates, and NaN/infinity are
  rejected. The 2,000-row repeat remains `unchanged=2000`.

## Docker build interruption audit (2026-08-14)

- BuildKit layer timestamps show the build started at 12:44:23 Asia/Shanghai
  and produced `marketlens-research-agent-api:latest` at 12:53:50 (about
  9 minutes 27 seconds of recorded layer activity).
- The foreground build client was safely terminated after roughly 15 minutes
  without incremental output. No Docker service, database container, volume,
  image, or BuildKit cache was stopped or removed. Subsequent inspection proved
  that BuildKit had completed the image before the client was terminated.
- Builder: Docker Engine 29.7.2, Docker Desktop 4.86.0, BuildKit v0.32.2 on the
  running `desktop-linux` builder.
- Build cache is present: 13.14 GB total; no prune or no-cache build was run.
- The effective local build context is approximately 2.51 MB. The current
  `.dockerignore` already excludes all `data/`, reports, tests, `.git`, and
  `.venv`, but still needs precise entries for other local-only caches and
  configuration.
- Host free space at diagnosis: C: 127.57 GB; D: 56.68 GB. Docker reports
  13.14 GB of images and 13.14 GB of build cache, so host disk exhaustion is
  not the demonstrated cause.
- The time-consuming layers were the single Python dependency install
  (12:44:24-12:52:13; history size 5.49 GB) and the combined two-model snapshot
  download (12:52:13-12:53:50; history size 1.78 GB). Source was copied before
  both expensive layers, so ordinary source changes invalidate them.
- The completed image ID is
  `sha256:1bd642a2b2b0e4c5c2338288e9a4afad9fc4cde4e12f4a8f8da75abf28216700`.
  Inspect reports 4,597,272,140 bytes; Docker's storage view reports about
  12 GB disk usage / 4.6 GB content size. It is configured for non-root user
  `appuser`.
- The currently running API container still uses the previous image
  (`sha256:2793efd...`), was created before this build, and has accumulated 54
  restarts. It and the database were healthy at diagnosis; the database volume
  was left untouched.
- Offline image inspection found `torch==2.13.0+cu130` plus 15 NVIDIA/CUDA
  distributions. This is an unintended GPU dependency chain for the CPU API
  and explains most of the abnormal image size.
- Both requested fixed model snapshot directories exist at their pinned commit
  IDs, but an isolated `--network none` load by canonical model name failed:
  the hash-pinned downloads did not create an offline-resolvable `main` ref.
  Therefore this image is built but is **not accepted** as the Phase 8 runtime.
- Current judgment: there was no stalled BuildKit layer; the apparent hang was
  buffered foreground output. The evidence supports a minimal Dockerfile fix:
  stable dependency/model layers before source, BuildKit package cache mounts,
  the same-version official CPU PyTorch wheel, separately observable pinned
  model downloads, and explicit offline snapshot paths or refs.

## Docker remediation and acceptance (2026-08-14)

- Reordered stable dependency/model layers before source, enabled BuildKit pip
  cache mounts, installed `torch==2.13.0+cpu`, separated fixed model snapshots,
  copied only runtime model artifacts, and created exact offline `refs/main`.
- Added precise build-context exclusions for local Phase 8 caches and tooling;
  no cache, report, label file, database, Docker volume, or image was deleted.
- Optimized build completed in about 2 minutes 28 seconds. Context was 1.24 MB;
  accepted image is
  `sha256:8a4ade30d51e0d8a9ff86f7f46a84b3d5051b638c83f4ece3d1753089e3cf8b5`
  and is 612,762,481 bytes by image inspect.
- Isolated `--network none --read-only` checks passed as UID 1000 with Python
  3.12.14, Torch 2.13.0+cpu, no CUDA/NVIDIA distributions,
  sentence-transformers 5.7.0, 384-dimensional embeddings, and a successful
  CrossEncoder prediction from the pinned local snapshots.
- Compose API uses the accepted image and real PostgreSQL/pgvector backends.
  Liveness/readiness, semantic, Hybrid, quality rerank, and a persisted FakeLLM
  agent/tool-call integration passed against the 2,000-product test catalog.
- The pre-existing `marketlens_test` emits a glibc collation version warning.
  No `ALTER DATABASE` or collation refresh was run. New isolated Phase 8 test
  databases use `template0`, UTF-8, and `C` locale.

## Frozen WANDS pgvector evaluation preparation (2026-08-14)

- Added `reports/phase8_run_manifest.json`, revision 2, config hash
  `69967c7a920f591d06a8409345fa24c8358b07e4f429d93427b9e19cce2cc356`.
  It freezes source hashes, seed 42, deterministic 288/96/96 split, BM25/RRF,
  top-k/candidate-k, exact model revisions, CPU/offline runtime, and evaluation
  policy before any Phase 8 test-split run.
- The manifest explicitly discloses that Phase 4 evaluated all 480 WANDS
  queries. Phase 8 does not tune on its frozen test split, but it is not a
  historically untouched holdout.
- Added a WANDS cache-to-pgvector indexer that never opens query/label data.
  Validated 42,994 ordered unique products and the paired 42,994x384 float32,
  finite cache against model/source metadata and the cache fingerprint.
- Created isolated `marketlens_wands_test` from `template0` with `C/C` locale
  and applied existing migrations only through 0004. Migrations 0001-0004 were
  not edited.
- The single atomic WANDS index run completed in 651.4 seconds: 42,994 products
  inserted and 42,994 embeddings inserted. Independent SQL verification found
  model `all-MiniLM-L6-v2`, dimension 384, count 42,994, migration 0004.
- Added a WANDS evaluator with validation-only limits, a hard prohibition on
  limiting test, refusal to overwrite a non-empty output directory, warm-up
  before qrels loading, preserved failures, and explicit not-run/not-applicable
  labels for real-LLM and pure-retrieval metrics.
- The first two-query validation smoke exposed optional Hugging Face HEAD
  retries despite local weights; it was stopped without evaluation output or
  database writes after preserving the negative log. Revision 2 makes offline
  loading explicit before any test run.
- The repeated two-query validation smoke completed in 93.1 seconds with no
  network attempts and no failures across popularity, BM25, semantic memory,
  semantic pgvector, Hybrid memory, Hybrid pgvector, and quality pgvector.
  Memory/pgvector semantic and Hybrid metrics matched on the smoke queries.
- Removed the shared Phase 4 `recall_at_50` output from this top-10-only run to
  prevent a mislabeled metric; Phase 8 reports the actually computed Recall@10.

## Frozen WANDS test result (first and only run)

- The final test run completed successfully in 376.5 seconds. It evaluated 96
  unique frozen test queries across 7 strategies, produced 672 run records, and
  recorded zero failures. No rerun occurred.
- Semantic memory and semantic pgvector had zero ranking mismatches across all
  96 queries. Hybrid memory and Hybrid pgvector also had zero ranking
  mismatches, proving backend result parity for this frozen corpus/query set.
- Final test metrics:

| Strategy | Recall@10 | Relevant MRR@10 | Exact MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.00044 | 0.02431 | 0.00000 | 0.00789 | 0.00 | 0.00 |
| BM25 | 0.05461 | 0.82517 | 0.43051 | 0.61309 | 50.14 | 103.13 |
| Semantic memory | 0.06285 | 0.88218 | 0.43116 | 0.66662 | 58.64 | 68.24 |
| Semantic pgvector | 0.06285 | 0.88218 | 0.43116 | 0.66662 | 256.29 | 271.71 |
| Hybrid memory | 0.06001 | 0.85965 | 0.44641 | 0.66366 | 112.77 | 180.15 |
| Hybrid pgvector | 0.06001 | 0.85965 | 0.44641 | 0.66366 | 311.36 | 367.85 |
| Quality pgvector | 0.06449 | 0.90972 | 0.54203 | 0.72505 | 2171.77 | 5441.87 |

- Output hashes: `results.json`
  `cac370c8d092b7feeb6a762856bf90e10eef7c07af9ed9adbc00384ee2a2a0e7`,
  `runs.jsonl`
  `bd1151f342a6137a3c7cf2f086fe45429c52373c8d59bce3246cf40739c9db42`,
  and `split.json`
  `6c265ea024f724def1f25db938ab674172c8f811f4120eb00d3a2067916b116d`.
- Real-LLM Agent, without-validator, and always-agent-vs-routed comparisons are
  explicitly `not run — no real LLM was authorized`. Tokens/cost are not
  measured. Constraint validity and unsupported-claim rate are not applicable
  to retrieval-only WANDS runs.

## Frozen ESCI result (first and only test run)

- Downloaded only the two approved parquet files from Amazon Science's official
  `esci-data` repository at commit
  `7916cdf6ab75a462e77f20ab40428a10923998d5`; verified the declared LFS sizes,
  SHA-256 digests, Parquet magic, schemas, and row counts. Raw files remain
  ignored and are not delivery artifacts.
- Prepared a deterministic English-US `small_version==1` fixed subset using
  seed `20260814`: 300 training, 100 validation, and 100 official-test queries,
  with pairwise-disjoint query groups and no missing product joins, duplicate
  query/product judgments, or empty titles. The derived catalog contains
  10,346 unique products.
- Created isolated `marketlens_esci_test` with `template0`, `C/C` locale, and
  migrations only through frozen 0004. Indexed 10,346 real 384-dimensional
  embeddings atomically; no product or vector failures occurred.
- Froze configuration hash
  `4187b20b99d3c1fef364184d7003e47070f92312958aa4bbdf0a36da54fd4e38`
  after full validation and before opening test qrels. No post-test tuning or
  rerun occurred.
- The first test run covered 100 unique queries, seven strategies, and 700
  runs with zero execution failures. Semantic and Hybrid memory/pgvector
  rankings matched exactly on 100/100 queries.

| Strategy | Recall@10 (E/S) | Relevant MRR@10 | Exact MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.00250 | 0.01000 | 0.01000 | 0.00390 | 0.00 | 0.00 |
| BM25 | 0.40173 | 0.83133 | 0.67435 | 0.62460 | 36.58 | 80.24 |
| Semantic memory | 0.43677 | 0.91093 | 0.73254 | 0.65920 | 81.95 | 91.85 |
| Semantic pgvector | 0.43677 | 0.91093 | 0.73254 | 0.65920 | 186.83 | 233.72 |
| Hybrid memory | 0.43436 | 0.88278 | 0.71836 | 0.66477 | 108.74 | 154.38 |
| Hybrid pgvector | 0.43436 | 0.88278 | 0.71836 | 0.66477 | 127.93 | 155.47 |
| Quality pgvector | 0.46589 | 0.90350 | 0.78875 | 0.71306 | 3856.74 | 4667.29 |

- Result hashes: `results.json`
  `78214e35bb12e249d45d053288cf14a034d2128c87930fa39fb89ba1e1f0af91`
  and `runs.jsonl`
  `47601aa551d8daaf7eea47926ebf718cdf8335dcf0828ce46a816b839943be77`.
- This is explicitly a fixed 100-query subset, not a claim over the complete
  ESCI benchmark. A post-run audit corrected the frozen manifest's reason text:
  migration 0002 does contain HNSW, but `EXPLAIN` on both frozen databases
  showed the active deterministic two-key query used `Seq Scan + Sort`, so the
  recorded results are exact. Approximate HNSW was not separately run.

## Final Docker HTTP load matrix (first run)

- Final image
  `sha256:0e27929407bf23f45ad744c2fae48edc526510295e8381309b6ce3c19e762e00`
  is 612,818,126 bytes, runs as `appuser`, contains Torch 2.13.0+cpu, and loads
  both pinned models offline.
- Formal load used the 2,000-product `marketlens_test` database. Both readiness
  profiles reported a populated real 384-dimensional index; no memory fallback
  and no real/external LLM call occurred.
- Eight scenarios x 100 measured requests = 800/800 HTTP 200 responses, with
  zero 4xx, 5xx, transport errors, invalid payloads, empty results, degraded
  agents, or unexpected modes. Five warmups per scenario were excluded.

| Backend / path | Concurrency | RPS | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| memory Hybrid | 1 | 8.55 | 118.08 | 129.51 | 135.05 |
| memory Hybrid | 10 | 8.31 | 1193.22 | 1249.29 | 1493.62 |
| pgvector Hybrid | 1 | 31.77 | 30.85 | 34.96 | 40.44 |
| pgvector Hybrid | 10 | 32.88 | 291.47 | 329.25 | 331.67 |
| pgvector quality | 1 | 0.87 | 1070.58 | 1433.98 | 1526.96 |
| pgvector quality | 10 | 0.88 | 11276.00 | 12487.00 | 12495.51 |
| pgvector Fake-Agent | 1 | 17.35 | 53.72 | 71.23 | 93.93 |
| pgvector Fake-Agent | 10 | 19.78 | 473.32 | 666.91 | 670.36 |

- The measured quality path is CPU-bound under concurrency, which is retained
  as the first result rather than tuned away. This is a local development
  benchmark, not a production capacity or SLA claim.
- Final report SHA-256:
  `33818b33bdbd4ac593670C137E48DFF79BD15C9D8478A649B53BECC6D6EDEAD0`.

## Commands actually run

- `Get-Content .python-version` (file was absent)
- `uv --version`
- `uv python list`
- `py -0p`
- `Get-Command python`
- `Get-Command uv`
- `git check-ignore`
- Read-only inspection of `pyproject.toml`, `uv.lock`, `.gitignore`, and `.venv/pyvenv.cfg`
- `uv python install 3.12.13 --install-dir .python-phase8 --no-registry --no-bin`
- `uv sync --frozen`
- `uv sync --frozen --extra dev --extra db --extra embeddings`
- `uv run python --version`
- `uv run ruff check .`
- `uv run mypy src scripts`
- `uv run pytest tests/test_semantic_backends.py tests/test_retrieval_service.py -q -rs`
- Repeated `uv run ruff check .`, `uv run mypy src scripts`, and `git diff --check`

## Test results

- Python: 3.12.13.
- Final full suite: 437 passed, 1 expected skip, 0 failed. PostgreSQL fixtures
  used only `marketlens_repo_test`; ordinary API tests used a workspace-local
  SQLite file. The initial gate exposed host Temp permissions and a missing
  locked `data` extra; the successful gate used a new workspace basetemp and
  the complete frozen extras without changing tests.
- Ruff: passed for the full first-party tree.
- mypy: passed for 59 source/script files.
- `git diff --check`: passed (Windows line-ending warnings only).
- Repository-level pgvector tests: 13 passed, including the 2,000-row real
  repeat, tolerant float32 comparison, material updates, and non-finite input.
- Phase 8 real PostgreSQL tests: 4 passed.
- WANDS workflow tests: 2 passed; Ruff and mypy pass for both new scripts.
- WANDS validation smoke: 7 strategies, 2 queries each, 0 failures.

## Blockers

- No implementation or evaluation blocker remains.
- Remote publication remains conditional on GitHub browser authentication and
  a clean non-conflicting remote target.

## Pre-publication object-store audit

- Created and verified repository-external bundle
  `D:\VibeCoding\MarketLens-Research-Agent-pre-public-20260814-163714.bundle`
  (7,143,843 bytes, SHA-256
  `ABF079043E00AF5ED5363308DE1F582DAFCAA78B55C0A5D792F844286E8FDD21`).
- Backed up both local data-quality reports outside the repository without
  deleting or modifying their originals.
- Rechecked blob `9e4f7962d0a9cf5e454ce35719bf1d5a30bb7a1d`, 66,038,912 bytes, previously
  associated with `data/cache/embeddings_fea3dff39a7bb8cc.npy` by an ephemeral
  object scan. It is an unreachable object, absent from all commits, refs, path
  logs, and the feature branch. The bundle's refs were independently audited.
- Because no reachable history contained the cache, `git-filter-repo` was not
  run and no commit SHA changed. The old-to-new mapping is identity for all
  Phase 0-8 commits.
- Current reachable history contains no raw dataset, embedding array, model
  weight, or credential. No reachable blob exceeds 50 MiB; the largest is
  2,105,328 bytes.

## Next step

- Install/authenticate GitHub CLI, verify the remote target, then normally push
  the audited branch as public `main` and wait for real GitHub Actions.
