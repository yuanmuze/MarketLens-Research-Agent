# 🔬 MarketLens Research Agent

> Built on [Open Deep Research](https://github.com/langchain-ai/open_deep_research) (MIT License)

MarketLens is a vertical product research system demonstrating AI Application, AI Backend, RAG/Search, and Agent Engineer skills. It combines **hybrid retrieval** (BM25 + embeddings), **LangGraph agents**, and a **FastAPI** backend for evidence-grounded product recommendations.

## What This Project Demonstrates

- **RAG/Search**: BM25 keyword search, embedding semantic search, Reciprocal Rank Fusion hybrid retrieval, optional reranker
- **Agent Engineering**: LangGraph single-agent workflow with 8 nodes, LangChain tools, FakeLLM for offline testing
- **AI Backend**: FastAPI with 6 endpoints, SQLAlchemy persistence (SQLite/PostgreSQL), Pydantic v2 validation
- **ML Systems**: Reproducible evaluation benchmarks (Recall@10, nDCG@10, constraint satisfaction), GitHub Actions CI
- **Vector Backend**: Explicit memory/pgvector retrieval with readiness checks,
  atomic idempotent indexing, and measured backend parity

## Quickstart (No API Keys Required)

```bash
# Clone and install (all extras: dev tools, data pipeline, embeddings)
git clone <this-repo>
cd MarketLens-Research-Agent
uv sync --python 3.11 --extra dev --extra data --extra embeddings

# Run all first-party tests (351 passed, 1 skip — no API keys needed)
uv run pytest

# Start the API
uv run uvicorn marketlens.api.main:app --reload

# Open http://127.0.0.1:8000/docs for interactive API docs
```

> **Legacy/upstream code** (`src/legacy/`, `src/open_deep_research/`,
> `src/security/`) is excluded from pytest collection, ruff, and mypy.
> These are vendored from the original Open Deep Research upstream and
> are kept for reference only, not maintained as first-party code.

## Architecture

```
User Query → FastAPI → LangGraph Agent (8 nodes) → Hybrid Retrieval
                         parse_request              BM25 + Embedding
                         retrieve_catalog            → RRF Fusion
                         assess_evidence             → Hard Filter
                         compare_products            → Reranker
                         validate_constraints
                         generate_report → ResearchReport (Markdown + Evidence)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with catalog size |
| `/search` | GET | Product search with filtering (budget, brand, rating) |
| `/research` | POST | Submit and execute research (synchronous) |
| `/research/jobs` | POST | Create async research job |
| `/research/jobs/{id}` | GET | Get job status |
| `/research/jobs/{id}/report` | GET | Get completed research report |
| `/agent/recommend` | POST | Natural language → evidence-backed recommendations |
| `/feedback` | POST | Record user feedback on an agent run |
| `/health/live` | GET | Liveness probe (process alive) |
| `/health/ready` | GET | Readiness probe (catalog + retrieval usable) |

## Project Structure

```
src/marketlens/
├── models.py         # Pydantic v2 domain models (10 models)
├── catalog.py        # ProductCatalog with filtering
├── fixtures/         # 20-product electronics sample data
├── retrieval/        # BM25, embedding (fake + sentence-transformers), hybrid, reranker
├── agent/            # LangGraph workflow, FakeLLM, tools
├── api/              # FastAPI app, routes, SQLAlchemy
└── evaluation/       # Benchmark metrics, 60-query eval set, 4-strategy comparison
scripts/
└── prepare_electronics_data.py  # Amazon Reviews 2023 data pipeline
```

## Running Tests

```bash
# Full test suite (351 passed, 1 skip — all offline, no API keys)
uv run pytest

# Specific test groups
uv run pytest tests/test_models.py -v        # 16 model tests
uv run pytest tests/test_retrieval_service.py -v  # retrieval service tests
uv run pytest tests/test_agent.py -v         # legacy agent tests
uv run pytest tests/test_agent_phase5.py -v  # 60 Phase 5 agent tests
uv run pytest tests/test_api.py -v           # 23 API tests
uv run pytest tests/test_wands_evaluation.py -v  # 30 WANDS metric tests
```

## Quality Gates

```bash
uv run ruff check .       # lints first-party code (legacy/upstream excluded)
uv run mypy src scripts   # type-checks first-party code (legacy/upstream excluded)
```

Both commands pass clean. Legacy/upstream directories (`src/legacy/`,
`src/open_deep_research/`, `src/security/`) are explicitly excluded in
`pyproject.toml` because they are vendored from the original Open Deep
Research project and are not maintained as first-party MarketLens code.

## Fake Mode vs. Real LLM Mode

**Fake mode** (default): Uses FakeLLM — rule-based parsing, templated reports. Works without API keys. Perfect for testing architecture, demos, and development.

**Real LLM mode**: Set environment variables and use LangChain's `init_chat_model`. The agent graph and tools remain the same; only the LLM backend changes. See `src/marketlens/agent/graph.py` for the `use_fake_llm` parameter.

## Upstream

This project is built on [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) (v0.0.16, commit `1b7d2e8`, MIT License).

**Upstream provides**: Multi-LLM deep research agent, Tavily/OpenAI/Anthropic search, MCP support, LangGraph Studio UI, LangSmith evaluation.

**MarketLens adds**: Product catalog + hybrid retrieval, evidence-grounded research agent, FastAPI + persistence, offline fake LLM, evaluation benchmarks, 166 pytest tests.

See [docs/UPSTREAM_VS_MY_WORK.md](docs/UPSTREAM_VS_MY_WORK.md) for full comparison.

## Evaluation (WANDS Benchmark)

Uses the [WANDS](https://github.com/wayfair/WANDS) public e-commerce search
benchmark (MIT License) with 42,994 products, 480 queries, and 233,448
human relevance labels (Exact/Partial/Irrelevant).

```bash
# Download WANDS
uv run python scripts/download_wands.py

# Verify integrity
uv run python scripts/verify_wands_data.py

# Run full benchmark (4 strategies × 480 queries)
uv run python scripts/evaluate_wands.py

# Resume interrupted run
uv run python scripts/evaluate_wands.py --resume
```

Results are written to `data/evaluation/wands/`.
See [reports/wands_evaluation_report.md](reports/wands_evaluation_report.md).

WANDS and Amazon are separate data links:
- **WANDS**: Official retrieval benchmark (furniture, human-labeled)
- **Amazon 2,000**: Structured filtering demo (electronics, auto-curated)

## Phase 8: Real pgvector and Frozen Evaluation

Phase 8 adds a real PostgreSQL cosine semantic backend without changing
frozen migrations 0001-0004. On fixed test splits, memory and pgvector rankings
matched exactly for all 96 WANDS queries and all 100 ESCI queries. The quality
reranker reached nDCG@10 0.72505 on WANDS and 0.71306 on the fixed 100-query
ESCI US reduced subset (not the full official ESCI benchmark).

The final local Docker matrix completed 800/800 validated requests across real
memory/pgvector Hybrid, pgvector quality, and deterministic Fake-Agent paths,
with no HTTP, payload, fallback, or external-LLM failures. At concurrency 10,
pgvector Hybrid measured 32.88 RPS / p95 329.25 ms; CPU reranking was the clear
bottleneck at 0.88 RPS / p95 12.49 s. These are local development results, not
a production SLA or capacity claim.

See [PHASE8_DELIVERY.md](docs/PHASE8_DELIVERY.md) and
[phase8_progress.md](reports/phase8_progress.md) for methodology, limitations,
metrics, reproducibility hashes, and interview-ready framing.

## Data Pipeline

Uses the official UCSD Amazon Reviews 2023 JSONL source via HuggingFace
`datasets` (json builder, `streaming=True`). No `trust_remote_code`, no
dataset scripts. Compatible with `datasets >= 5.0`.

```bash
# Install data dependency
uv sync --extra data

# Stream Electronics metadata (~2000 products, seed 42)
uv run python scripts/prepare_electronics_data.py --max-products 2000 --seed 42

# Dry run (validate pipeline without writing)
uv run python scripts/prepare_electronics_data.py --dry-run

# From local file
uv run python scripts/prepare_electronics_data.py --local-file /path/to/meta_Electronics.jsonl.gz
```

## Retrieval Benchmark

```bash
# Run 4-strategy comparison (fixture data, offline)
uv run pytest tests/test_retrieval_comparison.py -v -s

# Run with real embeddings (requires sentence-transformers)
uv pip install sentence-transformers
uv run python -c "
from marketlens.catalog import ProductCatalog
from marketlens.evaluation.retrieval_comparison import *
catalog = ProductCatalog.from_fixture('electronics_sample.json')
queries = build_eval_queries(catalog)
reports = run_full_comparison(catalog, queries, use_real_embeddings=True)
print(generate_markdown_report(reports, queries))
"
```

## Documentation

| Document | Description |
|----------|-------------|
| [UPSTREAM_BASELINE.md](docs/UPSTREAM_BASELINE.md) | Original upstream capabilities, commit SHA, environment |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, data flow, design decisions |
| [UPSTREAM_VS_MY_WORK.md](docs/UPSTREAM_VS_MY_WORK.md) | Quantitative comparison: upstream vs. my work |
| [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) | Benchmark methodology, metrics |
| [IMPLEMENTATION_REPORT.md](docs/IMPLEMENTATION_REPORT.md) | Per-phase implementation details |
| [LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) | Core files, request flow, 20 interview Q&A, 7-day plan |
| [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | Step-by-step demo walkthrough |
| [PHASE6_IMPLEMENTATION_REPORT.md](docs/PHASE6_IMPLEMENTATION_REPORT.md) | Data pipeline, real embeddings, comparison framework |
| [PHASE6_EVALUATION_REPORT.md](docs/PHASE6_EVALUATION_REPORT.md) | Phase 6 benchmark results (fixture) |
| [PHASE6_LEARNING_GUIDE.md](docs/PHASE6_LEARNING_GUIDE.md) | BM25/embedding/RRF deep dive, 15 new interview Q&A |
| [PHASE8_DELIVERY.md](docs/PHASE8_DELIVERY.md) | Real pgvector architecture, frozen evaluation, load evidence, and portfolio summary |
| [EVALUATION_ANNOTATION_GUIDE.md](docs/EVALUATION_ANNOTATION_GUIDE.md) | How to manually review eval queries |

## PostgreSQL Persistence (Phase 6)

### Start a development PostgreSQL

> The persistence layer uses a **synchronous** SQLAlchemy engine with the
> **psycopg2** driver (`postgresql+psycopg2://`). Alembic uses the same
> engine/driver.
> PostgreSQL is bound to **127.0.0.1:5432** (localhost only), not exposed
> on the LAN/Internet.

```bash
# Start dev PostgreSQL 16 (development only, not full deployment)
docker compose up -d db

# Configure DATABASE_URL (matching compose.yaml defaults)
export MARKETLENS_DATABASE_URL="postgresql+psycopg2://marketlens:marketlens@localhost:5432/marketlens"

# Run migrations (create tables)
uv run alembic upgrade head

# Import products from JSON (idempotent — safe to re-run)
uv run python scripts/import_products.py --input data/processed/electronics_2000.json

# Use postgres catalog backend
export MARKETLENS_CATALOG_BACKEND=postgres

# Shut down when done
docker compose down
```

> Never run `alembic downgrade` on a normal development database — it is
> destructive. Only test downgrade on a dedicated throwaway test database.

### Start the full stack (db + API) — one command

```bash
docker compose up -d
# API: http://127.0.0.1:8000  |  liveness: /health/live  |  readiness: /health/ready
```

The Compose credentials are for local development. Before pgvector mode can
become ready, import a catalog and build its matching real embedding index:

```bash
uv run python scripts/import_products.py --input data/processed/electronics_2000.json
uv run python scripts/index_product_embeddings.py \
  --products data/processed/electronics_2000.json \
  --model all-MiniLM-L6-v2

curl "http://127.0.0.1:8000/search?q=wireless%20headphones&strategy=hybrid&top_k=5"
curl -X POST "http://127.0.0.1:8000/agent/recommend" \
  -H "Content-Type: application/json" \
  -d '{"message":"best wireless headphones","mode":"balanced","max_results":5}'
```

Phase 8 index/evaluation commands refuse PostgreSQL database names that do not
contain `test`. Never point destructive test fixtures at the development DB.

### Phase 8 load test (local, hits the running Docker API)

```bash
uv run python scripts/load_test_phase8.py run --profile pgvector \
  --requests 100 --image-id <docker-image-id> --output tmp/phase8-load-pgvector.json
```

### Two catalog backends

| Backend | Behavior |
|---------|----------|
| `json` (default) | Loads products from JSON file (existing behavior) |
| `postgres` | Loads products from `products` table, then builds in-memory retrieval index |

Set `MARKETLENS_SEMANTIC_BACKEND=memory` or `pgvector`. The pgvector mode uses
PostgreSQL cosine search over a pre-built 384-dimensional index and fails
readiness instead of silently falling back to memory. The memory mode remains
available for local parity and offline tests.

### Tests

```bash
# Default unit + regression tests (no PostgreSQL required)
uv run pytest

# PostgreSQL integration tests (skip if no test DB configured)
export MARKETLENS_TEST_DATABASE_URL="postgresql+psycopg2://marketlens:marketlens@localhost:5432/marketlens_test"
uv run pytest -m postgres
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TAVILY_API_KEY` | No | empty | Enables web search tool |
| `OPENAI_API_KEY` | No | empty | For legacy real LLM mode |
| `MARKETLENS_AGENT_API_KEY` | No | empty | OpenAI-compatible Agent provider; unused with Fake LLM |
| `MARKETLENS_USE_FAKE_LLM` | No | `false` | Deterministic local Agent; no external LLM calls |
| `MARKETLENS_DATABASE_URL` | No | `sqlite:///marketlens_persistence.db` | DB connection (Phase 6 persistence) |
| `MARKETLENS_CATALOG_BACKEND` | No | `json` | `json` or `postgres` |
| `MARKETLENS_CATALOG_PATH` | No | auto | JSON source/cache identity path |
| `MARKETLENS_SEMANTIC_BACKEND` | No | `memory` | `memory` or PostgreSQL `pgvector` |
| `MARKETLENS_EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Real 384-dimensional model |
| `MARKETLENS_TEST_DATABASE_URL` | No | empty | Dedicated PostgreSQL test DB; name must contain `test` |

## ESCI provenance and reproduction

Phase 8 uses only Amazon Science's official
[`esci-data`](https://github.com/amazon-science/esci-data) repository at commit
`7916cdf6ab75a462e77f20ab40428a10923998d5`, licensed Apache-2.0. Only the
official examples and products parquet files are downloaded; their LFS sizes,
SHA-256 digests, schemas, and row counts are recorded in
[`esci_source.json`](data/manifests/esci_source.json). Raw files, derived query
details, qrels, embeddings, and model weights are absent from the public
reachable history. A pre-publication object-store audit also verified that a
local unreachable cache blob cannot be included by a normal branch push.

```bash
uv run python scripts/download_esci.py
uv run python scripts/prepare_esci_subset.py

# Requires a dedicated database whose name contains "test".
export MARKETLENS_DATABASE_URL="postgresql+psycopg2://marketlens:marketlens@localhost:5432/marketlens_esci_test"
uv run python scripts/index_esci_embeddings.py
```

The subset manifest fixes seed `20260814`, query-group selection, source/data
hashes, 300 train / 100 validation / 100 official-test queries, and pairwise
split disjointness. Evaluation commands and frozen results are documented in
[`PHASE8_DELIVERY.md`](docs/PHASE8_DELIVERY.md); running them again would create
a new experiment and is not part of the frozen result.

## Final quality gates

```bash
uv run pytest -q -rs
uv run pytest -m postgres -q -rs
uv run ruff check .
uv run mypy src scripts
uv run alembic current
uv run alembic check
docker compose config
```

Phase 8's final local gate passed 437 tests with one expected skip, the real
PostgreSQL subset passed, Ruff and mypy passed, and Alembic remained at 0004
with no migration changes.

## Known Limitations

- **Fixed evaluation subsets**: Phase 8 numbers do not represent complete
  official WANDS/ESCI leaderboards; WANDS is not a historically untouched holdout.
- **Exact measured pgvector path**: migration 0002 defines HNSW, but the active
  deterministic two-key query planned as `Seq Scan + Sort` in both frozen
  evaluation databases. Approximate HNSW behavior was not separately measured.
- **CPU reranker bottleneck**: quality mode improved nDCG but reached only 0.88
  RPS at concurrency 10 in the local Docker matrix.
- **Fake LLM evidence only**: Phase 8 made zero external LLM calls; it does not
  report token cost, real-provider quality, or production Agent reliability.
- **Development deployment**: no authentication, distributed index,
  autoscaling, production observability, or production SLA validation.

## Dataset licenses and citation

- WANDS: Wayfair Annotated Dataset for Search, MIT license. See the
  [official WANDS repository](https://github.com/wayfair/WANDS).
- ESCI: Amazon Science Shopping Queries Dataset, Apache-2.0 license. See the
  [official ESCI repository](https://github.com/amazon-science/esci-data) and
  its citation instructions. Phase 8 uses a reproducible reduced subset.

## License

MIT — Based on Open Deep Research by LangChain (Lance Martin).

---

*Built for AI Application, AI Backend, RAG/Search, and Agent Engineer roles.*
