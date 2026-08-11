# MarketLens Implementation Report

## Project Overview

MarketLens Research Agent transforms Open Deep Research into a vertical product research system. It demonstrates skills relevant to AI Application, AI Backend, RAG/Search, and Agent Engineer roles.

## Implementation by Phase

### Phase 0: Upstream Baseline (1 file, ~80 lines)

- Cloned [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)
- Established Python 3.11 environment with uv
- Documented upstream capabilities, license (MIT), and commit SHA
- Original tests require API keys (no pytest unit tests existed)
- Created `docs/UPSTREAM_BASELINE.md`

### Phase 1: Product Catalog & Hybrid Retrieval (10 files, ~2,650 lines)

**What was built:**
- 10 Pydantic v2 domain models (`models.py`): Product, ProductEvidence, UserConstraints, SearchQuery, SearchResult, ComparisonItem, ResearchRequest, ResearchJob, ResearchReport, and enums
- `ProductCatalog` (`catalog.py`): In-memory catalog with JSON loading, brand/category indexing, and deterministic hard constraint filtering (budget, brand, rating, review count, category)
- `BM25Retriever` (`bm25.py`): Okapi BM25 implementation with configurable k1/b/epsilon
- `EmbeddingRetriever` + `FakeEmbeddingBackend` (`embedding.py`): Cosine similarity search with deterministic hash-based embeddings (no model download)
- `HybridRetriever` (`hybrid.py`): Reciprocal Rank Fusion combining BM25 + embedding with configurable weights
- `KeywordReranker` + `NoOpReranker` (`reranker.py`): Jaccard similarity reranker
- 20-product electronics fixture (headphones/earbuds with real brands and prices)
- 105 pytest tests covering models, catalog, filtering, BM25, embedding, hybrid, reranker

**Key engineering decisions:**
- Fake embeddings use MD5 hash + random projection for deterministic testing
- RRF k=60 (standard), equal BM25/embedding weights
- Hard filters applied as pre-filter before RRF fusion
- All retrieval is local—no API keys, no model downloads

### Phase 2: Product Research Agent (6 files, ~1,500 lines)

**What was built:**
- FakeLLM (`fake_llm.py`): Rule-based parsing, evidence assessment, comparison, validation, and report generation. Extracts budget ($X), brands, and features from natural language queries.
- LangGraph workflow (`graph.py`): 8-node state machine:
  1. `parse_request` → extract search intent and constraints
  2. `retrieve_catalog` → hybrid search with pre-filters
  3. `assess_evidence` → quality scoring per product
  4. `optional_web_research` → Tavily integration (graceful fallback)
  5. `compare_products` → pros/cons and recommendation scores
  6. `validate_constraints` → deterministic Python checks
  7. `generate_report` → structured markdown report
  8. `handle_failure` → graceful error recovery
- LangChain tools (`tools.py`): `search_catalog`, `web_search`, `research_complete`
- Agent state tracking: node timings, tool call count, retries, request metadata

**Key engineering decisions:**
- Single-agent workflow (no unnecessary multi-agent complexity)
- Constraint validation is pure Python, not LLM
- Every recommendation traces to ProductEvidence
- Web search is optional; fake LLM works without any keys
- Max tool calls = 10, retryable failures, timeout awareness

### Phase 3: FastAPI & Persistence (6 files, ~1,000 lines)

**What was built:**
- FastAPI application (`main.py`): CORS, request_id middleware, global error handler (no stack leaks)
- 6 endpoints (`routes.py`): `GET /health`, `GET /search`, `POST /research`, `POST /research/jobs`, `GET /research/jobs/{job_id}`, `GET /research/jobs/{job_id}/report`
- SQLAlchemy models (`database.py`): `ResearchJobRecord`, `SearchQueryRecord`
- SQLite default, PostgreSQL/pgvector configurable via `MARKETLENS_DATABASE_URL`
- Docker Compose (`docker-compose.yml`): FastAPI + pgvector/pg16 with health checks

**Key engineering decisions:**
- Request IDs on every response for observability
- 4xx for validation errors, 5xx for server errors (never leak stack traces)
- Search queries automatically persisted for analytics
- Research jobs can be sync (POST /research) or async (POST /research/jobs)

### Phase 4: Evaluation & Reliability (3 files, ~700 lines)

**What was built:**
- Evaluation framework (`benchmark.py`): Recall@K, nDCG@K, constraint rate, completion rate, per-category breakdown
- 12 fixture queries across 7 categories with ground truth labels
- BM25 vs embedding vs hybrid comparison pipeline
- GitHub Actions CI (`ci.yml`): ruff linting, pytest, docker-compose validation
- All results explicitly marked as fixture benchmarks

### Phase 5: Documentation (7 files, ~5,000 words)

- Updated README with project overview, installation, API docs, testing
- Architecture document with diagrams and design decisions
- Upstream vs. My Work comparison (quantitative)
- Evaluation report with methodology
- Learning guide with interview questions
- Demo script for live presentation

## Technology Choices

| Technology | Why |
|-----------|-----|
| **LangGraph** | State machine with conditional routing; easier to understand than alternatives |
| **Pydantic v2** | Type-safe data validation; industry standard for FastAPI |
| **BM25 from scratch** | Demonstrates IR fundamentals; no external dependency |
| **FakeLLM** | Enables offline testing; proves agent architecture without API costs |
| **SQLite + SQLAlchemy** | Zero-config local dev; swap to PostgreSQL for production |
| **FastAPI** | Modern, fast, auto-generated OpenAPI docs |
| **pytest** | 166 tests, all runnable without API keys |
| **Docker Compose** | Reproducible production-like environment |

## Testing Summary

```
166 passed, 0 failed
- 16 model tests
- 16 catalog tests (+ 17 filter tests)
- 16 BM25 tests
- 16 embedding tests
- 17 hybrid tests
- 22 agent tests (fake LLM + workflow)
- 23 API tests (health, search, research, jobs, errors)
- 16 evaluation tests (metrics + benchmarks)
```

## Known Limitations

1. **Fixture data only**: The catalog contains 20 hand-crafted products, not real Amazon data
2. **Fake LLM**: Rule-based, not a real language model; responses are templated
3. **No streaming**: API responses are synchronous; no SSE/WebSocket for large reports
4. **In-memory catalog**: ProductCatalog is loaded in memory; no sharding or distributed index
5. **No auth**: API is open (CORS enabled); add auth for production
6. **No monitoring**: Prometheus/Grafana not configured; structured logging only
7. **Windows-only tested**: CI runs on ubuntu-latest but not tested locally on macOS/Linux

## Future Work

- Replace FakeLLM with real LLM (Claude/GPT) via LangChain `init_chat_model`
- Migrate to pgvector for production-scale embedding search
- Add streaming response for large reports
- Implement semantic chunking for long product descriptions
- Add image-based product search
- Integrate with real Amazon Reviews 2023 Electronics dataset
- Add A/B testing framework for retrieval strategies
