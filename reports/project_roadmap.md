# MarketLens Project Roadmap

## Project Goal

Build an AI product research agent that accepts natural-language product needs, finds relevant products via keyword + semantic + hybrid retrieval, applies hard constraints, and generates evidence-backed comparison reports.

## Six Phases

| # | Phase | Status | Key Skills |
|---|-------|--------|------------|
| 1 | Engineering Foundation & BM25 | ✅ Done | Git, pytest, ruff, mypy, Pydantic v2, BM25 from scratch |
| 2 | Real Data Pipeline | ✅ Done | Streaming data, field cleaning, dedup, provenance |
| 3 | **Retrieval Core v1** | ✅ Done | Embedding, RRF, reranker, unified service, API |
| 4 | Retrieval Evaluation | ⏳ Next | Human-reviewed eval set, Recall@10, nDCG@10, P50/P95 |
| 5 | Research Agent | 📋 Planned | LangGraph agent, FakeLLM→real LLM, evidence comparison |
| 6 | API, Testing, Docker, Demo | 📋 Planned | FastAPI finalization, Docker Compose, demo video |

## Phase 3 Detail (Current)

### What was built
- Unified `RetrievalService` orchestrating BM25, Embedding, Hybrid RRF, and Rerank
- Real sentence-transformers integration (all-MiniLM-L6-v2, 384-dim)
- Embedding cache system (numpy disk cache tied to data+model hash)
- Structured filtering (brand, price range, rating) — missing price correctly excluded
- Updated FastAPI `/search` with `strategy`, `min_price`, `candidate_k` params
- 32 new tests including filtering edge cases, API validation

### Key design decisions
- RRF (not score addition) for hybrid fusion — scale-free, explainable
- In-memory numpy for 2k products — no vector DB needed at this scale
- Two-stage retrieval: broad recall → candidate reranking
- Missing price = excluded from price filters (can't satisfy or violate)

### Interview knowledge
- BM25, TF-IDF, embedding similarity, RRF, two-stage retrieval
- FastAPI, Pydantic, SQLAlchemy, pytest, mypy
- How to explain retrieval tradeoffs to non-technical audiences

## Phase 4 (Next)

### What's needed
- Human-reviewed evaluation queries from Phase 3's 50 auto-curated candidates
- Compute Recall@10, nDCG@10, P50/P95 latency per strategy
- Compare BM25 vs Embedding vs Hybrid vs Rerank on real metrics
- Write evaluation report with methodology transparency

### Blocked by
- Human review of eval candidates (currently all `pending`)

## Phase 5 & 6

- Agent orchestration with LangGraph (already prototyped in Phase 2)
- Real LLM integration (Claude/GPT) replacing FakeLLM
- Final API hardening, Docker, demo recording
