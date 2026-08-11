# MarketLens Architecture

## Overview

MarketLens Research Agent is a vertical product research system built on Open Deep Research, designed for AI Application, AI Backend, RAG/Search, and Agent Engineer job applications. It uses LangGraph for agent orchestration, hybrid retrieval (BM25 + embeddings), and FastAPI for serving.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                            │
│  GET /health  GET /search  POST /research  /research/jobs/*    │
│  (routes.py, models.py, main.py)                               │
├─────────────────────────────────────────────────────────────────┤
│                     Research Agent (LangGraph)                   │
│  parse_request → retrieve → assess → compare → validate → report│
│  (graph.py, state.py, prompts.py, tools.py, fake_llm.py)       │
├─────────────────────────────────────────────────────────────────┤
│                    Retrieval Layer (RRF Hybrid)                  │
│  BM25 (bm25.py)  +  Embedding (embedding.py)  +  Reranker      │
│  → Reciprocal Rank Fusion (hybrid.py)                          │
├─────────────────────────────────────────────────────────────────┤
│                      Domain Models (Pydantic v2)                 │
│  Product, SearchResult, ProductEvidence, ResearchReport, etc.   │
│  (models.py)                                                    │
├─────────────────────────────────────────────────────────────────┤
│                   Product Catalog & Persistence                  │
│  ProductCatalog (catalog.py)  +  SQLAlchemy (database.py)      │
│  SQLite (default)  |  PostgreSQL/pgvector (production)          │
└─────────────────────────────────────────────────────────────────┘
```

## Package Structure

```
src/marketlens/
├── __init__.py                  # Package root, version
├── models.py                    # Pydantic v2 domain models
├── catalog.py                   # ProductCatalog: load, index, filter
├── fixtures/
│   └── electronics_sample.json  # 20-product electronics fixture
├── agent/
│   ├── __init__.py
│   ├── state.py                 # LangGraph AgentState (TypedDict)
│   ├── graph.py                 # LangGraph workflow (8 nodes)
│   ├── tools.py                 # LangChain tools for agent
│   ├── fake_llm.py              # FakeLLM for offline testing
│   └── prompts.py               # Prompt templates
├── retrieval/
│   ├── __init__.py
│   ├── bm25.py                  # Okapi BM25 keyword search
│   ├── embedding.py             # Embedding search (fake + sentence-transformers)
│   ├── hybrid.py                # RRF hybrid retrieval
│   └── reranker.py              # Reranker interface (keyword + no-op)
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app + lifespan
│   ├── routes.py                # Endpoint handlers
│   ├── models.py                # API request/response models
│   └── database.py              # SQLAlchemy models + session
└── evaluation/
    ├── __init__.py
    └── benchmark.py             # Evaluation metrics + report
```

## Request Flow

A typical research request flows through:

1. **User** sends `POST /research {"query": "best ANC headphones under $300"}`
2. **FastAPI** (`routes.py`) validates input, creates SQLAlchemy job record
3. **Agent** (`graph.py`) invokes LangGraph workflow:
   - `parse_request_node`: FakeLLM extracts constraints ($300 budget, ANC feature)
   - `retrieve_catalog_node`: HybridRetriever searches catalog with filters
   - `assess_evidence_node`: Evaluates evidence quality per product
   - `compare_products_node`: Generates pros/cons/comparison
   - `validate_constraints_node`: Plain Python checks budget, brand, rating
   - `generate_report_node`: Creates structured markdown report
4. **Response**: JSON with job_id, status, and report

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pydantic v2 domain models | Type-safe, validated, serializable data structures |
| BM25 + embedding + RRF hybrid | Complementary: keyword precision + semantic recall |
| FakeLLM for offline mode | Zero API key requirement for testing/demo |
| Deterministic constraint checks | Python, not LLM, enforces budget/brand/rating |
| SQLite default, pgvector optional | Zero-config local dev, scalable production |
| Single-agent LangGraph | Simple, debuggable; multi-agent adds complexity without benefit here |
| Evidence traceability | Every recommendation links to ProductEvidence |

## Technology Stack

- **Agent Framework**: LangGraph (state machine with conditional routing)
- **LLM Interface**: LangChain (FakeLLM / init_chat_model for real)
- **Retrieval**: Custom BM25 + numpy embedding + RRF
- **API**: FastAPI + Pydantic v2
- **Persistence**: SQLAlchemy (SQLite / PostgreSQL)
- **Testing**: pytest (166 tests, no API keys required)
- **Container**: Docker Compose (FastAPI + pgvector)
- **CI**: GitHub Actions (ruff + pytest + docker check)

## Data Flow Diagram

```
User Query → SearchQuery → UserConstraints (budget, brand, etc.)
    → HybridRetriever.search()
        → BM25 search (keyword ranking)
        → Embedding search (cosine similarity)
        → RRF fusion (weighted rank combination)
        → Optional reranker
        → Hard filter (catalog.filter_by_constraints)
    → [SearchResult, ...]
    → Agent (FakeLLM.parse_request, assess_evidence, compare)
    → ResearchReport (markdown + evidence + comparisons)
```
