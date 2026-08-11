# Upstream vs. My Work

## Upstream: Open Deep Research

**Repository**: [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)
**Version**: 0.0.16
**Commit**: `1b7d2e80db9faa586165c60e09096dbbfd483a64`
**License**: MIT

### What Upstream Provides

1. **Multi-LLM Deep Research Agent**: Research supervisor + sub-researcher architecture for web-based research
2. **Search API Integrations**: Tavily, OpenAI native, Anthropic native, DuckDuckGo, Exa
3. **MCP Support**: Model Context Protocol for external tool integration
4. **LangGraph Workflow**: Plan-execute-report workflow with parallel sub-agents
5. **LangSmith Integration**: Tracing, evaluation, experiment management
6. **LangGraph Studio UI**: Visual configuration and testing
7. **Report Generation**: Structured markdown with citations and sources
8. **Legacy Implementations**: Plan-and-execute and multi-agent architectures

### What Upstream Does NOT Provide

- Product catalog or structured product data
- Hybrid retrieval (BM25 + embedding)
- Evidence-grounded product comparisons
- Pydantic domain models for product research
- FastAPI REST API
- SQL persistence
- Offline fake LLM for testing
- Reproducible fixture-based benchmarks
- Hard constraint validation (budget, brand, rating)

---

## MarketLens: My Work

### What I Preserved

- MIT License (unchanged)
- `src/legacy/` directory (for reference)
- `src/security/auth.py` (LangGraph auth)
- `src/open_deep_research/` core (unchanged, for reference)
- Original `examples/` directory
- `pyproject.toml` structure (extended with new packages)

### What I Modified

- `pyproject.toml`: Renamed project to `marketlens-research-agent`, added new packages
- `CLAUDE.md`: Added MarketLens autonomous execution instructions
- `langgraph.json`: Will point to MarketLens graph

### What I Built (New)

#### Phase 1: Product Catalog & Hybrid Retrieval (~2,650 lines)

| Component | Files | Description |
|-----------|-------|-------------|
| Domain Models | `src/marketlens/models.py` | 10 Pydantic v2 models (Product, SearchResult, ProductEvidence, etc.) |
| Catalog | `src/marketlens/catalog.py` | ProductCatalog with JSON loading, indexing, filtering |
| BM25 | `src/marketlens/retrieval/bm25.py` | Okapi BM25 keyword retriever |
| Embedding | `src/marketlens/retrieval/embedding.py` | FakeEmbeddingBackend + SentenceTransformersBackend |
| Hybrid | `src/marketlens/retrieval/hybrid.py` | RRF fusion + hard filter + reranker |
| Reranker | `src/marketlens/retrieval/reranker.py` | KeywordReranker (Jaccard) + NoOpReranker |
| Fixture | `src/marketlens/fixtures/electronics_sample.json` | 20 electronics products |
| Tests | `tests/test_*.py` | 105 tests |

#### Phase 2: Research Agent (~1,500 lines)

| Component | Files | Description |
|-----------|-------|-------------|
| State | `src/marketlens/agent/state.py` | AgentState TypedDict with 20+ fields |
| Graph | `src/marketlens/agent/graph.py` | 8-node LangGraph workflow |
| Tools | `src/marketlens/agent/tools.py` | Catalog search, web search, research complete tools |
| FakeLLM | `src/marketlens/agent/fake_llm.py` | Rule-based fake LLM (no API keys) |
| Prompts | `src/marketlens/agent/prompts.py` | Prompt templates for each node |
| Tests | `tests/test_agent.py` | 22 tests |

#### Phase 3: API & Persistence (~1,000 lines)

| Component | Files | Description |
|-----------|-------|-------------|
| FastAPI App | `src/marketlens/api/main.py` | App with CORS, request_id middleware, error handler |
| Routes | `src/marketlens/api/routes.py` | 6 endpoints (health, search, research, jobs) |
| API Models | `src/marketlens/api/models.py` | Request/response Pydantic models |
| Database | `src/marketlens/api/database.py` | SQLAlchemy with ResearchJobRecord, SearchQueryRecord |
| Docker | `Dockerfile`, `docker-compose.yml` | FastAPI + pgvector/pg16 |
| Tests | `tests/test_api.py` | 23 tests |

#### Phase 4: Evaluation (~700 lines)

| Component | Files | Description |
|-----------|-------|-------------|
| Benchmark | `src/marketlens/evaluation/benchmark.py` | Recall@10, nDCG@10, constraint rate, etc. |
| CI | `.github/workflows/ci.yml` | pytest + ruff + docker check |
| Tests | `tests/test_evaluation.py` | 16 tests with 12 fixture queries |

#### Phase 5: Documentation (~5,000 words)

| Document | Description |
|----------|-------------|
| README.md | Updated project overview |
| ARCHITECTURE.md | System architecture and design decisions |
| UPSTREAM_VS_MY_WORK.md | This document |
| EVALUATION_REPORT.md | Benchmark methodology and results |
| IMPLEMENTATION_REPORT.md | Implementation summary by phase |
| LEARNING_GUIDE.md | Study guide for job interviews |
| DEMO_SCRIPT.md | Step-by-step demo walkthrough |

### Quantitative Summary

| Metric | Count |
|--------|-------|
| Total new Python files | 26 |
| Total new lines of code | ~7,500+ |
| Pytest tests | 166 |
| Test coverage areas | Models, catalog, BM25, embedding, hybrid, filtering, reranking, agent workflow, API endpoints, evaluation |
| Pydantic models | 18 |
| LangGraph nodes | 8 |
| FastAPI endpoints | 6 |
| SQLAlchemy models | 2 |
| Docker services | 2 |
| Evaluation queries | 12 |
| Evaluation metrics | 6 |
| Documentation pages | 7 |
