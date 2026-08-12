# 🔬 MarketLens Research Agent

> Built on [Open Deep Research](https://github.com/langchain-ai/open_deep_research) (MIT License)

MarketLens is a vertical product research system demonstrating AI Application, AI Backend, RAG/Search, and Agent Engineer skills. It combines **hybrid retrieval** (BM25 + embeddings), **LangGraph agents**, and a **FastAPI** backend for evidence-grounded product recommendations.

## What This Project Demonstrates

- **RAG/Search**: BM25 keyword search, embedding semantic search, Reciprocal Rank Fusion hybrid retrieval, optional reranker
- **Agent Engineering**: LangGraph single-agent workflow with 8 nodes, LangChain tools, FakeLLM for offline testing
- **AI Backend**: FastAPI with 6 endpoints, SQLAlchemy persistence (SQLite/PostgreSQL), Pydantic v2 validation
- **ML Systems**: Reproducible evaluation benchmarks (Recall@10, nDCG@10, constraint satisfaction), GitHub Actions CI

## Quickstart (No API Keys Required)

```bash
# Clone and install
git clone <this-repo>
cd MarketLens-Research-Agent
uv sync --python 3.11 --extra dev --extra data

# Run all tests (193 tests, 0 API keys needed)
uv run pytest tests/ -q --ignore=tests/extract_langsmith_data.py

# Start the API
uv run uvicorn marketlens.api.main:app --reload

# Open http://127.0.0.1:8000/docs for interactive API docs
```

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
# Full test suite (all offline, no API keys)
uv run pytest tests/ -q --ignore=tests/extract_langsmith_data.py

# Specific test groups
uv run pytest tests/test_models.py -v        # 16 model tests
uv run pytest tests/test_retrieval -v        # 49 retrieval tests
uv run pytest tests/test_agent.py -v         # 22 agent tests
uv run pytest tests/test_api.py -v           # 23 API tests
uv run pytest tests/test_evaluation.py -v -s  # 16 benchmark tests
```

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
benchmark (CC BY-NC 4.0) with 42,994 products, 480 queries, and 233,448
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
| [EVALUATION_ANNOTATION_GUIDE.md](docs/EVALUATION_ANNOTATION_GUIDE.md) | How to manually review eval queries |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TAVILY_API_KEY` | No | — | Enables web search tool |
| `OPENAI_API_KEY` | No | — | For real LLM mode (OpenAI) |
| `ANTHROPIC_API_KEY` | No | — | For real LLM mode (Anthropic) |
| `MARKETLENS_DATABASE_URL` | No | `sqlite:///marketlens.db` | Database connection string |

## Known Limitations

- **Fixture data only**: 20 hand-crafted products; not real Amazon Reviews data
- **Fake LLM**: Rule-based; swap to real LLM for production
- **No streaming**: Synchronous API responses
- **In-memory catalog**: No distributed index
- **No auth**: Open API; add auth for production

## License

MIT — Based on Open Deep Research by LangChain (Lance Martin).

---

*Built for AI Application, AI Backend, RAG/Search, and Agent Engineer roles.*
