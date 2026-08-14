# Upstream Baseline

> **Historical Phase 0 snapshot.** Versions, paths, and environment details in
> this document describe the imported upstream baseline, not the current
> MarketLens public entry point.

## Upstream Information

- **Upstream Repository**: https://github.com/langchain-ai/open_deep_research
- **Upstream Commit SHA**: `1b7d2e80db9faa586165c60e09096dbbfd483a64`
- **License**: MIT
- **Original Author**: Lance Martin (LangChain)
- **Base Version**: 0.0.16

## Upstream Core Capabilities

Open Deep Research is a configurable, fully open-source deep research agent that works across multiple model providers, search tools, and MCP (Model Context Protocol) servers:

1. **Multi-LLM Support**: OpenAI, Anthropic, Google, Groq, DeepSeek, and more via `init_chat_model()`
2. **Search API Integration**: Tavily, OpenAI native search, Anthropic native search, DuckDuckGo, Exa
3. **MCP Server Support**: Model Context Protocol for extended tool capabilities
4. **LangGraph Workflow**: Structured agent workflow with supervisor-researcher architecture
5. **Parallel Research**: Multiple sub-agents conducting concurrent research on subtopics
6. **Report Generation**: Structured final report with citations and sources
7. **LangGraph Studio UI**: Web-based configuration and testing interface
8. **LangSmith Integration**: Evaluation and tracing via LangSmith platform

## Original Test Results

- **Main tests/ directory**: Contains evaluation scripts (`run_evaluate.py`, `evaluators.py`, etc.) that require API keys and LangSmith access. These are integration/evaluation scripts, not pytest unit tests.
- **Legacy tests** (`src/legacy/tests/`): Contain pytest tests but require API keys to run.
- **pytest collection**: 0 items collected from `tests/` directory.
- **Test infrastructure**: Uses pytest with plugins (asyncio, benchmark, codspeed, recording, socket, syrupy).

## What MarketLens Preserves

- MIT License and copyright notices
- Original `src/legacy/` implementations (for reference)
- Original `src/security/auth.py` (LangGraph auth handler)
- Original `CLAUDE.md` content was retained during the baseline phase
- Original `examples/` directory
- Base LangGraph patterns and state management concepts from upstream

## What MarketLens Modifies

- `pyproject.toml`: Updated project name, description, dependencies for MarketLens
- `CLAUDE.md`: Was replaced with MarketLens contributor instructions
- `src/open_deep_research/`: Was retained as attributed upstream reference code
- `tests/`: Was extended with the MarketLens-specific offline and PostgreSQL suites

## What MarketLens Adds (New)

- `src/marketlens/`: Core MarketLens package with domain models, retrieval, agent, API
- `docs/`: Architecture, evaluation, learning, and implementation documentation
- Product catalog with BM25 + embedding + hybrid retrieval
- Evidence-grounded product research agent (LangGraph)
- FastAPI research API with SQLAlchemy persistence
- Standalone pytest suite (no API keys required)
- Structured evaluation benchmarks
- Docker Compose for PostgreSQL/pgvector deployment
- GitHub Actions CI/CD

## Environment Setup (Baseline)

- **Python**: 3.11 (matches `langgraph.json` requirement)
- **Package Manager**: uv (lock file) and pip
- **Key Dependencies**: langgraph, langchain, openai, anthropic, tavily, mcp
- **Dev Dependencies**: mypy, ruff
- **223 packages installed** via `uv sync --python 3.11`
