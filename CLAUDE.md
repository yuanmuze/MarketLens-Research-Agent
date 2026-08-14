# MarketLens contributor instructions

MarketLens is an offline-first FastAPI product research system with PostgreSQL,
pgvector, BM25, RRF hybrid retrieval, and a deterministic Fake LLM demo path.

- Use Python 3.12 and `uv sync --locked --extra dev --extra db --extra embeddings`.
- Keep default development and test flows offline; never require provider keys.
- Run `ruff`, `mypy`, unit tests, and the dedicated PostgreSQL marker suite.
- PostgreSQL integration tests may connect only to database names containing
  `test`.
- Do not modify frozen Alembic migrations 0001-0004 or frozen WANDS/ESCI
  results without an explicitly scoped new experiment.
- Do not commit raw datasets, embedding/model caches, local labels, secrets, or
  `.claude/` configuration.
- Use `compose.yaml` as the only Compose entry point.
- Vendored Open Deep Research code is retained for attribution/reference; the
  supported public application entry point is `marketlens.api.main:app`.
