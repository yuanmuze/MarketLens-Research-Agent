# MarketLens contributor instructions

## Repository layout

- `src/marketlens/`: supported application package.
- `tests/`: automated application and PostgreSQL integration tests.
- `scripts/`: data preparation, indexing, benchmark, and operational commands.
- `benchmarks/`: frozen machine-readable manifests and results.
- `docs/`: architecture, development, and evaluation documentation.
- `alembic/`: database migrations; existing revisions are immutable.

## Toolchain and commands

Use Python 3.12 and uv. Install the complete contributor environment with:

```text
uv sync --locked --group dev --extra data --extra embeddings --extra llm
```

Before completing a change, run:

```text
uv run ruff check .
uv run mypy src scripts
uv run pytest -m "not postgres" -q -rs
uv run pytest -m postgres -q -rs
uv run alembic check
uv build
docker compose config
```

Use `docker compose build` when packaging or runtime dependencies change.

## Safety and evidence rules

- PostgreSQL tests may connect only to a dedicated database whose name contains
  `test`. Never run destructive fixtures against development data.
- Never modify existing Alembic revisions `0001` through `0004`. Add a new
  migration for future schema changes.
- Default development and test flows must use the deterministic provider. Do
  not invoke a real LLM, paid API, LangSmith, or external evaluation service.
- Do not commit raw or processed datasets, human labels, model weights,
  embedding caches, local databases, environment files, tokens, or secrets.
- Treat files under `benchmarks/` as frozen evidence. Change result values only
  for a separately scoped, reproducible experiment; document the source,
  configuration, and reason.
- Keep `compose.yaml` as the only Compose entry point and bind published ports
  to localhost by default.
- Verify `git diff --check`, parse configuration files, check documentation
  links, and inspect wheel contents before delivery.
