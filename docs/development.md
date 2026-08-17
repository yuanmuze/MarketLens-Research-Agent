# Development

## Requirements

- Python 3.12
- uv 0.12.3 or compatible
- Docker with Compose for PostgreSQL/pgvector and image validation

Install all contributor dependencies:

```bash
uv sync --locked --group dev --extra data --extra embeddings --extra llm
```

For the API without data preparation or a live provider, omit `data` and
`llm`. The default deterministic provider never sends an external LLM request.

## Configuration

Copy `.env.example` and set only the values required by the selected backend.
Configuration is grouped into application, database, retrieval, embeddings,
optional LLM provider, and CORS/HTTP settings.

The simplest local process uses the bundled JSON fixture and deterministic
embeddings:

```bash
MARKETLENS_USE_FAKE_EMBEDDINGS=true uv run uvicorn marketlens.api.main:app --reload
```

On PowerShell, set the variable first:

```powershell
$env:MARKETLENS_USE_FAKE_EMBEDDINGS = "true"
uv run uvicorn marketlens.api.main:app --reload
```

## PostgreSQL and pgvector

Start the database with `docker compose up -d db`. Apply migrations with a
database URL that points to the intended local database:

```bash
uv run alembic upgrade head
uv run alembic current
uv run alembic check
```

Import a catalog with `scripts/import_products.py`, then create embeddings with
`scripts/index_product_embeddings.py`. WANDS and ESCI have dedicated download,
verification, preparation, and indexing scripts. Raw data, derived examples,
query judgments, and embedding arrays are ignored and must remain local.

The Compose API defaults to PostgreSQL catalog storage, pgvector semantic
search, real local embeddings, and the deterministic LLM provider. Readiness
will fail on an empty database until products are imported and fully indexed.
Published database and API ports bind only to `127.0.0.1`.

## Tests and quality gates

```bash
uv run ruff check .
uv run mypy src scripts
uv run pytest -m "not postgres" -q -rs
uv run pytest -m postgres -q -rs
uv run alembic check
uv build
docker compose config
docker compose build
```

Set `MARKETLENS_TEST_DATABASE_URL` for PostgreSQL tests. Its database name must
contain `test`; fixtures intentionally drop and recreate application tables.
Never point it at the Compose development database or any non-test data.

Inspect the wheel after `uv build`: it must contain only `marketlens`, package
metadata, and required fixture/package-data files. Tests, scripts, datasets,
benchmarks, reports, and model caches must not be packaged.

## Container image

The Dockerfile installs dependencies from `uv.lock`; it does not maintain a
second package list. The embeddings extra resolves CPU-only PyTorch on Linux
and Windows. Two Hugging Face snapshots are pinned and restricted to the
PyTorch files needed for embedding and reranking. This increases image size but
allows offline runtime after the build. The process runs as `appuser`.

## Operational checks

After startup, verify `/health/live`, `/health/ready`, `/search`, the selected
agent endpoint, feedback, and persisted job/run records. The local API benchmark
is `scripts/benchmark_api.py`; its results are development measurements, not a
service-level objective.
