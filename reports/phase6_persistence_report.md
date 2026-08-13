# Phase 6: PostgreSQL Persistence & Agent Run Audit

## Overview

Phase 6 adds PostgreSQL persistence for product data and agent run
auditing without changing Phase 1-5 retrieval/agent behavior.

Three new tables (products, agent_runs, agent_tool_calls), a Repository
layer, Alembic migrations, and a product import script. The existing
JSON catalog mode remains the default.

## Why Three Tables

| Table | Purpose |
|-------|---------|
| `products` | Product catalog (JSON → PostgreSQL) |
| `agent_runs` | Agent request lifecycle (one row per request) |
| `agent_tool_calls` | Per-tool-call records (FK → agent_runs) |

Relationship: one agent run has zero-to-many tool calls. Deleting an
agent run cascades to its tool calls (`ON DELETE CASCADE`).

## Why Repository

Raw SQL / ORM queries are isolated behind `ProductRepository` and
`AgentRunRepository`. API routes and the agent orchestrator call these
repositories instead of embedding SQL. This:

- Keeps business logic decoupled from storage details
- Makes transaction boundaries explicit
- Enables unit testing with in-memory SQLite

## Transaction Boundaries

Correct agent run timing (verified in Phase 6.1):

```text
Short transaction A:  create status=running record → commit/release
No DB transaction:    execute LLM + tool calls
Short transaction B:  write tool_calls + final status → commit together
   (or, on failure)
Short transaction C:  update SAME record → status=failed → commit
```

The running record is created BEFORE agent execution, so it is queryable
during execution. The SAME record is updated to `completed`/`degraded`/
`no_results`/`needs_clarification` (or `failed`), never a duplicate row.

**Why not wrap external model calls in a long DB transaction**: holding
a database transaction open during an LLM call (seconds) blocks other
writers, risks connection exhaustion, and can cause long lock waits.
Instead: short transactions only at the persistence points.

## Driver Usage

- SQLAlchemy uses a **synchronous** `Engine` (`create_engine`) and
  synchronous `Session`.
- PostgreSQL driver: **psycopg2** (via `postgresql+psycopg2://`).
- Alembic uses the same synchronous engine, so it also uses psycopg2.
- `asyncpg` was removed in Phase 6.1 — it was declared but never used
  (the persistence layer is fully synchronous).

## PostgreSQL Validation Status (Phase 6.2 — real DB freeze)

**PASSED — PostgreSQL 16.14 via Docker Compose.**

- Docker Desktop 29.7.2, Docker Compose v5.3.1
- Service: `db` (postgres:16), port 5432→5432, healthy
- Test database: `marketlens_test` (separate from `marketlens` dev DB)
- Driver: psycopg2 (synchronous)

### Migration (real PostgreSQL)

```
alembic upgrade head   → Running upgrade -> 0001_initial ✓
alembic current        → 0001_initial (head) ✓
alembic check          → No new upgrade operations detected ✓
alembic downgrade base → Running downgrade ✓ (dedicated test DB only)
alembic upgrade head   → Running upgrade ✓
```

ORM metadata and migration are consistent (fixed a `request_id`
UniqueConstraint vs unique-index drift in the migration).

### Verified schema (via SQLAlchemy Inspector)

- Tables: products, agent_runs, agent_tool_calls
- Primary keys: products.product_id, agent_runs.id, agent_tool_calls.id
- request_id unique index (ix_agent_runs_request_id)
- agent_run_id FK with ON DELETE CASCADE
- Indexes: brand, price, rating, request_id, agent_run_id
- JSONB fields: metadata, constraints, response, arguments, result_product_ids
- Numeric fields: price NUMERIC(12,2), rating NUMERIC(3,2), latency_ms NUMERIC(12,2)

### PostgreSQL integration tests (11 passed)

```
uv run pytest -m postgres -q → 11 passed (not skipped)
```

Covers: upsert idempotency, ORM→Pydantic, JSONB roundtrip, Decimal/Numeric
precision, request_id unique constraint, ON DELETE CASCADE, transaction
rollback, running→completed/failed, postgres catalog backend, Fake LLM
API request recording AgentRun + ToolCall.

## Error Sanitization

Only `error_type` (exception class name) and a truncated `error_message`
are stored. Never persisted: API keys, Authorization headers, hidden
reasoning, or full provider responses.

## Why Still In-Memory Retrieval

Phase 6 does NOT rewrite BM25/vector/rerank as SQL, and does NOT add
pgvector. The `postgres` catalog backend loads products from the database,
converts them to in-memory Pydantic `Product` objects, then builds the
existing in-memory retrieval index. Retrieval and agent logic (Phase 1-5)
remain unchanged. pgvector would be a future optimization for large-scale
vector search, out of scope here.

## Database Failure vs Agent Failure

- **DB failure** (connection error, constraint violation): caught and
  logged; does NOT break the API response. Persistence is best-effort.
- **Agent failure** (LLM error, tool error): the run is recorded with
  `status=failed` and sanitized error details; the API returns the
  degraded/failed response to the caller.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MARKETLENS_DATABASE_URL` | `sqlite:///marketlens_persistence.db` | DB connection |
| `MARKETLENS_CATALOG_BACKEND` | `json` | `json` or `postgres` |

`json` (default) loads catalog from JSON file. `postgres` loads from
the `products` table via `ProductRepository`.
