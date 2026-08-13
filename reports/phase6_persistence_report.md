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

## PostgreSQL Validation Status (Phase 6.1)

**BLOCKED — no PostgreSQL server available in this environment.**
No Docker, no `psql`/`pg_isready`, no local PostgreSQL install, no
listening port on 5432/5433. The real-DB freeze validation (migration
upgrade/downgrade/check against PostgreSQL, and running the `-m postgres`
integration tests) could not be executed.

What was verified instead:
- Migration upgrade/downgrade/check against a dedicated SQLite test DB
- 14 repository unit tests (in-memory SQLite) covering upsert idempotency,
  FK cascade, transaction rollback, request_id uniqueness, and
  running→completed/failed same-record timing
- 4 PostgreSQL integration tests are written and correctly skip when
  `MARKETLENS_TEST_DATABASE_URL` is unset (with a safety guard asserting
  postgresql dialect + database name containing "test")

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
