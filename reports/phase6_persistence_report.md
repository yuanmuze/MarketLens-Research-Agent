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

- Agent request start → `create_running` in its own short transaction
- Agent execution → runs OUTSIDE any DB transaction (LLM + tool calls
  can take seconds and should not hold a connection)
- Agent completion → a NEW transaction writes final response + tool calls
- Agent failure → rollback the failed write, then a NEW transaction
  marks the run `failed` with a sanitized error

**Why not wrap external model calls in a long DB transaction**: holding
a database transaction open during an LLM call (seconds) blocks other
writers, risks connection exhaustion, and can cause long lock waits.
Instead: short transactions only at the persistence points.

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
