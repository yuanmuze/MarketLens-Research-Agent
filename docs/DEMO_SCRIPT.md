# MarketLens Demo Script

Step-by-step walkthrough to demonstrate the MarketLens system. No API keys required.

## Prerequisites

```bash
git clone <this-repo>
cd MarketLens-Research-Agent
uv sync --python 3.11
uv pip install mypy ruff
```

## Step 1: Verify Tests (30 seconds)

```bash
uv run pytest tests/ -q --ignore=tests/extract_langsmith_data.py
```

Expected: 166 passed. Shows the system is healthy.

## Step 2: Run Evaluation (30 seconds)

```bash
uv run pytest tests/test_evaluation.py -v -s
```

Shows actual evaluation numbers with fixture data. Highlights the comparison between BM25, embedding, and hybrid retrieval.

## Step 3: Start the API (30 seconds)

```bash
uv run uvicorn marketlens.api.main:app --reload
```

Console shows:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Loaded 20 products from fixture
```

## Step 4: Test Health and Search (1 minute)

Open browser to `http://127.0.0.1:8000/docs` (OpenAPI docs) or use curl:

```bash
# Health check
curl http://127.0.0.1:8000/health

# Search for headphones
curl "http://127.0.0.1:8000/search?q=wireless+headphones&top_k=5"

# Search with budget
curl "http://127.0.0.1:8000/search?q=noise+cancelling&max_budget=200"

# Search with brand
curl "http://127.0.0.1:8000/search?q=earbuds&brand=Sony"
```

Point out: scores, ranks, sources, request_id headers.

## Step 5: Submit Research (1 minute)

```bash
# Full research with budget constraint
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best wireless noise cancelling headphones under $350",
    "max_results": 5
  }'

# Research with brand preferences
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best earbuds for calls and music",
    "preferred_brands": ["Sony", "Apple"],
    "max_results": 5
  }'
```

Copy the `job_id` from the response.

## Step 6: Check Job and Report (1 minute)

```bash
# Check job status
curl http://127.0.0.1:8000/research/jobs/<job_id>

# Get full report
curl http://127.0.0.1:8000/research/jobs/<job_id>/report
```

Point out: report structure, evidence references, constraint satisfaction.

## Step 7: Demonstrate Edge Cases (1 minute)

```bash
# Empty query → 422
curl "http://127.0.0.1:8000/search?q="

# Non-existent job → 404
curl http://127.0.0.1:8000/research/jobs/nonexistent

# Impossible budget → empty results
curl "http://127.0.0.1:8000/search?q=headphones&max_budget=1"
```

## Step 8: Show Code Highlights (2 minutes)

1. **Domain models**: `src/marketlens/models.py` — 10 Pydantic v2 models
2. **Hybrid retrieval**: `src/marketlens/retrieval/hybrid.py` — RRF fusion
3. **Agent graph**: `src/marketlens/agent/graph.py` — 8-node LangGraph
4. **FakeLLM**: `src/marketlens/agent/fake_llm.py` — offline agent
5. **API**: `src/marketlens/api/routes.py` — 6 endpoints

## Step 9: Run Docker Compose Check (30 seconds)

```bash
# Validate config
python -c "
import yaml
with open('docker-compose.yml') as f:
    config = yaml.safe_load(f)
print(f'Services: {list(config[\"services\"].keys())}')
"

# If Docker is available:
# docker compose up -d
# docker compose ps
# docker compose down
```

## Step 10: Lint and Type Check (30 seconds)

```bash
uv run ruff check src/marketlens/ tests/
echo "Ruff: clean"
```

## Key Talking Points During Demo

1. **"This is built on Open Deep Research"** — Shows ability to understand and extend existing codebases
2. **"No API keys needed"** — Emphasizes offline-first design for testing
3. **"Every recommendation links to evidence"** — Shows production-quality reliability thinking
4. **"Hard constraints are Python, not LLM"** — Demonstrates understanding of LLM limitations
5. **"166 tests, all green"** — Shows testing discipline
6. **"Swap SQLite for pgvector in production"** — Shows understanding of dev/prod differences

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8000 already in use | `uv run uvicorn marketlens.api.main:app --port 8001` |
| Database locked | Delete `marketlens.db` in project root |
| Module not found | Run `uv sync` to rebuild |
| Test failures | Run `uv pip install mypy ruff` to ensure dev deps |
