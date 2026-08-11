# MarketLens Learning Guide

Adapted for AI Application, AI Backend, RAG/Search, and Agent Engineer job interviews.

## Recommended Reading: 10 Core Files (in order)

1. **`src/marketlens/models.py`** — All Pydantic v2 domain models. Understand Product, SearchResult, ProductEvidence, ResearchReport
2. **`src/marketlens/catalog.py`** — ProductCatalog with filtering. See how hard constraints work
3. **`src/marketlens/retrieval/bm25.py`** — Okapi BM25 implementation. Core IR algorithm
4. **`src/marketlens/retrieval/embedding.py`** — Embedding interface + FakeEmbeddingBackend. See pluggable backends
5. **`src/marketlens/retrieval/hybrid.py`** — RRF hybrid retrieval. How BM25 + embeddings combine
6. **`src/marketlens/agent/graph.py`** — LangGraph workflow. The 8-node agent state machine
7. **`src/marketlens/agent/fake_llm.py`** — FakeLLM. How to build an offline agent for testing
8. **`src/marketlens/agent/tools.py`** — LangChain tools. How the agent calls catalog/web search
9. **`src/marketlens/api/routes.py`** — FastAPI endpoints. Full CRUD for research
10. **`src/marketlens/api/database.py`** — SQLAlchemy persistence. ORM patterns

## One Request's Complete Call Chain

```
User: "best Sony noise cancelling headphones under $300"

1. POST /research → routes.py:submit_research()
   → Creates ResearchJobRecord (status=pending)
   → Calls run_research(query, catalog, request_id)

2. agent/graph.py:parse_request_node
   → FakeLLM.parse_request("best Sony...")
   → Returns: {search_query, budget=300, preferred_brands=["sony"], constraints}

3. agent/graph.py:retrieve_catalog_node
   → HybridRetriever.search(SearchQuery(text=..., filters=UserConstraints(max_budget=300)))
   → BM25Retriever.search("best Sony noise cancelling...")
   → EmbeddingRetriever.search("best Sony noise cancelling...")
   → RRF fusion (k=60) → sorted product IDs
   → catalog.filter_by_constraints(max_budget=300) → hard filter
   → Returns: [SearchResult(product=B001, score=0.95), ...]

4. agent/graph.py:assess_evidence_node
   → FakeLLM.assess_evidence(products, query)
   → Returns: [ProductEvidence(product_id="B001", relevance=0.92), ...]

5. agent/graph.py:compare_products_node
   → FakeLLM.compare_products(products, query)
   → Returns: [{pros: [...], cons: [...], recommendation_score: 8.5}, ...]

6. agent/graph.py:validate_constraints_node
   → catalog.filter_by_constraints(max_budget=300)
   → Pure Python: check each product.price <= 300
   → Returns: {constraints_satisfied: True, violations: []}

7. agent/graph.py:generate_report_node
   → FakeLLM.generate_report(query, products, comparisons, validation)
   → Returns markdown string

8. routes.py:submit_research()
   → Updates ResearchJobRecord (status=completed, report_text=...)
   → Returns {job_id, status: "completed"}
```

## BM25, Embedding, Hybrid, and Reranker

| Method | How it works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **BM25** | Term frequency × inverse document frequency with saturation | Exact keyword matches; fast; deterministic | No semantic understanding; vocabulary mismatch |
| **Embedding** | Cosine similarity between dense vectors | Semantic similarity; synonym handling | Computationally heavier; embedding quality matters |
| **Hybrid (RRF)** | Weighted rank fusion: `Σ w/(k+rank)` | Best of both worlds; robust | Slightly more complex; needs weight tuning |
| **Reranker** | Cross-encoder or feature-based rescoring | Can dramatically improve precision | Adds latency; can over-fit |

## LangGraph: State, Node, Edge, Tool Calling

### State
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    query: str                                # Original user query
    search_results: list[SearchResult]        # Accumulated as agent runs
    evidence: list[ProductEvidence]           # Evidence per product
    comparisons: list[ComparisonItem]         # Generated comparisons
    final_report: str                         # Output
    status: str                               # running/completed/failed
```

### Nodes
Functions that take `state` → return `dict` (partial state update):
```python
def retrieve_catalog_node(state: AgentState) -> dict:
    results = retriever.search(SearchQuery(text=state["search_query"]))
    return {"search_results": results, "products": [r.product for r in results]}
```

### Edges
- **Normal edges**: Always go to the same next node
- **Conditional edges**: Route to different nodes based on state:
```python
def route_after_parse(state: AgentState) -> Literal["retrieve_catalog", "handle_failure"]:
    if state.get("error"):
        return "handle_failure"
    return "retrieve_catalog"
```

### Tool Calling
Tools are functions wrapped with `@tool` decorator:
```python
@tool
def search_catalog(query: str, top_k: int = 10) -> str:
    """Search the product catalog."""
    ...
```

The LLM decides when to call tools. In MarketLens, FakeLLM simulates this decision.

## FastAPI, Pydantic, SQLAlchemy Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **FastAPI** | HTTP routing, request validation, response serialization, middleware, error handling |
| **Pydantic** | Data validation, serialization, type coercion, schema generation (OpenAPI) |
| **SQLAlchemy** | ORM, database abstraction, query building, migration support, connection pooling |

## 20 High-Frequency Interview Questions

### Architecture & Design

1. **Q**: Why LangGraph over a simple function chain?
   **A**: LangGraph provides conditional routing, state management, retry logic, and observability out of the box. For a multi-step agent with error handling, it's more maintainable than raw async functions.

2. **Q**: Why single-agent instead of multi-agent?
   **A**: Product research is a linear pipeline. Multi-agent adds coordination overhead without benefit here. Simple is better when it meets requirements.

3. **Q**: How do you ensure recommendation quality?
   **A**: Triple safety net: (1) hybrid retrieval (BM25 + embedding), (2) deterministic constraint validation (Python, not LLM), (3) evidence traceability (every recommendation → ProductEvidence).

### Retrieval & Search

4. **Q**: Explain RRF. Why k=60?
   **A**: RRF = Reciprocal Rank Fusion. `score(d) = Σ w/(k+rank)`. k=60 is the standard from the original paper—it provides enough smoothing that rank differences matter but not so much that only #1 counts. Tested empirically.

5. **Q**: BM25 vs TF-IDF?
   **A**: BM25 adds term frequency saturation (k1 parameter) and document length normalization (b parameter). TF-IDF has no saturation—a term appearing 100 times is 100× more important, which is unrealistic.

6. **Q**: Why fake embeddings instead of real ones?
   **A**: Fake embeddings enable offline testing without model downloads. They're deterministic (same text → same vector), fast, and prove the architecture works. Swap in sentence-transformers for production.

7. **Q**: How do you handle vocabulary mismatch?
   **A**: Hybrid retrieval: BM25 catches exact matches, embeddings catch synonyms. RRF ensures both contribute. Example: "earbuds" vs "in-ear headphones"—different words, same product category.

### Agent Engineering

8. **Q**: What is FakeLLM and why use it?
   **A**: FakeLLM is a rule-based LLM simulator. It parses queries with regex (budget extraction), assesses relevance with Jaccard similarity, and generates templated reports. It proves the agent pipeline works without API keys or cost.

9. **Q**: How do you prevent the agent from hallucinating?
   **A**: (1) Product data comes from the catalog, not LLM generation; (2) Constraints validated by Python, not LLM; (3) Evidence linked to specific product IDs; (4) FakeLLM only rearranges catalog data—it can't invent products.

10. **Q**: Tool failure handling?
    **A**: Each tool call is wrapped in try/except. Timeouts (30s), max tool calls (10), and retries (3). Web search gracefully degrades to "Web search disabled" without API keys.

### API & Backend

11. **Q**: Why SQLite for testing, PostgreSQL for production?
    **A**: SQLite needs zero config—ideal for pytest and local dev. PostgreSQL + pgvector enables production-scale vector search. SQLAlchemy abstracts the difference.

12. **Q**: How does request_id flow through the system?
    **A**: FastAPI middleware adds `X-Request-ID` to every response. The agent passes it through state for log correlation. Search/research endpoints generate UUID-based IDs.

13. **Q**: Error handling strategy?
    **A**: Global exception handler catches all unhandled errors → 500 with structured JSON (no stack traces). Per-endpoint try/except for expected errors → 4xx. LangGraph has a dedicated `handle_failure` node.

14. **Q**: How would you add authentication?
    **A**: Add FastAPI dependency injection with JWT/OAuth2. Check `Authorization: Bearer <token>` in a middleware or dependency. LangGraph has built-in auth at `src/security/auth.py`.

### Evaluation & Reliability

15. **Q**: Why Recall@10 and nDCG@10?
    **A**: Standard IR metrics. Recall@10 = "did we find relevant items?" nDCG@10 = "are they ranked well?" Both needed—high recall with bad ranking is still a bad search experience.

16. **Q**: What does constraint satisfaction rate measure?
    **A**: Fraction of queries where all hard constraints (budget, brand, rating) are met by ALL returned results. Being right for some products isn't enough—all results must satisfy constraints.

17. **Q**: How do you test without real data?
    **A**: Fixture catalog (20 products) + fixture queries (12) + ground truth labels. All tests are deterministic and require no external services. Explicitly labeled as fixture benchmarks.

### General

18. **Q**: Biggest technical challenge?
    **A**: Building a complete agent pipeline that works without any API keys. Required FakeLLM, fake embeddings, offline BM25, and careful separation of "things LLM does" from "things Python checks."

19. **Q**: What would you do differently with more time?
    **A**: (1) Replace FakeLLM with real Claude/GPT, (2) pgvector for production retrieval, (3) real Amazon Reviews data, (4) streaming response, (5) A/B testing framework.

20. **Q**: Show me a Pydantic model with validation.
    **A**: `Product` validates: product_id non-empty, price ≥ 0, rating 0–5, review_count ≥ 0. `UserConstraints` validates min_budget ≤ max_budget. All validated at construction time via `@field_validator`.

## 7-Day Learning Plan (2-3 hours/day)

### Day 1: Understand the Domain
- Read `src/marketlens/models.py` — all Pydantic models
- Read `docs/ARCHITECTURE.md` — system overview
- Run `uv run pytest tests/test_models.py -v`
- **Goal**: Understand what data flows through the system

### Day 2: Master Retrieval
- Read `src/marketlens/retrieval/bm25.py` — BM25 algorithm
- Read `src/marketlens/retrieval/embedding.py` — embedding interface
- Read `src/marketlens/retrieval/hybrid.py` — RRF fusion
- Run `uv run pytest tests/test_bm25.py tests/test_embedding.py tests/test_hybrid.py -v`
- **Goal**: Understand how search works end-to-end

### Day 3: Agent Workflow
- Read `src/marketlens/agent/state.py` — state definitions
- Read `src/marketlens/agent/tools.py` — LangChain tools
- Read `src/marketlens/agent/graph.py` — LangGraph workflow
- Run `uv run pytest tests/test_agent.py -v`
- **Goal**: Understand the 8-node agent state machine

### Day 4: FakeLLM & Testing
- Read `src/marketlens/agent/fake_llm.py` — how fake LLM works
- Study how `parse_request`, `assess_evidence`, `compare_products` are simulated
- Run `uv run pytest tests/test_agent.py::TestFakeLLM -v`
- **Goal**: Understand offline agent architecture

### Day 5: API & Persistence
- Read `src/marketlens/api/routes.py` — endpoints
- Read `src/marketlens/api/database.py` — SQLAlchemy
- Run `uv run pytest tests/test_api.py -v`
- Start the API: `uv run uvicorn marketlens.api.main:app --reload`
- Visit `http://127.0.0.1:8000/docs` for OpenAPI
- **Goal**: Test the API with curl/OpenAPI

### Day 6: Evaluation & CI
- Read `src/marketlens/evaluation/benchmark.py` — metrics
- Run `uv run pytest tests/test_evaluation.py -v -s`
- Read `.github/workflows/ci.yml`
- **Goal**: Understand how quality is measured

### Day 7: Review & Interview Prep
- Read `docs/LEARNING_GUIDE.md` — 20 interview questions
- Review all 10 core files
- Run full test suite: `uv run pytest tests/ -v`
- Run ruff: `uv run ruff check src/marketlens/ tests/`
- Practice: Start the API and make research requests
- **Goal**: Ready to discuss the project in an interview
