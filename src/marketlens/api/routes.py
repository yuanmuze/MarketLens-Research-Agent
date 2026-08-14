"""FastAPI route handlers for MarketLens Research API."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from marketlens.agent.models import (
    AgentRequest,
    AgentResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from marketlens.agent.orchestrator import AgentOrchestrator
from marketlens.agent.providers.base import FakeLLMClient, LLMClient
from marketlens.agent.tools import AgentTools
from marketlens.api.database import (
    ResearchJobRecord,
    SearchQueryRecord,
    get_session,
)
from marketlens.api.models import (
    HealthResponse,
    ResearchJobResponse,
    ResearchReportResponse,
    ResearchSubmitResponse,
    SearchResponse,
    SearchResultItem,
)
from marketlens.api.models import (
    ResearchRequest as APIResearchRequest,
)
from marketlens.catalog import ProductCatalog
from marketlens.config import MarketLensSettings
from marketlens.retrieval.semantic import SemanticBackendUnavailableError
from marketlens.retrieval.service import RetrievalService, SessionFactory

logger = logging.getLogger(__name__)

router = APIRouter()

# Global catalog and service (initialized at app startup)
_catalog: ProductCatalog | None = None
_service: RetrievalService | None = None
_settings = MarketLensSettings()
_startup_error = ""


def init_catalog(
    catalog: ProductCatalog,
    *,
    data_path: Path | None = None,
    settings: MarketLensSettings | None = None,
    session_factory: SessionFactory | None = None,
) -> bool:
    """Initialize the global catalog and retrieval service.

    Args:
        catalog: ProductCatalog instance.
        data_path: Path to product JSON (for embedding cache key).
        settings: Validated runtime settings. Defaults to environment values.
        session_factory: Optional injected PostgreSQL session factory.

    Returns:
        Whether the configured retrieval service initialized successfully.
    """
    global _catalog, _service, _settings, _startup_error
    _catalog = catalog
    _settings = settings or MarketLensSettings.from_env()
    _service = None
    _startup_error = ""

    # Fixture-based memory tests stay offline. pgvector never receives this
    # implicit fake override.
    use_fake = _settings.use_fake_embeddings or (
        data_path is None and _settings.semantic_backend == "memory"
    )
    if _settings.semantic_backend == "pgvector" and use_fake:
        _startup_error = (
            "pgvector requires a real 384-dimensional embedding model; "
            "fake embeddings are not allowed"
        )
        logger.error("Retrieval startup failed: invalid pgvector embedding configuration")
        return False

    if _settings.semantic_backend == "pgvector" and session_factory is None:
        from marketlens.persistence.engine import get_session_factory

        session_factory = get_session_factory()

    try:
        service = RetrievalService(
            catalog,
            data_path=data_path,
            use_fake_embeddings=use_fake,
            semantic_backend=_settings.semantic_backend,
            session_factory=session_factory,
            embedding_model_name=_settings.embedding_model,
        )
        service.initialize()
    except (OSError, RuntimeError, ValueError, SemanticBackendUnavailableError) as exc:
        _startup_error = f"retrieval unavailable: {type(exc).__name__}: {exc}"
        logger.error("Retrieval startup failed: %s", type(exc).__name__)
        return False

    _service = service
    return True


def mark_startup_unavailable(message: str) -> None:
    """Record a sanitized startup failure while keeping liveness available."""
    global _service, _startup_error
    _service = None
    _startup_error = message


def get_service() -> RetrievalService:
    """Get the global RetrievalService instance.

    Returns:
        RetrievalService instance.

    Raises:
        HTTPException: If not initialized.
    """
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail=_startup_error or "retrieval service not initialized",
        )
    return _service


def get_catalog() -> ProductCatalog:
    """Get the global catalog instance."""
    if _catalog is None:
        raise HTTPException(status_code=500, detail="Catalog not initialized")
    return _catalog


# ---------------------------------------------------------------------------
# GET /health  (backward-compatible summary)
# GET /health/live  (liveness)
# GET /health/ready (readiness)
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with retrieval service status."""
    catalog_size = len(_catalog) if _catalog else 0

    extra: dict[str, Any] = {}
    if _service is not None:
        extra = _service.status()
    else:
        extra = {
            "catalog_backend": _settings.catalog_backend,
            "semantic_backend": _settings.semantic_backend,
            "embedding_model": _settings.embedding_model,
            "semantic_index_ready": False,
            "startup_error": _startup_error or "service not initialized",
        }

    return HealthResponse(
        status="ok",
        version="0.1.0",
        catalog_size=catalog_size,
        **extra,
    )


@router.get("/health/live")
async def health_live() -> dict[str, Any]:
    """Liveness probe: process is running (does NOT check external deps)."""
    return {"status": "ok", "alive": True}


@router.get("/health/ready")
async def health_ready() -> dict[str, Any]:
    """Readiness probe: necessary config + PostgreSQL + retrieval backend.

    Returns 503 when the retrieval service or catalog is not ready.
    """
    if _service is None or _catalog is None:
        raise HTTPException(
            status_code=503,
            detail=_startup_error or "not ready: service not initialized",
        )

    # Check retrieval service is actually usable
    try:
        status = _service.status()
        retrieval_ready = status.get("retrieval_service_ready", False) and status.get(
            "semantic_index_ready", False
        )
    except Exception as e:
        logger.warning("Readiness check failed: %s", e)
        raise HTTPException(status_code=503, detail="not ready: retrieval unavailable") from e

    if not retrieval_ready:
        raise HTTPException(status_code=503, detail="not ready: retrieval not initialized")

    return {
        "status": "ready",
        "catalog_size": len(_catalog),
        "catalog_backend": _settings.catalog_backend,
        "semantic_backend": status.get("semantic_backend", "unknown"),
        "embedding_backend": status.get("embedding_backend", "unknown"),
        "embedding_model": status.get("embedding_model", "unknown"),
        "embedding_dim": status.get("embedding_dim", 0),
        "semantic_index_ready": status.get("semantic_index_ready", False),
        "semantic_indexed_count": status.get("semantic_indexed_count", 0),
    }


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------
@router.get("/search", response_model=SearchResponse)
async def search_products(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    strategy: str = Query(default="hybrid", description="Strategy: bm25, embedding, hybrid, rerank"),
    top_k: int = Query(default=20, ge=1, le=100, description="Max results"),
    candidate_k: int = Query(default=50, ge=1, le=200, description="Candidates for reranker"),
    max_budget: float | None = Query(default=None, ge=0, description="Max price"),
    min_price: float | None = Query(default=None, ge=0, description="Min price"),
    min_rating: float | None = Query(default=None, ge=0, le=5, description="Min rating"),
    brand: str | None = Query(default=None, description="Exact brand filter"),
) -> SearchResponse:
    """Search the product catalog using the specified retrieval strategy.

    Supports four strategies: bm25, embedding, hybrid, rerank.
    Structured filters: brand, price range, rating.
    """
    request_id = getattr(request.state, "request_id", f"req-{uuid.uuid4().hex[:12]}")

    try:
        service = get_service()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Search initialization error: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="search service unavailable") from e

    try:
        output = service.search(
            query=q,
            strategy=strategy,
            top_k=top_k,
            candidate_k=candidate_k,
            max_budget=max_budget,
            min_price=min_price,
            max_price=max_budget,
            min_rating=min_rating,
            brand=brand,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except SemanticBackendUnavailableError as e:
        logger.error("Semantic search unavailable: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="semantic backend unavailable") from e
    except Exception as e:
        logger.error("Search error: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="search failed") from e

    # Persist search query
    try:
        session = get_session()
        record = SearchQueryRecord(
            query=q,
            top_k=top_k,
            result_count=output.total_found,
            duration_ms=output.elapsed_ms,
            source=output.strategy,
        )
        session.add(record)
        session.commit()
    except Exception as e:
        logger.warning("Failed to persist search query: %s", e)

    items = [
        SearchResultItem(
            rank=item.rank,
            product_id=item.product_id,
            title=item.title,
            brand=item.brand,
            price=item.price,
            rating=item.rating,
            review_count=item.review_count,
            score=item.final_score,
            source=output.strategy,
        )
        for item in output.results
    ]

    return SearchResponse(
        request_id=request_id,
        query=q,
        results=items,
        total_results=len(items),
        duration_ms=output.elapsed_ms,
    )


# ---------------------------------------------------------------------------
# POST /research
# ---------------------------------------------------------------------------
@router.post("/research", response_model=ResearchSubmitResponse, status_code=201)
async def submit_research(request: APIResearchRequest) -> ResearchSubmitResponse:
    """Submit a research request (synchronous).

    The research is executed synchronously and the result is persisted.
    For async execution, use POST /research/jobs.
    """
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    # Persist the job
    try:
        session = get_session()
        record = ResearchJobRecord(
            job_id=job_id,
            request_id=request_id,
            query=request.query,
            status="pending",
            max_results=request.max_results,
            enable_web_search=1 if request.enable_web_search else 0,
        )
        session.add(record)
        session.commit()
    except Exception as e:
        logger.error("Failed to persist job: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create research job")

    # Execute research synchronously
    try:
        from marketlens.agent.graph import run_research

        catalog = get_catalog()
        t0 = time.monotonic()
        result = await run_research(request.query, catalog, request_id)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Update job record (values cast for SQLAlchemy Column compatibility)
        record.status = "completed"  # type: ignore[assignment]
        record.completed_at = datetime.now(timezone.utc)  # type: ignore[assignment]
        record.duration_ms = elapsed_ms  # type: ignore[assignment]
        record.report_text = str(result.get("final_report", ""))  # type: ignore[assignment]
        record.product_count = len(result.get("products", []))  # type: ignore[assignment]
        record.tool_calls = int(result.get("tool_calls", 0))  # type: ignore[assignment]
        record.evidence_count = len(result.get("evidence", []))  # type: ignore[assignment]
        record.constraints_satisfied = 1 if result.get("constraints_satisfied") else 0  # type: ignore[assignment]
        session.commit()

    except Exception as e:
        logger.error("Research execution error: %s", e, exc_info=True)
        record.status = "failed"  # type: ignore[assignment]
        record.error_message = str(e)  # type: ignore[assignment]
        session.commit()
        raise HTTPException(status_code=500, detail=f"Research failed: {e}")

    return ResearchSubmitResponse(
        job_id=job_id,
        request_id=request_id,
        status="completed",
        query=request.query,
        message="Research completed successfully",
    )


# ---------------------------------------------------------------------------
# POST /research/jobs
# ---------------------------------------------------------------------------
@router.post("/research/jobs", response_model=ResearchSubmitResponse, status_code=201)
async def create_research_job(request: APIResearchRequest) -> ResearchSubmitResponse:
    """Create a new research job (async-friendly).

    The job is persisted with status 'pending'. Execute it separately.
    """
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    try:
        session = get_session()
        record = ResearchJobRecord(
            job_id=job_id,
            request_id=request_id,
            query=request.query,
            status="pending",
            max_results=request.max_results,
            enable_web_search=1 if request.enable_web_search else 0,
        )
        session.add(record)
        session.commit()
    except Exception as e:
        logger.error("Failed to persist job: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create research job")

    return ResearchSubmitResponse(
        job_id=job_id,
        request_id=request_id,
        status="pending",
        query=request.query,
        message="Research job created. Execute to get results.",
    )


# ---------------------------------------------------------------------------
# GET /research/jobs/{job_id}
# ---------------------------------------------------------------------------
@router.get("/research/jobs/{job_id}", response_model=ResearchJobResponse)
async def get_job_status(job_id: str) -> ResearchJobResponse:
    """Get the status of a research job.

    Args:
        job_id: The job identifier.

    Returns:
        Job status and metadata.

    Raises:
        HTTPException 404: Job not found.
    """
    session = get_session()
    record = session.query(ResearchJobRecord).filter_by(job_id=job_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Extract typed values from SQLAlchemy columns for Pydantic
    return ResearchJobResponse(
        job_id=str(record.job_id),
        request_id=str(record.request_id),
        status=str(record.status),
        query=str(record.query),
        created_at=record.created_at,  # type: ignore[arg-type]
        started_at=record.started_at,  # type: ignore[arg-type]
        completed_at=record.completed_at,  # type: ignore[arg-type]
        duration_ms=float(record.duration_ms) if record.duration_ms is not None else None,
        product_count=int(record.product_count) if record.product_count is not None else None,
        tool_calls=int(record.tool_calls or 0),
        error_message=str(record.error_message) if record.error_message is not None else None,
    )


# ---------------------------------------------------------------------------
# GET /research/jobs/{job_id}/report
# ---------------------------------------------------------------------------
@router.get("/research/jobs/{job_id}/report", response_model=ResearchReportResponse)
async def get_job_report(job_id: str) -> ResearchReportResponse:
    """Get the report for a completed research job.

    Args:
        job_id: The job identifier.

    Returns:
        Complete research report.

    Raises:
        HTTPException 404: Job not found.
        HTTPException 400: Job not yet completed.
    """
    session = get_session()
    record = session.query(ResearchJobRecord).filter_by(job_id=job_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if record.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {record.status}",
        )

    duration = float(record.duration_ms) if record.duration_ms is not None else 0.0
    product_n = int(record.product_count) if record.product_count is not None else 0

    return ResearchReportResponse(
        job_id=str(record.job_id),
        query=str(record.query),
        summary=f"Research completed in {duration:.0f}ms. Found {product_n} products.",
        recommendations=[],
        comparisons=[],
        evidence=[],
        constraints_satisfied=bool(record.constraints_satisfied),
        generated_at=record.completed_at or datetime.now(timezone.utc),  # type: ignore[arg-type]
        report_text=str(record.report_text or ""),
    )


# ---------------------------------------------------------------------------
# POST /agent/recommend  (Phase 5 Agent)
# ---------------------------------------------------------------------------

class _NoOpLLM(LLMClient):
    """Placeholder LLM client — triggers degraded fallback immediately."""
    def send(self, messages, tools, *, timeout_s=30.0):
        raise ConnectionError("No LLM configured. Set MARKETLENS_AGENT_API_KEY.")
    @property
    def model_name(self):
        return "none"


def _build_llm_client() -> LLMClient:
    """Build LLM client from environment variables or return NoOp."""
    import os
    if _settings.use_fake_llm:
        return FakeLLMClient(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "phase8_fake_search",
                            "type": "function",
                            "function": {
                                "name": "search_catalog",
                                "arguments": (
                                    '{"query":"wireless headphones",'
                                    '"mode":"balanced","top_k":5}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "content": (
                        "Here are evidence-backed recommendations from the "
                        "local MarketLens catalog."
                    )
                },
            ],
            model_name="phase8-deterministic-fake",
        )
    api_key = os.environ.get("MARKETLENS_AGENT_API_KEY", "")
    base_url = os.environ.get("MARKETLENS_AGENT_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MARKETLENS_AGENT_MODEL", "gpt-4.1-mini")
    if not api_key:
        return _NoOpLLM()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)

        class OpenAILLMClient(LLMClient):
            def __init__(self, client, model):
                self._client = client
                self._model = model
            @property
            def model_name(self):
                return self._model
            def send(self, messages, tools, *, timeout_s=30.0):
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, tools=tools,
                    tool_choice="auto", timeout=timeout_s,
                )
                msg = resp.choices[0].message
                result = {"content": msg.content or ""}
                if msg.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ]
                return result
        return OpenAILLMClient(client, model)
    except ImportError:
        return _NoOpLLM()


@router.post("/agent/recommend", response_model=AgentResponse)
async def agent_recommend(request: AgentRequest) -> AgentResponse:
    """Product discovery agent — natural language to recommendations.

    Accepts natural language queries and returns evidence-backed
    product recommendations via an LLM-driven tool-calling loop.
    Records the run and tool calls to PostgreSQL when enabled.
    """
    service = get_service()
    if service is None:
        raise HTTPException(status_code=500, detail="Retrieval service not initialized")

    catalog = get_catalog()
    product_index = {p.product_id: p.model_dump() for p in catalog.get_all_products()}

    llm = _build_llm_client()
    tools = AgentTools(service)
    orch = AgentOrchestrator(llm, tools, product_index)

    # Record agent run to PostgreSQL (optional, depends on env config)
    return await _recorded_run(request, orch, tools, product_index)


async def _recorded_run(
    request: AgentRequest,
    orch: AgentOrchestrator,
    tools: AgentTools,
    product_index: dict,
) -> AgentResponse:
    """Run the agent and persist run + tool calls with correct transaction timing.

    Timing (correct):
      1. Short transaction A: create status=running record, commit/release.
      2. No DB transaction: execute LLM + tool calls.
      3. Short transaction B (success): write tool_calls + final status,
         commit together.
      4. Short transaction C (failure): update SAME record to failed.
    """
    import uuid

    # Idempotency: honor the client-provided request_id if present.
    request_id = request.request_id or f"req-{uuid.uuid4().hex[:12]}"
    request_hash = _compute_request_hash(request)

    # Check for an existing run with the same request_id.
    existing = _find_existing_run(request_id)
    if existing is not None:
        if existing.request_hash == request_hash:
            # Idempotent replay: return the stored result.
            if existing.response and existing.status in (
                "completed", "degraded", "no_results", "needs_clarification",
            ):
                return _response_from_record(existing)
        else:
            # Same request_id, different content → conflict.
            raise HTTPException(status_code=409, detail="request_id conflict: different request content")

    # 1. Create running record BEFORE agent execution (short tx A)
    run_id = _create_running_record(request_id, request, request_hash)

    try:
        response = orch.run(request)
    except Exception as e:
        # 4. Update the SAME running record to failed (short tx C)
        try:
            _mark_run_failed(run_id, e)
        except Exception:
            logger.exception("Failed to record agent failure")
        raise

    # 3. Update the SAME record to final status + write tool calls (short tx B)
    try:
        _mark_run_completed(run_id, response, orch)
    except Exception:
        logger.exception("Failed to record agent run completion")

    return response


def _compute_request_hash(request: AgentRequest) -> str:
    """Compute a stable content hash for idempotency/conflict detection."""
    import hashlib

    payload = f"{request.message}|{request.mode}|{request.max_results}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _find_existing_run(request_id: str):
    """Find an existing AgentRunRecord by request_id, or None."""
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import AgentRunRepository

    try:
        with session_scope() as session:
            repo = AgentRunRepository(session)
            return repo.get_by_request_id(request_id)
    except Exception:
        logger.exception("Failed to check existing run")
        return None


def _response_from_record(record) -> AgentResponse:
    """Reconstruct an AgentResponse from a stored AgentRunRecord."""
    stored = record.response or {}
    return AgentResponse(
        request_id=record.request_id,
        status=record.status,
        answer=stored.get("answer", ""),
        recommendations=[],
        mode_requested=record.mode_requested,
        mode_used=record.mode_used or record.mode_requested,
        degraded=record.degraded,
        warnings=[],
        tool_calls=0,
        latency_ms=float(record.latency_ms or 0.0),
    )


def _create_running_record(request_id: str, request: AgentRequest, request_hash: str) -> int | None:
    """Create a running record in a short transaction; return its DB id."""
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import AgentRunRepository

    try:
        with session_scope() as session:
            repo = AgentRunRepository(session)
            record = repo.create_running(
                request_id=request_id,
                user_query=request.message,
                mode_requested=request.mode,
                request_hash=request_hash,
            )
            session.commit()
            return record.id
    except Exception:
        logger.exception("Failed to create running record (persistence unavailable)")
        return None


def _mark_run_completed(
    run_id: int | None,
    response: AgentResponse,
    orch: AgentOrchestrator,
) -> None:
    """Update the running record to final status + write tool calls."""
    if run_id is None:
        return
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import AgentRunRepository

    with session_scope() as session:
        repo = AgentRunRepository(session)
        # Sanitized tool calls (no API keys, no hidden reasoning)
        repo.add_tool_calls(run_id, orch.tool_call_log)
        repo.mark_completed(
            run_id,
            response.status,
            response.mode_used,
            response.degraded,
            {
                "answer": response.answer,
                "recommendation_count": len(response.recommendations),
                "product_ids": [r.product_id for r in response.recommendations],
            },
            response.latency_ms,
        )


def _mark_run_failed(run_id: int | None, error: Exception) -> None:
    """Update the running record to failed with a sanitized error message."""
    if run_id is None:
        return
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import AgentRunRepository

    error_type = type(error).__name__
    safe_message = str(error)[:500]  # Sanitized: no API keys / hidden reasoning

    with session_scope() as session:
        repo = AgentRunRepository(session)
        repo.mark_failed(run_id, error_type, safe_message)


# ---------------------------------------------------------------------------
# POST /feedback  (Phase 7 minimal feedback loop)
# ---------------------------------------------------------------------------
@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record minimal user feedback on an agent run (idempotent)."""
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import FeedbackRepository

    try:
        with session_scope() as session:
            repo = FeedbackRepository(session)
            # Must reference an existing agent run
            if not repo.agent_run_exists(request.agent_run_id):
                raise HTTPException(status_code=404, detail="agent run not found")
            # Idempotency: skip if the same idempotency_key already recorded
            if request.idempotency_key and repo.idempotency_key_exists(request.idempotency_key):
                raise HTTPException(status_code=200, detail="already recorded")
            record = repo.create(
                agent_run_id=request.agent_run_id,
                feedback_type=request.feedback_type,
                reason=request.reason,
                idempotency_key=request.idempotency_key,
            )
            session.commit()
            return FeedbackResponse(feedback_id=record.id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Feedback submission failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="feedback storage unavailable") from e
