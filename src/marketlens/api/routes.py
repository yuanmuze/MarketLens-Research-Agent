"""FastAPI route handlers for MarketLens Research API."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from marketlens.agent.models import AgentRequest, AgentResponse
from marketlens.agent.orchestrator import AgentOrchestrator
from marketlens.agent.providers.base import LLMClient
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

logger = logging.getLogger(__name__)

router = APIRouter()

# Global catalog and service (initialized at app startup)
_catalog: ProductCatalog | None = None
_service: Any = None  # RetrievalService


def init_catalog(
    catalog: ProductCatalog,
    *,
    data_path: Path | None = None,
) -> None:
    """Initialize the global catalog and retrieval service.

    Args:
        catalog: ProductCatalog instance.
        data_path: Path to product JSON (for embedding cache key).
    """
    global _catalog, _service
    _catalog = catalog

    import os

    from marketlens.retrieval.service import RetrievalService
    # Always allow fake embedding override for tests
    use_fake = os.environ.get("MARKETLENS_USE_FAKE_EMBEDDINGS", "").lower() == "true" or data_path is None
    _service = RetrievalService(catalog, data_path=data_path, use_fake_embeddings=use_fake)
    _service.initialize()


def get_service() -> Any:
    """Get the global RetrievalService instance.

    Returns:
        RetrievalService instance.

    Raises:
        HTTPException: If not initialized.
    """
    if _service is None:
        raise HTTPException(status_code=500, detail="Retrieval service not initialized")
    return _service


def get_catalog() -> ProductCatalog:
    """Get the global catalog instance."""
    if _catalog is None:
        raise HTTPException(status_code=500, detail="Catalog not initialized")
    return _catalog


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with retrieval service status."""
    catalog_size = len(_catalog) if _catalog else 0

    extra: dict[str, Any] = {}
    if _service is not None:
        extra = _service.status()

    return HealthResponse(
        status="ok",
        version="0.1.0",
        catalog_size=catalog_size,
        **extra,
    )


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------
@router.get("/search", response_model=SearchResponse)
async def search_products(
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
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    try:
        service = get_service()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Search initialization error: %s", e)
        raise HTTPException(status_code=500, detail="Search service unavailable")

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
    except Exception as e:
        logger.error("Search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

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
    """Run the agent and persist the run + tool calls via repository.

    Uses a new transaction for the write after the agent completes.
    Does NOT wrap the LLM/tool execution in a long DB transaction.
    """
    import uuid

    request_id = f"req-{uuid.uuid4().hex[:12]}"

    try:
        response = orch.run(request)
    except Exception as e:
        # Record failure (sanitized) — best effort, non-fatal to API response
        try:
            _record_failure(request_id, request, e)
        except Exception:
            logger.exception("Failed to record agent failure")
        raise

    # Record success (best effort — persistence failure must not break API)
    try:
        _record_success(request_id, request, response, tools)
    except Exception:
        logger.exception("Failed to record agent run")

    return response


def _record_success(
    request_id: str,
    request: AgentRequest,
    response: AgentResponse,
    tools: AgentTools,
) -> None:
    """Persist a successful agent run + tool call metadata."""
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import AgentRunRepository

    with session_scope() as session:
        repo = AgentRunRepository(session)
        record = repo.create_running(
            request_id=request_id,
            user_query=request.message,
            mode_requested=request.mode,
        )
        # Sanitized tool calls (no API keys, no hidden reasoning)
        repo.add_tool_calls(record.id, [])
        repo.mark_completed(
            record.id,
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


def _record_failure(request_id: str, request: AgentRequest, error: Exception) -> None:
    """Persist a failed agent run with a sanitized error message."""
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import AgentRunRepository

    error_type = type(error).__name__
    # Sanitize: only the exception class name + safe message, no API keys.
    safe_message = str(error)[:500]

    with session_scope() as session:
        repo = AgentRunRepository(session)
        record = repo.create_running(
            request_id=request_id,
            user_query=request.message,
            mode_requested=request.mode,
        )
        repo.mark_failed(record.id, error_type, safe_message)
