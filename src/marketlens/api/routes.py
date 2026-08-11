"""FastAPI route handlers for MarketLens Research API."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

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
from marketlens.models import SearchQuery, UserConstraints
from marketlens.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

router = APIRouter()

# Global catalog instance (initialized at app startup)
_catalog: ProductCatalog | None = None
_retriever: HybridRetriever | None = None


def init_catalog(catalog: ProductCatalog) -> None:
    """Initialize the global catalog and retriever.

    Args:
        catalog: ProductCatalog instance.
    """
    global _catalog, _retriever
    _catalog = catalog
    if len(catalog) > 0:
        _retriever = HybridRetriever(catalog).fit()


def get_catalog() -> ProductCatalog:
    """Get the global catalog instance.

    Returns:
        ProductCatalog instance.

    Raises:
        HTTPException: If catalog not initialized.
    """
    if _catalog is None:
        raise HTTPException(status_code=500, detail="Catalog not initialized")
    return _catalog


def get_retriever() -> HybridRetriever:
    """Get the global retriever instance.

    Returns:
        HybridRetriever instance.

    Raises:
        HTTPException: If retriever not initialized.
    """
    if _retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    return _retriever


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    catalog = _catalog
    return HealthResponse(
        status="ok",
        version="0.1.0",
        catalog_size=len(catalog) if catalog else 0,
    )


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------
@router.get("/search", response_model=SearchResponse)
async def search_products(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=20, ge=1, le=100, description="Max results"),
    max_budget: float | None = Query(default=None, ge=0, description="Max price"),
    min_rating: float | None = Query(default=None, ge=0, le=5, description="Min rating"),
    brand: str | None = Query(default=None, description="Brand filter"),
    use_reranker: bool = Query(default=False, description="Enable reranker"),
) -> SearchResponse:
    """Search the product catalog.

    Returns ranked product results with scores and evidence metadata.
    """
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    t0 = time.monotonic()

    try:
        _ = get_catalog()  # Verify catalog is initialized
        retriever = get_retriever()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Search initialization error: %s", e)
        raise HTTPException(status_code=500, detail="Search service unavailable")

    # Build constraints
    filters = UserConstraints()
    if max_budget is not None:
        filters.max_budget = max_budget
    if min_rating is not None:
        filters.min_rating = min_rating
    if brand:
        filters.preferred_brands = [brand]

    query_obj = SearchQuery(
        text=q,
        top_k=top_k,
        filters=filters,
        use_reranker=use_reranker,
    )

    try:
        results = retriever.search(query_obj)
    except Exception as e:
        logger.error("Search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    elapsed_ms = (time.monotonic() - t0) * 1000

    # Persist search query
    try:
        session = get_session()
        record = SearchQueryRecord(
            query=q,
            top_k=top_k,
            result_count=len(results),
            duration_ms=elapsed_ms,
            source="hybrid",
        )
        session.add(record)
        session.commit()
    except Exception as e:
        logger.warning("Failed to persist search query: %s", e)

    items = [
        SearchResultItem(
            rank=r.rank,
            product_id=r.product.product_id,
            title=r.product.title,
            brand=r.product.brand,
            price=r.product.price,
            rating=r.product.rating,
            review_count=r.product.review_count,
            score=round(r.score, 4),
            source=r.source,
        )
        for r in results
    ]

    return SearchResponse(
        request_id=request_id,
        query=q,
        results=items,
        total_results=len(items),
        duration_ms=round(elapsed_ms, 2),
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

        # Update job record
        record.status = "completed"
        record.completed_at = datetime.now(timezone.utc)
        record.duration_ms = elapsed_ms
        record.report_text = result.get("final_report", "")
        record.product_count = len(result.get("products", []))
        record.tool_calls = result.get("tool_calls", 0)
        record.evidence_count = len(result.get("evidence", []))
        record.constraints_satisfied = 1 if result.get("constraints_satisfied") else 0
        session.commit()

    except Exception as e:
        logger.error("Research execution error: %s", e, exc_info=True)
        record.status = "failed"
        record.error_message = str(e)
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

    return ResearchJobResponse(
        job_id=record.job_id,
        request_id=record.request_id,
        status=record.status,
        query=record.query,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration_ms=record.duration_ms,
        product_count=record.product_count,
        tool_calls=record.tool_calls or 0,
        error_message=record.error_message,
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

    return ResearchReportResponse(
        job_id=record.job_id,
        query=record.query,
        summary=f"Research completed in {record.duration_ms:.0f}ms. Found {record.product_count or 0} products.",
        recommendations=[],
        comparisons=[],
        evidence=[],
        constraints_satisfied=bool(record.constraints_satisfied),
        generated_at=record.completed_at or datetime.now(timezone.utc),
        report_text=record.report_text or "",
    )
