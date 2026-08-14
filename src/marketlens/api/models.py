"""API request/response models for MarketLens FastAPI."""

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """GET /health response — includes retrieval service status."""

    model_config = {"extra": "allow"}
    status: str = "ok"
    version: str = "0.1.1"
    catalog_size: int = 0


class SearchRequest(BaseModel):
    """GET /search query parameters."""

    q: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=20, ge=1, le=100, description="Max results")
    max_budget: float | None = Field(default=None, ge=0, description="Max price filter")
    min_rating: float | None = Field(default=None, ge=0, le=5, description="Min rating filter")
    brand: str | None = Field(default=None, description="Brand filter")
    use_reranker: bool = Field(default=False, description="Enable reranker")


class SearchResultItem(BaseModel):
    """A single search result in the API response."""

    rank: int
    product_id: str
    title: str
    brand: str | None
    price: float | None
    rating: float | None
    review_count: int | None
    score: float
    source: str


class SearchResponse(BaseModel):
    """GET /search response."""

    request_id: str
    query: str
    results: list[SearchResultItem]
    total_results: int
    duration_ms: float


class ResearchRequest(BaseModel):
    """POST /research request body."""

    query: str = Field(..., min_length=1, description="Natural language research query")
    max_results: int = Field(default=10, ge=1, le=50)
    enable_web_search: bool = Field(default=False)
    max_budget: float | None = Field(default=None, ge=0)
    min_rating: float | None = Field(default=None, ge=0, le=5)
    preferred_brands: list[str] = Field(default_factory=list)
    excluded_brands: list[str] = Field(default_factory=list)


class ProductRef(BaseModel):
    """Lightweight product reference in API responses."""

    product_id: str
    title: str
    brand: str | None
    price: float | None
    rating: float | None


class ComparisonRef(BaseModel):
    """Comparison item in API response."""

    product: ProductRef
    pros: list[str]
    cons: list[str]
    recommendation_score: float | None


class EvidenceRef(BaseModel):
    """Evidence reference in API response."""

    product_id: str
    source_type: str
    source_detail: str
    relevance_score: float
    evidence_text: str


class ResearchReportResponse(BaseModel):
    """GET /research/jobs/{job_id}/report response."""

    job_id: str
    query: str
    summary: str
    recommendations: list[ProductRef]
    comparisons: list[ComparisonRef]
    evidence: list[EvidenceRef]
    constraints_satisfied: bool
    generated_at: datetime
    report_text: str


class ResearchJobResponse(BaseModel):
    """GET /research/jobs/{job_id} response."""

    job_id: str
    request_id: str
    status: str
    query: str
    created_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: float | None
    product_count: int | None
    tool_calls: int
    error_message: str | None


class ResearchSubmitResponse(BaseModel):
    """POST /research or POST /research/jobs response."""

    job_id: str
    request_id: str
    status: str
    query: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None
