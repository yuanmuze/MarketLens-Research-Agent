"""Pydantic models for Phase 5 agent request, tools, and response."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Agent Request
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    """Incoming request from API or test harness."""

    message: str = Field(..., min_length=1, description="User's natural language request")
    mode: Literal["fast", "balanced", "quality"] = Field(
        default="balanced", description="Retrieval mode"
    )
    max_results: int = Field(default=5, ge=1, le=20, description="Max products to recommend")


# ---------------------------------------------------------------------------
# Tool Parameters (typed, validated by Pydantic)
# ---------------------------------------------------------------------------

class SearchCatalogParams(BaseModel):
    """Parameters for the search_catalog tool."""

    query: str = Field(..., min_length=1, description="Search query string")
    mode: Literal["fast", "balanced", "quality"] = Field(
        default="balanced",
        description="fast=BM25, balanced=Hybrid, quality=Rerank",
    )
    top_k: int = Field(default=10, ge=1, le=20)
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    brands: list[str] | None = Field(default=None)
    min_rating: float | None = Field(default=None, ge=0, le=5)

    model_config = {"extra": "forbid"}


class GetProductDetailsParams(BaseModel):
    """Parameters for the get_product_details tool."""

    product_ids: list[str] = Field(..., min_length=1, max_length=10)

    model_config = {"extra": "forbid"}


class CompareProductsParams(BaseModel):
    """Parameters for the compare_products tool."""

    product_ids: list[str] = Field(..., min_length=2, max_length=5)
    fields: list[str] = Field(
        default=["title", "price", "rating", "brand", "review_count"],
        max_length=10,
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Tool Results (structured, typed)
# ---------------------------------------------------------------------------

class SearchResultItem(BaseModel):
    """A single search result from any strategy."""

    rank: int
    product_id: str
    title: str
    brand: str | None
    price: float | None
    rating: float | None
    review_count: int | None
    score: float


class SearchCatalogResult(BaseModel):
    """Result of search_catalog tool call."""

    query: str
    mode_used: str  # "bm25" | "hybrid" | "rerank"
    total_found: int
    results: list[SearchResultItem]


class ProductDetail(BaseModel):
    """Full detail for a single product."""

    product_id: str
    title: str
    brand: str | None
    price: float | None
    rating: float | None
    review_count: int | None
    description: str | None
    attributes: dict[str, str] = Field(default_factory=dict)
    url: str | None


class GetProductDetailsResult(BaseModel):
    """Result of get_product_details tool call."""

    products: list[ProductDetail]


class CompareProductsResult(BaseModel):
    """Result of compare_products tool call."""

    products: list[ProductDetail]
    comparison: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class EvidenceRef(BaseModel):
    """Traceable evidence linking a claim to a product field."""

    product_id: str
    field: str
    observed_value: Any


# ---------------------------------------------------------------------------
# Agent Final Response
# ---------------------------------------------------------------------------

class RecommendationItem(BaseModel):
    """A single product recommendation with evidence."""

    product_id: str
    title: str
    brand: str | None
    price: float | None
    rating: float | None
    review_count: int | None
    reason: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    constraint_checks: dict[str, bool] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Final response from the agent API."""

    request_id: str
    status: Literal["completed", "needs_clarification", "no_results", "degraded", "failed"]
    answer: str
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    comparison: list[dict[str, Any]] | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    mode_requested: str = "balanced"
    mode_used: str = "balanced"
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    latency_ms: float = 0.0
    error: str | None = None
