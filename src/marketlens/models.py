"""Pydantic v2 domain models for MarketLens Research Agent."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProductCategory(str, Enum):
    """Product category enumeration for AI/electronics domain."""

    AI_APPLICATION = "ai_application"
    AI_BACKEND = "ai_backend"
    RAG_SEARCH = "rag_search"
    AGENT_ENGINEER = "agent_engineer"
    ELECTRONICS = "electronics"
    SOFTWARE = "software"
    OTHER = "other"


class Product(BaseModel):
    """A product in the MarketLens catalog.

    Represents a single product from the Amazon Electronics reviews dataset
    or any structured product catalog. All fields are validated on creation.
    """

    product_id: str = Field(..., description="Unique product identifier (e.g., ASIN)")
    title: str = Field(..., description="Product title/name")
    brand: str | None = Field(default=None, description="Brand or manufacturer name")
    category: ProductCategory = Field(
        default=ProductCategory.ELECTRONICS,
        description="Product category",
    )
    price: float | None = Field(
        default=None,
        ge=0,
        description="Product price in USD",
    )
    rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Average rating (0-5)",
    )
    review_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of reviews",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible key-value attributes (e.g., RAM, storage, color)",
    )
    description: str | None = Field(
        default=None,
        description="Product description or feature text",
    )
    images: list[str] = Field(
        default_factory=list,
        description="List of image URLs",
    )
    url: str | None = Field(default=None, description="Product page URL")

    @field_validator("product_id")
    @classmethod
    def product_id_must_be_non_empty(cls, v: str) -> str:
        """Validate that product_id is not empty."""
        if not v.strip():
            raise ValueError("product_id must not be empty")
        return v.strip()

    def to_search_text(self) -> str:
        """Combine key fields into a searchable text representation."""
        parts = [self.title]
        if self.brand:
            parts.append(self.brand)
        if self.description:
            parts.append(self.description)
        if self.attributes:
            parts.append(" ".join(str(v) for v in self.attributes.values()))
        return " ".join(parts)


class ProductEvidence(BaseModel):
    """Evidence linking a recommendation to a specific product and source.

    Every recommendation or comparison must be traceable to one or more
    ProductEvidence instances, ensuring grounded outputs.
    """

    product_id: str = Field(..., description="ID of the product this evidence supports")
    source_type: str = Field(
        default="catalog",
        description="Source type: catalog, web_search, review, etc.",
    )
    source_detail: str = Field(
        default="",
        description="Human-readable source reference (e.g., URL, page number)",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Relevance score (0-1) assigned by retrieval",
    )
    evidence_text: str = Field(
        default="",
        description="The evidential text snippet",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the evidence was collected",
    )


class UserConstraints(BaseModel):
    """Deterministic constraints from the user request.

    These are extracted and validated by the agent, then enforced
    by plain Python (not by the LLM).
    """

    max_budget: float | None = Field(
        default=None,
        ge=0,
        description="Maximum budget in USD",
    )
    min_budget: float | None = Field(
        default=None,
        ge=0,
        description="Minimum budget in USD",
    )
    preferred_brands: list[str] = Field(
        default_factory=list,
        description="List of preferred brands",
    )
    excluded_brands: list[str] = Field(
        default_factory=list,
        description="List of brands to exclude",
    )
    categories: list[ProductCategory] = Field(
        default_factory=list,
        description="Required product categories",
    )
    min_rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Minimum acceptable rating",
    )
    min_review_count: int | None = Field(
        default=None,
        ge=0,
        description="Minimum number of reviews",
    )
    required_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Required attribute values (e.g., {'RAM': '16GB'})",
    )
    excluded_product_ids: list[str] = Field(
        default_factory=list,
        description="Product IDs to exclude",
    )

    @field_validator("min_budget")
    @classmethod
    def check_budget_range(cls, v: float | None, info) -> float | None:
        """Validate budget range consistency."""
        if v is not None and info.data.get("max_budget") is not None:
            if v > info.data["max_budget"]:
                raise ValueError("min_budget must not exceed max_budget")
        return v


class SearchQuery(BaseModel):
    """A structured search query for product retrieval."""

    text: str = Field(..., description="Free-text query string")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of results to return")
    filters: UserConstraints | None = Field(
        default=None,
        description="Hard filters to apply",
    )
    use_bm25: bool = Field(default=True, description="Include BM25 retrieval")
    use_embedding: bool = Field(default=True, description="Include embedding retrieval")
    use_reranker: bool = Field(default=False, description="Apply reranker after fusion")
    reranker_model: str | None = Field(
        default=None,
        description="Optional reranker model identifier",
    )


class SearchResult(BaseModel):
    """A single search result with ranking and provenance."""

    product: Product = Field(..., description="The matched product")
    score: float = Field(..., description="Final relevance score")
    bm25_score: float | None = Field(
        default=None,
        description="BM25 component score",
    )
    embedding_score: float | None = Field(
        default=None,
        description="Embedding component score",
    )
    reranker_score: float | None = Field(
        default=None,
        description="Reranker score (if applied)",
    )
    rank: int = Field(..., ge=1, description="1-indexed rank in result list")
    source: str = Field(
        default="hybrid",
        description="Source method: bm25, embedding, hybrid, reranked",
    )


class ComparisonItem(BaseModel):
    """A product comparison entry for agent output."""

    product: Product = Field(..., description="The product being compared")
    evidence: list[ProductEvidence] = Field(
        default_factory=list,
        description="Evidence supporting this comparison",
    )
    pros: list[str] = Field(default_factory=list, description="Advantages")
    cons: list[str] = Field(default_factory=list, description="Disadvantages")
    recommendation_score: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Overall recommendation score (0-10)",
    )


class ResearchRequest(BaseModel):
    """A complete product research request from the user."""

    request_id: str = Field(
        default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}",
        description="Unique request identifier",
    )
    query: str = Field(..., description="Natural language research query")
    constraints: UserConstraints = Field(
        default_factory=UserConstraints,
        description="Extracted hard constraints",
    )
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results")
    enable_web_search: bool = Field(
        default=False,
        description="Whether to supplement with web search",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Request creation timestamp",
    )


class JobStatus(str, Enum):
    """Research job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchJob(BaseModel):
    """Tracks a research job through its lifecycle."""

    job_id: str = Field(..., description="Unique job identifier")
    request: ResearchRequest = Field(..., description="The original request")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Current status")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job creation timestamp",
    )
    started_at: datetime | None = Field(default=None, description="When execution started")
    completed_at: datetime | None = Field(default=None, description="When execution ended")
    duration_ms: float | None = Field(default=None, description="Execution duration in ms")
    tool_calls: int = Field(default=0, description="Number of tool calls made")
    retries: int = Field(default=0, description="Number of retries")
    error_message: str | None = Field(default=None, description="Error message if failed")


class ResearchReport(BaseModel):
    """The final research report with evidence and comparisons."""

    job_id: str = Field(..., description="Associated job identifier")
    query: str = Field(..., description="Original research query")
    summary: str = Field(..., description="Executive summary")
    comparisons: list[ComparisonItem] = Field(
        default_factory=list,
        description="Product comparisons",
    )
    recommendations: list[Product] = Field(
        default_factory=list,
        description="Top recommended products",
    )
    evidence: list[ProductEvidence] = Field(
        default_factory=list,
        description="All supporting evidence",
    )
    constraints_satisfied: bool = Field(
        default=True,
        description="Whether all hard constraints were met",
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Report generation timestamp",
    )
    raw_agent_output: str | None = Field(
        default=None,
        description="Raw agent output for debugging",
    )
