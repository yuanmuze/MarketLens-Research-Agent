"""Tests for Pydantic v2 domain models."""

import pytest

from marketlens.models import (
    JobStatus,
    Product,
    ProductCategory,
    ResearchJob,
    ResearchRequest,
    SearchQuery,
    SearchResult,
    UserConstraints,
)


class TestProduct:
    """Product model tests."""

    def test_valid_product_creation(self) -> None:
        """Test creating a valid product."""
        p = Product(
            product_id="TEST001",
            title="Test Product",
            brand="TestBrand",
            category=ProductCategory.ELECTRONICS,
            price=99.99,
            rating=4.5,
            review_count=100,
        )
        assert p.product_id == "TEST001"
        assert p.title == "Test Product"
        assert p.price == 99.99

    def test_product_id_must_not_be_empty(self) -> None:
        """Test that empty product_id raises ValueError."""
        with pytest.raises(ValueError, match="product_id must not be empty"):
            Product(product_id="", title="Test")

        with pytest.raises(ValueError, match="product_id must not be empty"):
            Product(product_id="   ", title="Test")

    def test_price_must_be_non_negative(self) -> None:
        """Test that negative price raises validation error."""
        with pytest.raises(Exception):
            Product(product_id="T1", title="Test", price=-1.0)

    def test_rating_bounds(self) -> None:
        """Test that rating must be 0-5."""
        with pytest.raises(Exception):
            Product(product_id="T1", title="Test", rating=5.5)

        with pytest.raises(Exception):
            Product(product_id="T1", title="Test", rating=-0.1)

    def test_to_search_text(self) -> None:
        """Test search text generation."""
        p = Product(
            product_id="T1",
            title="Wireless Earbuds",
            brand="AudioCo",
            description="Great sound quality",
            attributes={"color": "Black"},
        )
        text = p.to_search_text()
        assert "Wireless Earbuds" in text
        assert "AudioCo" in text
        assert "Great sound quality" in text
        assert "Black" in text

    def test_default_values(self) -> None:
        """Test default field values."""
        p = Product(product_id="T1", title="Test")
        assert p.category == ProductCategory.ELECTRONICS
        assert p.price is None
        assert p.rating is None
        assert p.review_count is None
        assert p.attributes == {}
        assert p.images == []
        assert p.url is None


class TestUserConstraints:
    """UserConstraints model tests."""

    def test_valid_constraints(self) -> None:
        """Test creating valid constraints."""
        c = UserConstraints(
            max_budget=500.0,
            min_budget=100.0,
            preferred_brands=["Sony"],
            min_rating=4.0,
        )
        assert c.max_budget == 500.0
        assert c.min_budget == 100.0
        assert c.preferred_brands == ["Sony"]
        assert c.min_rating == 4.0

    def test_budget_range_validation(self) -> None:
        """Test that min_budget must not exceed max_budget."""
        with pytest.raises(ValueError, match="min_budget must not exceed max_budget"):
            UserConstraints(max_budget=100.0, min_budget=200.0)

    def test_default_constraints(self) -> None:
        """Test default empty constraints."""
        c = UserConstraints()
        assert c.max_budget is None
        assert c.min_budget is None
        assert c.preferred_brands == []


class TestSearchQuery:
    """SearchQuery model tests."""

    def test_valid_query(self) -> None:
        """Test creating a valid search query."""
        q = SearchQuery(text="wireless headphones", top_k=10)
        assert q.text == "wireless headphones"
        assert q.top_k == 10
        assert q.use_bm25 is True
        assert q.use_embedding is True
        assert q.use_reranker is False

    def test_top_k_bounds(self) -> None:
        """Test top_k validation."""
        with pytest.raises(Exception):
            SearchQuery(text="test", top_k=0)

        with pytest.raises(Exception):
            SearchQuery(text="test", top_k=101)


class TestSearchResult:
    """SearchResult model tests."""

    def test_valid_result(self, sample_products: list[Product]) -> None:
        """Test creating a valid search result."""
        sr = SearchResult(
            product=sample_products[0],
            score=0.95,
            bm25_score=0.8,
            embedding_score=0.9,
            rank=1,
            source="hybrid",
        )
        assert sr.rank == 1
        assert sr.score == 0.95
        assert sr.source == "hybrid"


class TestResearchRequest:
    """ResearchRequest model tests."""

    def test_valid_request(self) -> None:
        """Test creating a valid research request."""
        req = ResearchRequest(query="best noise cancelling headphones under $300")
        assert "noise cancelling" in req.query
        assert req.max_results == 10
        assert req.enable_web_search is False
        assert req.request_id.startswith("req-")

    def test_auto_request_id(self) -> None:
        """Test auto-generated request ID."""
        req1 = ResearchRequest(query="test")
        req2 = ResearchRequest(query="test")
        assert req1.request_id != req2.request_id


class TestResearchJob:
    """ResearchJob model tests."""

    def test_valid_job(self) -> None:
        """Test creating a valid research job."""
        req = ResearchRequest(query="test")
        job = ResearchJob(job_id="job-001", request=req)
        assert job.job_id == "job-001"
        assert job.status == JobStatus.PENDING
        assert job.error_message is None

    def test_job_status_transitions(self) -> None:
        """Test job status transitions."""
        req = ResearchRequest(query="test")
        job = ResearchJob(job_id="job-001", request=req, status=JobStatus.RUNNING)
        assert job.status == JobStatus.RUNNING
        job.status = JobStatus.COMPLETED
        assert job.status == JobStatus.COMPLETED
