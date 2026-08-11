"""Tests for the MarketLens FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from marketlens.api.database import drop_db, init_db
from marketlens.api.main import app
from marketlens.api.routes import init_catalog
from marketlens.catalog import ProductCatalog


@pytest.fixture(autouse=True)
def setup_database() -> None:
    """Set up fresh database for each test."""
    init_db()
    yield
    # Clean up
    drop_db()
    init_db()


@pytest.fixture
def client(catalog: ProductCatalog) -> TestClient:
    """Create TestClient with catalog loaded."""
    init_catalog(catalog)
    return TestClient(app)


@pytest.fixture
def empty_client() -> TestClient:
    """Create TestClient with empty catalog."""
    init_catalog(ProductCatalog())
    return TestClient(app)


class TestHealthEndpoint:
    """GET /health tests."""

    def test_health_ok(self, client: TestClient) -> None:
        """Test health check returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_has_catalog_size(self, client: TestClient) -> None:
        """Test health response includes catalog size."""
        response = client.get("/health")
        data = response.json()
        assert data["catalog_size"] > 0

    def test_health_empty_catalog(self, empty_client: TestClient) -> None:
        """Test health with empty catalog."""
        response = empty_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["catalog_size"] == 0


class TestSearchEndpoint:
    """GET /search tests."""

    def test_basic_search(self, client: TestClient) -> None:
        """Test basic product search."""
        response = client.get("/search?q=headphones")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "headphones"
        assert "results" in data
        assert data["total_results"] > 0
        assert "request_id" in data

    def test_search_with_top_k(self, client: TestClient) -> None:
        """Test search with top_k parameter."""
        response = client.get("/search?q=headphones&top_k=3")
        data = response.json()
        assert data["total_results"] <= 3

    def test_search_with_budget(self, client: TestClient) -> None:
        """Test search with budget filter."""
        response = client.get("/search?q=headphones&max_budget=100")
        data = response.json()
        for result in data["results"]:
            assert result["price"] is not None
            assert result["price"] <= 100.0

    def test_search_with_rating(self, client: TestClient) -> None:
        """Test search with rating filter."""
        response = client.get("/search?q=headphones&min_rating=4.8")
        data = response.json()
        for result in data["results"]:
            assert result["rating"] is not None
            assert result["rating"] >= 4.8

    def test_search_with_brand(self, client: TestClient) -> None:
        """Test search with brand filter."""
        response = client.get("/search?q=headphones&brand=AudioBrand")
        data = response.json()
        for result in data["results"]:
            assert result["brand"] == "AudioBrand"

    def test_search_empty_query_400(self, client: TestClient) -> None:
        """Test empty query returns 422."""
        response = client.get("/search?q=")
        assert response.status_code == 422

    def test_search_invalid_top_k(self, client: TestClient) -> None:
        """Test invalid top_k returns 422."""
        response = client.get("/search?q=test&top_k=0")
        assert response.status_code == 422
        response = client.get("/search?q=test&top_k=101")
        assert response.status_code == 422

    def test_search_results_have_scores(self, client: TestClient) -> None:
        """Test results include scores."""
        response = client.get("/search?q=wireless")
        data = response.json()
        for result in data["results"]:
            assert "score" in result
            assert result["score"] >= 0
            assert "source" in result

    def test_search_has_request_id_header(self, client: TestClient) -> None:
        """Test response has X-Request-ID header."""
        response = client.get("/search?q=headphones")
        assert "X-Request-ID" in response.headers


class TestResearchEndpoint:
    """POST /research tests."""

    def test_submit_research(self, client: TestClient) -> None:
        """Test submitting a research request."""
        response = client.post("/research", json={
            "query": "best wireless headphones under $300",
            "max_results": 5,
        })
        assert response.status_code == 201
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "completed"
        assert data["query"] == "best wireless headphones under $300"

    def test_submit_research_empty_query(self, client: TestClient) -> None:
        """Test empty query returns 422."""
        response = client.post("/research", json={"query": ""})
        assert response.status_code == 422

    def test_research_with_budget(self, client: TestClient) -> None:
        """Test research with budget constraint."""
        response = client.post("/research", json={
            "query": "noise cancelling headphones",
            "max_budget": 200.0,
        })
        assert response.status_code == 201

    def test_research_with_brands(self, client: TestClient) -> None:
        """Test research with brand preferences."""
        response = client.post("/research", json={
            "query": "best earbuds",
            "preferred_brands": ["Sony"],
        })
        assert response.status_code == 201


class TestJobEndpoints:
    """POST /research/jobs and GET /research/jobs/{job_id} tests."""

    def test_create_job(self, client: TestClient) -> None:
        """Test creating a research job."""
        response = client.post("/research/jobs", json={"query": "test query"})
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"
        job_id = data["job_id"]

        # Check job status
        status_response = client.get(f"/research/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] == "pending"
        assert status_data["query"] == "test query"

    def test_get_nonexistent_job_404(self, client: TestClient) -> None:
        """Test getting a nonexistent job returns 404."""
        response = client.get("/research/jobs/nonexistent-123")
        assert response.status_code == 404

    def test_get_report_pending_400(self, client: TestClient) -> None:
        """Test getting report for pending job returns 400."""
        job_resp = client.post("/research/jobs", json={"query": "test"})
        job_id = job_resp.json()["job_id"]

        report_resp = client.get(f"/research/jobs/{job_id}/report")
        assert report_resp.status_code == 400

    def test_get_report_completed(self, client: TestClient) -> None:
        """Test getting report for a completed job."""
        # Submit research directly (synchronous, completes immediately)
        resp = client.post("/research", json={"query": "wireless headphones"})
        job_id = resp.json()["job_id"]

        report_resp = client.get(f"/research/jobs/{job_id}/report")
        assert report_resp.status_code == 200
        data = report_resp.json()
        assert data["job_id"] == job_id
        assert data["report_text"] is not None


class TestErrorHandling:
    """Error handling tests."""

    def test_404_not_found(self, client: TestClient) -> None:
        """Test 404 for unknown endpoint."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404

    def test_error_does_not_leak_stack(self, client: TestClient) -> None:
        """Test error responses don't leak stack traces."""
        # This is tested indirectly via the global error handler
        response = client.get("/research/jobs/nonexistent/report")
        assert response.status_code == 404
        data = response.json()
        assert "stack" not in str(data).lower()
        assert "traceback" not in str(data).lower()

    def test_request_id_in_error(self, client: TestClient) -> None:
        """Test error responses include request_id."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
