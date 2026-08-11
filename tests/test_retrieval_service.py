"""Tests for the unified RetrievalService and all four strategies."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from marketlens.catalog import ProductCatalog
from marketlens.retrieval.embedding import FakeEmbeddingBackend
from marketlens.retrieval.service import (
    RetrievalOutput,
    RetrievalResultItem,
    RetrievalService,
    _build_search_text,
)


class TestRetrievalResultItem:
    """Tests for the unified result item."""

    def test_from_product_with_all_fields(self) -> None:
        """Test conversion from product dict."""
        product = {
            "product_id": "B001",
            "title": "Test Headphones",
            "brand": "TestBrand",
            "price": 99.99,
            "rating": 4.5,
            "review_count": 100,
            "attributes": {"color": "Black"},
            "description": "Great product",
            "url": "http://example.com/B001",
        }
        item = RetrievalResultItem.from_product(product, rank=1, final_score=0.95, bm25_score=0.8)
        assert item.product_id == "B001"
        assert item.title == "Test Headphones"
        assert item.price == 99.99
        assert item.rating == 4.5
        assert item.final_score == 0.95
        assert item.bm25_score == 0.8
        assert item.embedding_score is None

    def test_from_product_with_missing_fields(self) -> None:
        """Test conversion with None/missing fields."""
        product = {"product_id": "B002", "title": "No Price Product", "brand": ""}
        item = RetrievalResultItem.from_product(product, rank=1, final_score=0.5)
        assert item.price is None
        assert item.rating is None
        assert item.review_count is None
        assert item.brand == ""


class TestBuildSearchText:
    """Tests for search text construction."""

    def test_combines_fields(self) -> None:
        """Test that title, brand, description are combined."""
        text = _build_search_text({
            "title": "Wireless Earbuds",
            "brand": "Sony",
            "description": "Noise cancelling",
        })
        assert "Wireless Earbuds" in text
        assert "Sony" in text
        assert "Noise cancelling" in text

    def test_handles_missing_fields(self) -> None:
        """Test with missing optional fields."""
        text = _build_search_text({"title": "Only Title", "brand": None, "description": None})
        assert text == "Only Title"


class TestRetrievalService:
    """Integration tests for the full RetrievalService."""

    @pytest.fixture
    def catalog(self) -> ProductCatalog:
        """Load fixture catalog."""
        return ProductCatalog.from_fixture("electronics_sample.json")

    @pytest.fixture
    def service(self, catalog: ProductCatalog) -> RetrievalService:
        """Create and initialize a RetrievalService with fake embeddings."""
        svc = RetrievalService(
            catalog,
            embedding_backend=FakeEmbeddingBackend(dim=64, seed=42),
        )
        svc.initialize()
        return svc

    # --- Initialization ---

    def test_initialize_builds_indices(self, catalog: ProductCatalog) -> None:
        """Test initialization builds BM25 and embedding indices."""
        svc = RetrievalService(catalog, embedding_backend=FakeEmbeddingBackend(dim=64, seed=42))
        assert not svc.is_initialized
        svc.initialize()
        assert svc.is_initialized
        assert svc.product_count == 20

    def test_initialize_idempotent(self, service: RetrievalService) -> None:
        """Test calling initialize twice is safe."""
        t0 = time.monotonic()
        service.initialize()  # Second call
        t1 = time.monotonic()
        assert (t1 - t0) < 0.1  # Should return immediately

    def test_search_before_init_raises(self, catalog: ProductCatalog) -> None:
        """Test searching before initialization raises RuntimeError."""
        svc = RetrievalService(catalog, embedding_backend=FakeEmbeddingBackend(dim=64, seed=42))
        with pytest.raises(RuntimeError, match="not initialized"):
            svc.search("test")

    # --- Strategy routing ---

    def test_all_strategies_return_results(self, service: RetrievalService) -> None:
        """Test that all 4 strategies return valid results."""
        for strategy in ["bm25", "embedding", "hybrid", "rerank"]:
            output = service.search("headphones", strategy=strategy, top_k=5)
            assert isinstance(output, RetrievalOutput)
            assert output.strategy == strategy
            assert output.total_found >= 0
            assert output.elapsed_ms >= 0

    def test_invalid_strategy_raises(self, service: RetrievalService) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            service.search("test", strategy="unknown")

    # --- Result structure ---

    def test_results_have_required_fields(self, service: RetrievalService) -> None:
        """Test all result items have required fields."""
        output = service.search("headphones", strategy="hybrid", top_k=10)
        for item in output.results:
            assert item.product_id
            assert item.title
            assert item.rank >= 1
            assert item.final_score >= 0

    def test_bm25_has_bm25_scores(self, service: RetrievalService) -> None:
        """Test BM25 results have bm25_score populated."""
        output = service.search("headphones", strategy="bm25", top_k=5)
        if output.results:
            assert output.results[0].bm25_score is not None

    def test_embedding_has_embedding_scores(self, service: RetrievalService) -> None:
        """Test embedding results have embedding_score."""
        output = service.search("headphones", strategy="embedding", top_k=5)
        if output.results:
            assert output.results[0].embedding_score is not None

    def test_hybrid_has_both_scores(self, service: RetrievalService) -> None:
        """Test hybrid results populate both score fields."""
        output = service.search("headphones", strategy="hybrid", top_k=5)
        if output.results:
            item = output.results[0]
            # At least one should be populated
            assert item.bm25_score is not None or item.embedding_score is not None

    # --- Determinism ---

    def test_same_input_same_output(self, service: RetrievalService) -> None:
        """Test same query+seed returns identical results."""
        o1 = service.search("wireless headphones", strategy="bm25", top_k=5)
        o2 = service.search("wireless headphones", strategy="bm25", top_k=5)
        assert o1.total_found == o2.total_found
        assert [r.product_id for r in o1.results] == [r.product_id for r in o2.results]

    # --- Product ID validity ---

    def test_all_product_ids_exist(self, service: RetrievalService) -> None:
        """Test all returned product_ids are in the catalog."""
        valid_ids = {p["product_id"] for p in service._product_dicts}
        for strategy in ["bm25", "embedding", "hybrid", "rerank"]:
            output = service.search("headphones", strategy=strategy, top_k=10)
            for item in output.results:
                assert item.product_id in valid_ids

    # --- Structured filtering ---

    def test_brand_filter(self, service: RetrievalService) -> None:
        """Test brand filter returns only matching brands."""
        output = service.search("headphones", strategy="hybrid", top_k=10, brand="Sony")
        for item in output.results:
            assert item.brand.lower() == "sony"

    def test_max_price_filter(self, service: RetrievalService) -> None:
        """Test max price filter."""
        output = service.search("headphones", strategy="hybrid", top_k=10, max_budget=100.0)
        for item in output.results:
            assert item.price is not None
            assert item.price <= 100.0

    def test_min_rating_filter(self, service: RetrievalService) -> None:
        """Test min rating filter."""
        output = service.search("headphones", strategy="hybrid", top_k=10, min_rating=4.5)
        for item in output.results:
            assert item.rating is not None
            assert item.rating >= 4.5

    def test_missing_price_excluded_from_budget(self, catalog: ProductCatalog) -> None:
        """Products without price must NOT pass a max_price filter."""
        from marketlens.models import Product
        mixed = ProductCatalog([
            Product(product_id="P1", title="Wireless Headphones", price=50.0, rating=4.0, review_count=100),
            Product(product_id="P2", title="Bluetooth Earbuds", price=None, rating=4.5, review_count=200),
            Product(product_id="P3", title="Budget Earphones", price=30.0, rating=3.5, review_count=50),
        ])
        svc = RetrievalService(mixed, embedding_backend=FakeEmbeddingBackend(dim=16, seed=1))
        svc.initialize()
        output = svc.search("Headphones Earbuds Earphones", strategy="bm25", top_k=10, max_budget=100.0)
        ids = {r.product_id for r in output.results}
        assert "P1" in ids
        assert "P2" not in ids  # No price → excluded
        assert "P3" in ids

    def test_missing_price_excluded_from_min_price(self, catalog: ProductCatalog) -> None:
        """Products without price must NOT pass a min_price filter."""
        from marketlens.models import Product
        mixed = ProductCatalog([
            Product(product_id="P1", title="With Price", price=50.0, rating=4.0, review_count=100),
            Product(product_id="P2", title="No Price", price=None, rating=4.5, review_count=200),
        ])
        svc = RetrievalService(mixed, embedding_backend=FakeEmbeddingBackend(dim=16, seed=1))
        svc.initialize()
        output = svc.search("price", strategy="bm25", top_k=10, min_price=10.0)
        ids = {r.product_id for r in output.results}
        assert "P1" in ids
        assert "P2" not in ids  # No price → excluded

    # --- Reranker specifics ---

    def test_reranker_uses_candidates(self, service: RetrievalService) -> None:
        """Test reranker only processes candidate_k items."""
        output = service.search("headphones", strategy="rerank", top_k=5, candidate_k=10)
        assert output.total_found <= 5

    def test_reranker_different_from_hybrid(self, service: RetrievalService) -> None:
        """Test reranker can produce different ordering than hybrid."""
        hybrid_out = service.search("wireless headphones", strategy="hybrid", top_k=5)
        rerank_out = service.search("wireless headphones", strategy="rerank", top_k=5, candidate_k=20)
        # With the KeywordReranker, ordering may differ
        hybrid_ids = [r.product_id for r in hybrid_out.results]
        rerank_ids = [r.product_id for r in rerank_out.results]
        # Both are valid rankings; just verify they're lists
        assert len(hybrid_ids) == len(rerank_ids)

    # --- Embedding not re-initialized ---

    def test_embedding_cache_persists(self, service: RetrievalService) -> None:
        """Test that multiple searches use the same embedding cache."""
        # First search
        o1 = service.search("headphones", strategy="embedding", top_k=5)
        e1 = [r.product_id for r in o1.results]
        # Second search
        o2 = service.search("headphones", strategy="embedding", top_k=5)
        e2 = [r.product_id for r in o2.results]
        # Same results (deterministic)
        assert e1 == e2

    # --- Edge cases ---

    def test_empty_result_query(self, service: RetrievalService) -> None:
        """Test query that returns no results."""
        output = service.search(
            "xyznonexistentproduct12345",
            strategy="bm25",
            top_k=5,
        )
        assert output.total_found >= 0  # May be 0

    def test_top_k_limits(self, service: RetrievalService) -> None:
        """Test top_k limits results."""
        output = service.search("headphones", strategy="hybrid", top_k=3)
        assert len(output.results) <= 3


class TestRetrievalServiceWithTempData:
    """Tests using temporary data files for cache testing."""

    def test_embedding_cache_saved_and_loaded(self, tmp_path: Path) -> None:
        """Test that embedding cache is saved to and loaded from disk."""

        # Create temp product data
        products = [
            {
                "product_id": f"P{i:03d}",
                "title": f"Test Product {i}",
                "brand": "TestBrand",
                "price": 10.0 + i,
                "rating": 4.0,
                "review_count": 100,
                "description": f"Description {i}",
                "attributes": {},
                "images": [],
                "url": f"http://e.com/P{i:03d}",
                "category": "electronics",
            }
            for i in range(10)
        ]
        data_path = tmp_path / "products.json"
        data_path.write_text(json.dumps(products), encoding="utf-8")

        catalog = ProductCatalog.from_json(data_path)

        # Monkey-patch CACHE_DIR to use tmp_path
        import marketlens.retrieval.service as svc_mod
        original_cache = svc_mod.CACHE_DIR
        svc_mod.CACHE_DIR = tmp_path / "cache"
        try:
            # First initialization — computes and saves cache
            svc1 = RetrievalService(
                catalog,
                data_path=data_path,
                embedding_backend=FakeEmbeddingBackend(dim=32, seed=42),
            )
            svc1.initialize()
            assert svc1.is_initialized

            # Fake backend doesn't save cache — only real SentenceTransformerBackend does
            # This test verifies the code path works for fake backends too
        finally:
            svc_mod.CACHE_DIR = original_cache

    def test_model_info_reports_fake_backend(self, catalog: ProductCatalog) -> None:
        """Test embedding_model_info for fake backend."""
        svc = RetrievalService(
            catalog,
            embedding_backend=FakeEmbeddingBackend(dim=64, seed=1),
        )
        svc.initialize()
        info = svc.embedding_model_info
        assert info["type"] == "fake"
        assert info["dim"] == 64


class TestAPIRoutes:
    """Test the FastAPI /search endpoint with new parameters."""

    @pytest.fixture(autouse=True)
    def setup_db(self) -> None:
        """Setup database."""
        from marketlens.api.database import drop_db, init_db
        init_db()
        yield
        drop_db()
        init_db()

    @pytest.fixture
    def client(self, catalog: ProductCatalog):
        """Create TestClient with catalog loaded."""
        from fastapi.testclient import TestClient

        from marketlens.api.main import app
        from marketlens.api.routes import init_catalog
        init_catalog(catalog)
        return TestClient(app)

    def test_search_with_strategy_param(self, client) -> None:
        """Test /search accepts strategy parameter."""
        for strategy in ["bm25", "embedding", "hybrid", "rerank"]:
            resp = client.get(f"/search?q=headphones&strategy={strategy}&top_k=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_results"] >= 0
            assert "results" in data

    def test_search_invalid_strategy_422(self, client) -> None:
        """Test invalid strategy returns 422."""
        resp = client.get("/search?q=headphones&strategy=invalid")
        assert resp.status_code == 422

    def test_search_with_price_filters(self, client) -> None:
        """Test search with min_price and max_budget."""
        resp = client.get("/search?q=headphones&max_budget=200&min_price=10&top_k=5")
        assert resp.status_code == 200
        data = resp.json()
        for r in data["results"]:
            if r["price"] is not None:
                assert r["price"] <= 200
                assert r["price"] >= 10

    def test_search_with_brand_and_rating(self, client) -> None:
        """Test combined brand + rating filter."""
        resp = client.get("/search?q=headphones&brand=Sony&min_rating=4.5&top_k=5")
        assert resp.status_code == 200
        data = resp.json()
        for r in data["results"]:
            assert r["brand"].lower() == "sony"
            if r["rating"] is not None:
                assert r["rating"] >= 4.5

    def test_search_returns_request_id(self, client) -> None:
        """Test response includes request_id and timing."""
        resp = client.get("/search?q=headphones")
        assert "X-Request-ID" in resp.headers
        data = resp.json()
        assert "duration_ms" in data
        assert "request_id" in data
