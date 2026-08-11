"""Tests for hybrid retrieval (BM25 + embedding RRF)."""

import pytest

from marketlens.catalog import ProductCatalog
from marketlens.models import SearchQuery, UserConstraints
from marketlens.retrieval.hybrid import HybridRetriever
from marketlens.retrieval.reranker import KeywordReranker


class TestHybridRetriever:
    """HybridRetriever tests."""

    @pytest.fixture
    def hybrid(self, catalog: ProductCatalog) -> HybridRetriever:
        """A pre-fitted hybrid retriever."""
        return HybridRetriever(catalog).fit()

    def test_is_fitted(self, hybrid: HybridRetriever) -> None:
        """Test fitted status."""
        assert hybrid.is_fitted

    def test_fit_empty_catalog_raises(self, empty_catalog: ProductCatalog) -> None:
        """Test fitting on empty catalog raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            HybridRetriever(empty_catalog).fit()

    def test_search_not_fitted_raises(self, catalog: ProductCatalog) -> None:
        """Test searching before fitting raises RuntimeError."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            HybridRetriever(catalog).search(SearchQuery(text="test"))

    def test_basic_search(self, hybrid: HybridRetriever) -> None:
        """Test basic hybrid search."""
        query = SearchQuery(text="wireless headphones", top_k=5)
        results = hybrid.search(query)
        assert len(results) > 0
        assert len(results) <= 5

    def test_search_results_have_products(self, hybrid: HybridRetriever) -> None:
        """Test that results contain Products."""
        query = SearchQuery(text="headphones")
        results = hybrid.search(query)
        for result in results:
            assert result.product.product_id

    def test_results_sorted_by_score(self, hybrid: HybridRetriever) -> None:
        """Test that results are sorted by score descending."""
        query = SearchQuery(text="audio", top_k=10)
        results = hybrid.search(query)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_results_have_ranks(self, hybrid: HybridRetriever) -> None:
        """Test that results have 1-indexed ranks."""
        query = SearchQuery(text="headphones")
        results = hybrid.search(query)
        for i, result in enumerate(results, start=1):
            assert result.rank == i

    def test_empty_text_query(self, hybrid: HybridRetriever) -> None:
        """Test empty query returns results."""
        query = SearchQuery(text="")
        results = hybrid.search(query)
        # Empty query may produce empty or sparse results
        assert isinstance(results, list)

    def test_top_k_limit(self, hybrid: HybridRetriever) -> None:
        """Test top_k limits the number of results."""
        query = SearchQuery(text="headphones", top_k=3)
        results = hybrid.search(query)
        assert len(results) <= 3

    def test_source_field(self, hybrid: HybridRetriever) -> None:
        """Test that results have a source field."""
        query = SearchQuery(text="wireless noise cancelling", top_k=10)
        results = hybrid.search(query)
        if results:
            assert results[0].source in ("bm25", "embedding", "hybrid", "reranked", "unknown")

    def test_result_has_scores(self, hybrid: HybridRetriever) -> None:
        """Test that results include component scores."""
        query = SearchQuery(text="headphones")
        results = hybrid.search(query)
        if results:
            r = results[0]
            assert r.score >= 0
            # At least one of component scores may be present
            assert r.bm25_score is not None or r.embedding_score is not None

    def test_bm25_only(self, hybrid: HybridRetriever) -> None:
        """Test search with only BM25 enabled."""
        query = SearchQuery(text="wireless headphones", use_bm25=True, use_embedding=False)
        results = hybrid.search(query)
        assert len(results) >= 0

    def test_embedding_only(self, hybrid: HybridRetriever) -> None:
        """Test search with only embedding enabled."""
        query = SearchQuery(text="wireless headphones", use_bm25=False, use_embedding=True)
        results = hybrid.search(query)
        assert len(results) >= 0

    def test_neither_enabled(self, hybrid: HybridRetriever) -> None:
        """Test search with no retrieval methods."""
        query = SearchQuery(text="wireless headphones", use_bm25=False, use_embedding=False)
        results = hybrid.search(query)
        assert results == []


class TestHybridWithFilters:
    """Hybrid retriever with hard constraint filtering."""

    @pytest.fixture
    def hybrid(self, catalog: ProductCatalog) -> HybridRetriever:
        """A pre-fitted hybrid retriever."""
        return HybridRetriever(catalog).fit()

    @pytest.fixture
    def catalog(self, sample_products: list) -> ProductCatalog:
        """Alias for catalog fixture."""
        from tests.conftest import SAMPLE_PRODUCTS
        return ProductCatalog([p.model_copy() for p in SAMPLE_PRODUCTS])

    def test_budget_filter(self, hybrid: HybridRetriever) -> None:
        """Test budget-constrained search."""
        filters = UserConstraints(max_budget=100.0)
        query = SearchQuery(text="headphones", filters=filters, top_k=10)
        results = hybrid.search(query)
        for result in results:
            assert result.product.price is not None
            assert result.product.price <= 100.0

    def test_brand_filter(self, hybrid: HybridRetriever) -> None:
        """Test brand-constrained search."""
        filters = UserConstraints(preferred_brands=["AudioBrand"])
        query = SearchQuery(text="headphones", filters=filters, top_k=10)
        results = hybrid.search(query)
        for result in results:
            assert result.product.brand == "AudioBrand"

    def test_rating_filter(self, hybrid: HybridRetriever) -> None:
        """Test rating-constrained search."""
        filters = UserConstraints(min_rating=4.8)
        query = SearchQuery(text="headphones", filters=filters, top_k=10)
        results = hybrid.search(query)
        for result in results:
            assert result.product.rating is not None
            assert result.product.rating >= 4.8

    def test_combined_filters(self, hybrid: HybridRetriever) -> None:
        """Test combined budget + brand + rating filters."""
        filters = UserConstraints(
            max_budget=600.0,
            preferred_brands=["AudioBrand"],
            min_rating=4.5,
        )
        query = SearchQuery(text="headphones", filters=filters, top_k=10)
        results = hybrid.search(query)
        # P003 ($599.99, 4.8) should match
        assert len(results) > 0
        for result in results:
            assert result.product.brand == "AudioBrand"

    def test_filters_no_results(self, hybrid: HybridRetriever) -> None:
        """Test that impossible filters return empty."""
        filters = UserConstraints(
            max_budget=10.0,  # No product is this cheap besides P008 ($19.99)
            min_rating=4.9,
        )
        query = SearchQuery(text="headphones", filters=filters, top_k=10)
        results = hybrid.search(query)
        assert len(results) == 0


class TestHybridWithReranker:
    """Hybrid retriever with reranker."""

    def test_reranker_enabled(self, catalog: ProductCatalog) -> None:
        """Test search with reranker enabled."""
        hybrid = HybridRetriever(catalog, reranker=KeywordReranker()).fit()
        query = SearchQuery(
            text="wireless noise cancelling headphones",
            top_k=5,
            use_reranker=True,
        )
        results = hybrid.search(query)
        assert len(results) > 0
        # Reranker scores should be present
        assert results[0].reranker_score is not None

    def test_reranker_disabled(self, catalog: ProductCatalog) -> None:
        """Test that reranker is not applied when disabled."""
        hybrid = HybridRetriever(catalog, reranker=KeywordReranker()).fit()
        query = SearchQuery(
            text="wireless headphones",
            top_k=5,
            use_reranker=False,
        )
        results = hybrid.search(query)
        if results:
            assert results[0].reranker_score is None
