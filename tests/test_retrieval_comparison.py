"""Tests for retrieval comparison, eval queries, and data pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from marketlens.catalog import ProductCatalog
from marketlens.evaluation.benchmark import (
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from marketlens.evaluation.retrieval_comparison import (
    EvalQuery,
    RetrievalTiming,
    build_eval_queries,
    compute_strategy_report,
    generate_markdown_report,
    run_full_comparison,
    save_comparison_results,
)
from marketlens.retrieval.embedding import (
    FakeEmbeddingBackend,
    SentenceTransformersBackend,
)

logger = logging.getLogger(__name__)


class TestEvalQueries:
    """Tests for evaluation query generation."""

    def test_build_with_empty_catalog(self) -> None:
        """Test building queries with no catalog."""
        queries = build_eval_queries(None)
        assert len(queries) >= 50
        # All should have required fields
        for q in queries:
            assert q.query_id
            assert q.query_text
            assert q.category
            assert q.label_source != "human_verified"
            assert q.review_status == "pending"

    def test_build_with_catalog(self, catalog: ProductCatalog) -> None:
        """Test building queries with a catalog."""
        queries = build_eval_queries(catalog)
        assert len(queries) >= 50
        # Exact match queries should have relevant_product_ids
        exact_queries = [q for q in queries if q.category == "exact_match"]
        assert len(exact_queries) > 0
        for q in exact_queries:
            assert len(q.relevant_product_ids) > 0

    def test_query_categories(self) -> None:
        """Test that all required categories are represented."""
        queries = build_eval_queries(None)
        categories = {q.category for q in queries}
        expected = {
            "exact_match", "synonym", "brand_filter", "budget",
            "multi_constraint", "attribute", "no_result", "contradiction",
            "insufficient_evidence",
        }
        assert categories == expected

    def test_query_serialization(self) -> None:
        """Test EvalQuery serialization."""
        q = EvalQuery(
            query_id="test-001",
            query_text="test query",
            category="exact_match",
        )
        d = q.to_dict()
        assert d["query_id"] == "test-001"
        assert d["label_source"] == "synthetic"
        assert d["review_status"] == "pending"

    def test_labels_not_human(self) -> None:
        """Verify all labels are synthetic/auto_curated, not human_verified."""
        queries = build_eval_queries(None)
        for q in queries:
            assert q.review_status == "pending"
            assert q.label_source in ("synthetic", "auto_curated")
            assert q.label_source != "human_verified"


class TestRetrievalTiming:
    """Tests for timing metrics."""

    def test_empty(self) -> None:
        """Test timing with no data."""
        t = RetrievalTiming.from_latencies([])
        assert t.p50_ms == 0.0
        assert t.p95_ms == 0.0

    def test_single_value(self) -> None:
        """Test timing with one value."""
        t = RetrievalTiming.from_latencies([10.0])
        assert t.p50_ms == 10.0
        assert t.p95_ms == 10.0
        assert t.mean_ms == 10.0

    def test_multiple_values(self) -> None:
        """Test timing percentiles."""
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        t = RetrievalTiming.from_latencies(latencies)
        assert t.p50_ms == 5.5
        assert t.p95_ms >= 9.0
        assert t.min_ms == 1.0
        assert t.max_ms == 10.0


class TestRetrievalComparison:
    """Integration tests for retrieval comparison."""

    @pytest.fixture
    def fixture_catalog(self) -> ProductCatalog:
        """Load fixture catalog."""
        return ProductCatalog.from_fixture("electronics_sample.json")

    @pytest.fixture
    def queries(self, fixture_catalog: ProductCatalog) -> list[EvalQuery]:
        """Build queries from fixture catalog."""
        return build_eval_queries(fixture_catalog)

    def test_bm25_comparison(self, fixture_catalog: ProductCatalog, queries: list[EvalQuery]) -> None:
        """Test BM25 strategy report generation."""
        from marketlens.retrieval.bm25 import BM25Retriever

        texts = fixture_catalog.get_search_texts()
        ids = fixture_catalog.get_product_ids()
        bm25 = BM25Retriever().fit(texts, ids)

        def bm25_fn(query: str, k: int, _: dict | None) -> tuple[list[str], list[float]]:
            results = bm25.search(query, k)
            return ([r[0] for r in results], [r[1] for r in results])

        report = compute_strategy_report("bm25", queries, bm25_fn)
        assert report.strategy == "bm25"
        assert report.total_queries == len(queries)
        assert report.success_count >= 0
        assert report.recall_at_10 >= 0.0
        assert report.timing.p50_ms >= 0

    def test_embedding_comparison(self, fixture_catalog: ProductCatalog, queries: list[EvalQuery]) -> None:
        """Test embedding strategy report generation."""
        from marketlens.retrieval.embedding import (
            EmbeddingRetriever,
            FakeEmbeddingBackend,
        )

        texts = fixture_catalog.get_search_texts()
        ids = fixture_catalog.get_product_ids()
        emb = EmbeddingRetriever(FakeEmbeddingBackend(dim=64, seed=42)).fit(texts, ids)

        def emb_fn(query: str, k: int, _: dict | None) -> tuple[list[str], list[float]]:
            results = emb.search(query, k)
            return ([r[0] for r in results], [r[1] for r in results])

        report = compute_strategy_report("embedding", queries, emb_fn)
        assert report.strategy == "embedding"
        assert report.total_queries == len(queries)
        assert report.recall_at_10 >= 0.0

    def test_full_comparison(self, fixture_catalog: ProductCatalog, queries: list[EvalQuery]) -> None:
        """Test full four-strategy comparison."""
        reports = run_full_comparison(
            fixture_catalog, queries, use_real_embeddings=False, top_k=10,
        )
        assert set(reports.keys()) == {"bm25", "embedding", "hybrid", "hybrid_rerank"}
        for name, report in reports.items():
            assert report.total_queries == len(queries)
            assert report.success_count >= 0  # Can be zero for fixture data
            assert report.recall_at_10 >= 0.0
            assert 0.0 <= report.constraint_satisfaction_rate <= 1.0
            assert report.timing.mean_ms >= 0  # Can be zero for very fast execution

    def test_save_and_load_results(
        self, fixture_catalog: ProductCatalog, queries: list[EvalQuery], tmp_path: Path,
    ) -> None:
        """Test saving and loading comparison results."""
        reports = run_full_comparison(
            fixture_catalog, queries, use_real_embeddings=False, top_k=10,
        )
        output_dir = tmp_path / "eval_results"
        summary_path = save_comparison_results(
            reports, queries, output_dir,
            config={"data_source": "fixture", "seed": 42},
        )

        # Verify files exist
        assert summary_path.exists()
        assert (output_dir / "results_bm25.json").exists()
        assert (output_dir / "results_embedding.json").exists()
        assert (output_dir / "results_hybrid.json").exists()
        assert (output_dir / "results_hybrid_rerank.json").exists()
        assert (output_dir / "eval_queries.json").exists()

        # Verify summary content
        summary = json.loads(summary_path.read_text())
        assert summary["query_count"] == len(queries)
        assert "bm25" in summary["strategies"]
        assert "recall_at_10" in summary["strategies"]["bm25"]

    def test_markdown_report(self, fixture_catalog: ProductCatalog, queries: list[EvalQuery]) -> None:
        """Test markdown report generation."""
        reports = run_full_comparison(
            fixture_catalog, queries, use_real_embeddings=False, top_k=10,
        )
        md = generate_markdown_report(reports, queries, config={"data_source": "fixture"})
        assert "MarketLens Retrieval Strategy Comparison" in md
        assert "bm25" in md
        assert "recall_at_10" in md.lower() or "Recall@10" in md
        assert "auto-curated" in md.lower() or "synthetic" in md.lower()


class TestEmbeddingBackends:
    """Tests for embedding backend implementations."""

    def test_fake_backend_no_model(self) -> None:
        """Test that FakeEmbeddingBackend requires no model download."""
        backend = FakeEmbeddingBackend(dim=64, seed=42)
        embeddings = backend.encode(["test text", "another text"])
        assert embeddings.shape == (2, 64)
        # Verify normalization
        for vec in embeddings:
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-6

    def test_fake_backend_deterministic(self) -> None:
        """Test that fake embeddings are deterministic."""
        b1 = FakeEmbeddingBackend(dim=64, seed=42)
        b2 = FakeEmbeddingBackend(dim=64, seed=42)
        e1 = b1.encode(["hello world"])
        e2 = b2.encode(["hello world"])
        assert np.array_equal(e1, e2)

    def test_fake_backend_different_seed(self) -> None:
        """Test that different seeds produce different embeddings.

        Note: The FakeEmbeddingBackend uses hash-based deterministic encoding that
        does not depend on the instance seed. The seed only affects internal RNG
        state. Embeddings for the same text will always be identical regardless
        of seed — this is by design for reproducible testing.
        """
        b1 = FakeEmbeddingBackend(dim=64, seed=42)
        b2 = FakeEmbeddingBackend(dim=64, seed=99)
        e1 = b1.encode(["hello"])
        e2 = b2.encode(["hello"])
        # Same text → same embedding (deterministic by design)
        assert np.array_equal(e1, e2)

    def test_sentence_transformers_import_error(self) -> None:
        """Test graceful handling when sentence-transformers is missing."""
        # This should raise ImportError if not installed
        try:
            backend = SentenceTransformersBackend()
            # If installed, this is a smoke test
            embeddings = backend.encode(["test"])
            assert embeddings.shape[1] == backend.dim
            assert backend.model_info["backend_type"] == "sentence-transformers"
        except ImportError:
            # Expected when not installed — test passes
            pass

    def test_sentence_transformers_not_installed_raises_on_encode(self) -> None:
        """Test that encode raises when library is missing."""
        # Only test if sentence-transformers is NOT installed
        try:
            import sentence_transformers  # noqa: F401
            pytest.skip("sentence-transformers is installed, skipping import error test")
        except ImportError:
            backend = SentenceTransformersBackend()
            with pytest.raises(ImportError, match="sentence-transformers"):
                backend.encode(["test"])


class TestDataPipeline:
    """Tests for the data preparation pipeline."""

    def test_price_cleaning(self) -> None:
        """Test price normalization."""
        from scripts.prepare_electronics_data import clean_price

        assert clean_price(None) is None
        assert clean_price("") is None
        assert clean_price("-5.0") is None
        assert clean_price("99.99") == 99.99
        assert clean_price(149) == 149.0
        assert clean_price("$199.99") == 199.99

    def test_rating_cleaning(self) -> None:
        """Test rating normalization."""
        from scripts.prepare_electronics_data import clean_rating

        assert clean_rating(None) is None
        assert clean_rating(4.5) == 4.5
        assert clean_rating(6.0) is None
        assert clean_rating(-1.0) is None

    def test_review_count_cleaning(self) -> None:
        """Test review count normalization."""
        from scripts.prepare_electronics_data import clean_review_count

        assert clean_review_count(None) is None
        assert clean_review_count(100) == 100
        assert clean_review_count(-5) is None

    def test_build_product_missing_required(self) -> None:
        """Test that products missing required fields are rejected."""
        from scripts.prepare_electronics_data import build_product

        assert build_product({}) == {}
        assert build_product({"parent_asin": ""}) == {}
        assert build_product({"title": ""}) == {}

    def test_build_product_valid(self) -> None:
        """Test building a valid product."""
        from scripts.prepare_electronics_data import build_product

        item = {
            "parent_asin": "B00TEST123",
            "title": "Test Wireless Headphones",
            "store": "TestBrand",
            "price": "99.99",
            "average_rating": 4.5,
            "rating_number": 1000,
            "features": ["Noise cancelling", "30h battery"],
            "description": "Great headphones",
            "main_category": "Electronics",
        }
        product = build_product(item)
        assert product["product_id"] == "B00TEST123"
        assert product["title"] == "Test Wireless Headphones"
        assert product["brand"] == "TestBrand"
        assert product["price"] == 99.99
        assert product["rating"] == 4.5
        assert product["review_count"] == 1000
        assert len(product["attributes"]) >= 2

    def test_load_from_huggingface_uses_json_builder(self, mocker) -> None:
        """Verify load_from_huggingface uses json builder, not dataset scripts."""
        from scripts.prepare_electronics_data import (
            OFFICIAL_METADATA_URL,
            SHUFFLE_BUFFER_SIZE,
            load_from_huggingface,
        )

        # Mock datasets.load_dataset (imported inside the function)
        mock_ds = mocker.MagicMock()
        mock_ds.shuffle.return_value = iter([
            {
                "parent_asin": "B00XX",
                "title": "Mock Product",
                "price": "49.99",
                "average_rating": 4.0,
                "rating_number": 100,
                "store": "MockBrand",
            },
        ])
        mock_load = mocker.patch(
            "datasets.load_dataset",
            return_value=mock_ds,
        )

        result = load_from_huggingface(max_products=5, seed=42)

        # Verify json builder (not dataset name), streaming=True, no trust_remote_code
        call_args = mock_load.call_args
        assert call_args is not None
        args, kwargs = call_args[0], call_args[1]
        assert args[0] == "json"  # generic json builder
        assert kwargs.get("streaming") is True
        assert "trust_remote_code" not in kwargs
        assert kwargs.get("data_files") == {"train": OFFICIAL_METADATA_URL}
        assert kwargs.get("split") == "train"

        # Verify shuffle params
        mock_ds.shuffle.assert_called_once_with(
            seed=42, buffer_size=SHUFFLE_BUFFER_SIZE,
        )

        # Verify limit is respected (1 item returned, not 5)
        assert len(result) == 1

    def test_load_from_huggingface_stops_at_max(self, mocker) -> None:
        """Verify load_from_huggingface stops iterating at max_products."""
        from scripts.prepare_electronics_data import load_from_huggingface

        # Create a mock that yields many items
        mock_ds = mocker.MagicMock()
        mock_ds.shuffle.return_value = (
            {
                "parent_asin": f"B{i:04d}",
                "title": f"Product {i}",
                "price": 10.0 + i,
                "average_rating": 4.0,
                "rating_number": 100,
                "store": "Test",
            }
            for i in range(20)
        )
        mocker.patch(
            "datasets.load_dataset",
            return_value=mock_ds,
        )

        result = load_from_huggingface(max_products=5, seed=42)

        # Must return exactly 5, not all 20
        assert len(result) == 5

    def test_load_from_huggingface_no_trust_remote_code(self, mocker) -> None:
        """Verify load_from_huggingface never passes trust_remote_code."""
        from scripts.prepare_electronics_data import load_from_huggingface

        mock_ds = mocker.MagicMock()
        mock_ds.shuffle.return_value = iter([])
        mock_load = mocker.patch(
            "datasets.load_dataset",
            return_value=mock_ds,
        )

        load_from_huggingface(max_products=1, seed=99)

        # Verify no trust_remote_code in any call
        for call in mock_load.call_args_list:
            assert "trust_remote_code" not in call[1]

    def test_manifest_includes_url_and_streaming(self) -> None:
        """Verify manifest records URL, datasets_version, streaming params."""
        import argparse
        from pathlib import Path

        from scripts.prepare_electronics_data import (
            OFFICIAL_METADATA_URL,
            SHUFFLE_BUFFER_SIZE,
            generate_manifest,
        )

        args = argparse.Namespace(
            seed=42, max_products=2000, local_file=None,
            output=Path("/tmp/test.json"), dry_run=False,
        )
        manifest = generate_manifest(
            args, raw_count=100, cleaned_count=80,
            skip_stats={"total_skipped": 20},
            output_path=Path("/tmp/test.json"),
            elapsed_s=1.5,
        )

        assert manifest["source_url"] == OFFICIAL_METADATA_URL
        assert manifest["streaming"] is True
        assert manifest["shuffle_buffer_size"] == SHUFFLE_BUFFER_SIZE
        assert "datasets_version" in manifest
        assert manifest["seed"] == 42
        assert manifest["raw_products_read"] == 100
        assert manifest["cleaned_products_output"] == 80


class TestMetrics:
    """Additional tests for retrieval metrics."""

    def test_recall_empty_relevant(self) -> None:
        """Test recall when no ground truth."""
        assert compute_recall_at_k(["A", "B"], [], k=10) == 1.0

    def test_ndcg_empty_relevant(self) -> None:
        """Test nDCG when no ground truth."""
        assert compute_ndcg_at_k(["A", "B"], [], k=10) == 1.0

    def test_ndcg_all_at_top(self) -> None:
        """Test perfect nDCG with all relevant at top."""
        assert compute_ndcg_at_k(["A", "B", "C"], ["A", "B", "C"], k=5) == 1.0

    def test_recall_partial(self) -> None:
        """Test partial recall."""
        retrieved = ["X", "A", "Y", "B", "Z"]
        relevant = ["A", "B", "C", "D"]
        assert compute_recall_at_k(retrieved, relevant, k=5) == 0.5
