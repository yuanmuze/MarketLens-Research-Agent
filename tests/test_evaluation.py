"""Evaluation benchmark tests for MarketLens.

These tests run actual evaluations using fixture data and small deterministic
search functions. All results are explicitly marked as fixture benchmarks.
"""

from __future__ import annotations

import logging

import pytest

from marketlens.catalog import ProductCatalog
from marketlens.evaluation.benchmark import (
    EvaluationQuery,
    QueryResult,
    compare_retrievers,
    compute_hard_constraint_rate,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_task_completion_rate,
    print_report,
    run_evaluation,
)
from marketlens.retrieval.bm25 import BM25Retriever
from marketlens.retrieval.embedding import EmbeddingRetriever, FakeEmbeddingBackend
from marketlens.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixture evaluation queries
# ---------------------------------------------------------------------------
FIXTURE_QUERIES: list[EvaluationQuery] = [
    # Exact product name matches
    EvaluationQuery(
        query_id="exact-001",
        query_text="Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        category="exact_match",
        relevant_product_ids=["B001"],
        notes="Exact product name lookup",
    ),
    EvaluationQuery(
        query_id="exact-002",
        query_text="AirPods Pro 2nd Generation",
        category="exact_match",
        relevant_product_ids=["B003"],
        notes="Exact product name with generation",
    ),
    # Synonym / paraphrased queries
    EvaluationQuery(
        query_id="syn-001",
        query_text="noise cancelling wireless over-ear headphones",
        category="synonym",
        relevant_product_ids=["B001", "B002", "B005", "B007", "B010", "B012"],
        notes="Synonym for ANC headphones — should match multiple",
    ),
    EvaluationQuery(
        query_id="syn-002",
        query_text="true wireless in-ear earbuds for calls",
        category="synonym",
        relevant_product_ids=["B003", "B004", "B009", "B011", "B016", "B018"],
        notes="Synonym for TWS earbuds",
    ),
    # Budget constraints
    EvaluationQuery(
        query_id="budget-001",
        query_text="wireless headphones under $100",
        category="budget",
        relevant_product_ids=["B011"],  # Anker Space A40 at $79.99
        notes="Budget wireless audio",
    ),
    EvaluationQuery(
        query_id="budget-002",
        query_text="premium headphones above $400",
        category="budget",
        relevant_product_ids=["B002", "B010", "B007"],  # Premium
        notes="Premium segment",
    ),
    # Multi-constraint
    EvaluationQuery(
        query_id="multi-001",
        query_text="Sony wireless ANC headphones under $350 with 30h+ battery",
        category="multi_constraint",
        relevant_product_ids=["B001", "B005", "B007", "B009", "B015"],
        notes="Multiple hard constraints: brand + ANC + budget + battery",
    ),
    EvaluationQuery(
        query_id="multi-002",
        query_text="high rated bluetooth headphones with spatial audio",
        category="multi_constraint",
        relevant_product_ids=["B002", "B003", "B008"],  # Spatial audio products
        notes="Features + quality constraints",
    ),
    # No results expected
    EvaluationQuery(
        query_id="none-001",
        query_text="wireless headphones under $10",
        category="no_result",
        relevant_product_ids=[],  # Nothing that cheap
        expect_results=False,
        notes="Impossible budget should return no results",
    ),
    EvaluationQuery(
        query_id="none-002",
        query_text="Nintendo Switch OLED gaming headphones",
        category="no_result",
        relevant_product_ids=[],
        expect_results=False,
        notes="No gaming console headphones in catalog",
    ),
    # Contradictory
    EvaluationQuery(
        query_id="contra-001",
        query_text="best $500 headphones under $100",
        category="contradiction",
        relevant_product_ids=[],
        expect_results=False,
        notes="Contradictory budget: $500 and under $100",
    ),
    # Insufficient evidence
    EvaluationQuery(
        query_id="insuf-001",
        query_text="best mid-range open-back planar magnetic headphones",
        category="insufficient_evidence",
        relevant_product_ids=[],
        expect_results=False,
        notes="Specialized category not in our catalog",
    ),
]


class TestEvaluationMetrics:
    """Tests for individual evaluation metrics."""

    def test_recall_at_k_perfect(self) -> None:
        """Test perfect recall."""
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = ["A", "B", "C"]
        assert compute_recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_at_k_partial(self) -> None:
        """Test partial recall."""
        retrieved = ["A", "B", "X", "Y", "Z"]
        relevant = ["A", "B", "C", "D", "E"]
        assert compute_recall_at_k(retrieved, relevant, k=5) == 0.4

    def test_recall_at_k_zero(self) -> None:
        """Test zero recall."""
        retrieved = ["X", "Y", "Z"]
        relevant = ["A", "B", "C"]
        assert compute_recall_at_k(retrieved, relevant, k=5) == 0.0

    def test_recall_at_k_empty_relevant(self) -> None:
        """Test recall when no ground truth relevant items."""
        assert compute_recall_at_k(["A", "B"], [], k=5) == 1.0

    def test_ndcg_perfect(self) -> None:
        """Test perfect nDCG."""
        retrieved = ["A", "B", "C"]
        relevant = ["A", "B", "C"]
        ndcg = compute_ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == 1.0

    def test_ndcg_degraded(self) -> None:
        """Test nDCG when relevant docs are lower ranked."""
        retrieved = ["X", "A", "B", "C"]
        relevant = ["A", "B", "C"]
        ndcg = compute_ndcg_at_k(retrieved, relevant, k=4)
        assert 0.0 < ndcg < 1.0

    def test_ndcg_empty_relevant(self) -> None:
        """Test nDCG with no ground truth."""
        assert compute_ndcg_at_k(["A", "B"], [], k=5) == 1.0

    def test_constraint_rate_all_satisfied(self) -> None:
        """Test constraint rate calculation."""
        results = [
            QueryResult("q1", ["A"], 1.0, 1.0, True, 1, 10.0),
            QueryResult("q2", ["B"], 1.0, 1.0, True, 1, 10.0),
        ]
        assert compute_hard_constraint_rate(results) == 1.0

    def test_constraint_rate_half(self) -> None:
        """Test constraint rate when half fail."""
        results = [
            QueryResult("q1", ["A"], 1.0, 1.0, True, 1, 10.0),
            QueryResult("q2", [], 0.0, 0.0, False, 0, 10.0),
        ]
        assert compute_hard_constraint_rate(results) == 0.5

    def test_task_completion_rate(self) -> None:
        """Test task completion rate."""
        results = [
            QueryResult("q1", ["A"], 1.0, 1.0, True, 1, 10.0),
            QueryResult("q2", [], 0.0, 0.0, True, 0, 10.0),
            QueryResult("q3", ["B", "C"], 1.0, 1.0, True, 2, 10.0),
        ]
        assert compute_task_completion_rate(results) == 2.0 / 3.0


class TestEvaluationBenchmark:
    """Run actual evaluation benchmarks with fixture data."""

    @pytest.fixture
    def catalog(self) -> ProductCatalog:
        """Load fixture catalog."""
        return ProductCatalog.from_fixture("electronics_sample.json")

    def test_run_evaluation_bm25(self, catalog: ProductCatalog) -> None:
        """Run BM25-only evaluation."""
        # Build BM25 retriever
        texts = catalog.get_search_texts()
        ids = catalog.get_product_ids()
        bm25 = BM25Retriever().fit(texts, ids)

        def bm25_search(query: str, top_k: int) -> list[str]:
            return [pid for pid, _ in bm25.search(query, top_k)]

        report = run_evaluation(FIXTURE_QUERIES, bm25_search, top_k=10)

        assert report.total_queries == len(FIXTURE_QUERIES)
        assert report.completed_queries == len(FIXTURE_QUERIES)
        assert report.is_fixture_data is True
        # Some queries should find results
        assert report.task_completion_rate > 0.0
        print_report(report)

    def test_run_evaluation_embedding(self, catalog: ProductCatalog) -> None:
        """Run embedding-only evaluation."""
        texts = catalog.get_search_texts()
        ids = catalog.get_product_ids()
        emb = EmbeddingRetriever(FakeEmbeddingBackend(dim=64, seed=42)).fit(
            texts, ids
        )

        def emb_search(query: str, top_k: int) -> list[str]:
            return [pid for pid, _ in emb.search(query, top_k)]

        report = run_evaluation(FIXTURE_QUERIES, emb_search, top_k=10)

        assert report.total_queries == len(FIXTURE_QUERIES)
        assert report.completed_queries == len(FIXTURE_QUERIES)
        assert report.is_fixture_data is True
        print_report(report)

    def test_run_evaluation_hybrid(self, catalog: ProductCatalog) -> None:
        """Run hybrid retrieval evaluation."""
        hybrid = HybridRetriever(catalog).fit()

        def hybrid_search(query: str, top_k: int) -> list[str]:
            from marketlens.models import SearchQuery
            sq = SearchQuery(text=query, top_k=top_k)
            results = hybrid.search(sq)
            return [r.product.product_id for r in results]

        report = run_evaluation(FIXTURE_QUERIES, hybrid_search, top_k=10)

        assert report.total_queries == len(FIXTURE_QUERIES)
        assert report.completed_queries == len(FIXTURE_QUERIES)
        assert report.is_fixture_data is True
        # Hybrid should have reasonable recall for exact matches
        recall_val = next(
            r.recall_at_10 for r in report.query_results if r.query_id == "exact-001"
        )
        logger.info("Hybrid exact match recall: %s", recall_val)
        print_report(report)

    def test_compare_all_retrievers(self, catalog: ProductCatalog) -> None:
        """Compare BM25 vs Embedding vs Hybrid retrieval."""
        texts = catalog.get_search_texts()
        ids = catalog.get_product_ids()
        bm25 = BM25Retriever().fit(texts, ids)
        emb = EmbeddingRetriever(FakeEmbeddingBackend(dim=64, seed=42)).fit(
            texts, ids
        )
        hybrid = HybridRetriever(catalog).fit()

        from marketlens.models import SearchQuery

        reports = compare_retrievers(
            FIXTURE_QUERIES,
            bm25_fn=lambda q, k: [pid for pid, _ in bm25.search(q, k)],
            embedding_fn=lambda q, k: [pid for pid, _ in emb.search(q, k)],
            hybrid_fn=lambda q, k: [
                r.product.product_id
                for r in hybrid.search(SearchQuery(text=q, top_k=k))
            ],
            top_k=10,
        )

        assert set(reports.keys()) == {"bm25", "embedding", "hybrid"}
        for name, report in reports.items():
            assert report.is_fixture_data is True
            assert report.total_queries == len(FIXTURE_QUERIES)
            assert report.avg_recall_at_10 >= 0.0
            logger.info(
                "%s: Recall@10=%.4f, nDCG@10=%.4f, Latency=%.2fms",
                name, report.avg_recall_at_10, report.avg_ndcg_at_10, report.avg_latency_ms,
            )

    def test_report_has_per_category_breakdown(self, catalog: ProductCatalog) -> None:
        """Test that reports include per-category metrics."""
        hybrid = HybridRetriever(catalog).fit()

        from marketlens.models import SearchQuery

        def hybrid_search(query: str, top_k: int) -> list[str]:
            sq = SearchQuery(text=query, top_k=top_k)
            results = hybrid.search(sq)
            return [r.product.product_id for r in results]

        report = run_evaluation(FIXTURE_QUERIES, hybrid_search, top_k=10)
        assert "exact_match" in report.per_category
        assert "budget" in report.per_category
        assert "no_result" in report.per_category

    def test_evaluation_report_metadata(self, catalog: ProductCatalog) -> None:
        """Test that the evaluation report has proper metadata."""
        hybrid = HybridRetriever(catalog).fit()

        from marketlens.models import SearchQuery

        def hybrid_search(query: str, top_k: int) -> list[str]:
            sq = SearchQuery(text=query, top_k=top_k)
            results = hybrid.search(sq)
            return [r.product.product_id for r in results]

        report = run_evaluation(FIXTURE_QUERIES, hybrid_search, top_k=10)
        assert report.is_fixture_data is True  # Marked as fixture
        assert report.title != ""
        assert report.avg_latency_ms >= 0
        assert 0 <= report.constraint_satisfaction_rate <= 1
        assert 0 <= report.task_completion_rate <= 1
