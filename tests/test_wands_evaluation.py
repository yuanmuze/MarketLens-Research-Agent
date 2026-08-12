"""Tests for WANDS data loading, metrics, and evaluation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from marketlens.evaluation.metrics import (
    compute_all_metrics,
    exact_mrr_at_10,
    exact_success_at_10,
    judged_ratio_at_k,
    macro_average,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from marketlens.evaluation.wands import (
    LABEL_MAP,
    WandsProduct,
    get_judged_products,
    get_relevant_products,
    load_qrels,
)


class TestLabelMapping:
    """Test WANDS label → numeric grade mapping."""

    def test_exact_map(self) -> None:
        """Exact maps to 2."""
        assert LABEL_MAP["Exact"] == 2

    def test_partial_map(self) -> None:
        """Partial maps to 1."""
        assert LABEL_MAP["Partial"] == 1

    def test_irrelevant_map(self) -> None:
        """Irrelevant maps to 0."""
        assert LABEL_MAP["Irrelevant"] == 0


class TestNDCG:
    """Test nDCG computation."""

    def test_perfect_ndcg(self) -> None:
        """Perfect ranking gives nDCG=1."""
        qrels = {"A": 2, "B": 1, "C": 1}
        assert ndcg_at_k(["A", "B", "C"], qrels, 3) == 1.0

    def test_degraded_ndcg(self) -> None:
        """Degraded ranking gives lower nDCG."""
        qrels = {"A": 2, "B": 1, "C": 1}
        ndcg_best = ndcg_at_k(["A", "B", "C"], qrels, 3)
        ndcg_worse = ndcg_at_k(["C", "B", "A"], qrels, 3)
        assert ndcg_worse < ndcg_best

    def test_no_relevant(self) -> None:
        """No relevant docs gives nDCG=0."""
        qrels = {"A": 0, "B": 0}
        assert ndcg_at_k(["C", "D"], qrels, 3) < 0.01

    def test_graded_prefers_exact(self) -> None:
        """Exact (2) should rank higher than Partial (1)."""
        qrels = {"A": 2, "B": 1}
        ndcg_a_first = ndcg_at_k(["A", "B"], qrels, 2)
        ndcg_b_first = ndcg_at_k(["B", "A"], qrels, 2)
        assert ndcg_a_first > ndcg_b_first


class TestPrecision:
    """Test Precision computation."""

    def test_perfect_precision(self) -> None:
        """All results relevant."""
        qrels = {"A": 2, "B": 1}
        assert precision_at_k(["A", "B", "C"], qrels, 3) == 2 / 3

    def test_no_relevant(self) -> None:
        """No relevant results."""
        qrels = {"A": 0, "B": 0}
        assert precision_at_k(["C", "D"], qrels, 5) == 0.0

    def test_irrelevant_not_counted(self) -> None:
        """Irrelevant (0) not counted as relevant."""
        qrels = {"A": 0}
        assert precision_at_k(["A"], qrels, 1) == 0.0


class TestMRR:
    """Test MRR computation."""

    def test_exact_at_rank_1(self) -> None:
        """Exact at rank 1 → MRR=1."""
        qrels = {"A": 2}
        assert exact_mrr_at_10(["A", "B"], qrels) == 1.0

    def test_exact_at_rank_3(self) -> None:
        """Exact at rank 3 → MRR=1/3."""
        qrels = {"C": 2}
        assert exact_mrr_at_10(["A", "B", "C"], qrels) == pytest.approx(1.0 / 3)

    def test_no_exact(self) -> None:
        """No Exact in top-10 → MRR=0."""
        qrels = {"A": 1, "B": 0}
        assert exact_mrr_at_10(["A", "B", "C"], qrels) == 0.0


class TestSuccess:
    """Test Success@10."""

    def test_has_exact(self) -> None:
        """Contains Exact → 1."""
        qrels = {"A": 2}
        assert exact_success_at_10(["A"], qrels) == 1

    def test_no_exact(self) -> None:
        """No Exact → 0."""
        qrels = {"A": 1}
        assert exact_success_at_10(["A"], qrels) == 0


class TestRecall:
    """Test Recall computation."""

    def test_full_recall(self) -> None:
        """All relevant items found."""
        assert recall_at_k(["A", "B"], {"A", "B"}, 10) == 1.0

    def test_partial_recall(self) -> None:
        """Some relevant items found."""
        assert recall_at_k(["A", "C"], {"A", "B", "D", "E"}, 10) == 0.25

    def test_empty_relevant(self) -> None:
        """No known relevant items → 1.0."""
        assert recall_at_k(["X"], set(), 10) == 1.0


class TestJudgedRatio:
    """Test judged/unjudged ratio at k."""

    def test_all_judged(self) -> None:
        """All results are in the judged set."""
        r = judged_ratio_at_k(["A", "B", "C"], {"A", "B", "C"}, 10)
        assert r["judged_ratio"] == 1.0
        assert r["unjudged_ratio"] == 0.0

    def test_half_judged(self) -> None:
        """Half judged."""
        r = judged_ratio_at_k(["A", "B", "C", "D"], {"A", "B"}, 10)
        assert r["judged_ratio"] == 0.5
        assert r["unjudged_ratio"] == 0.5


class TestMacroAverage:
    """Test macro averaging."""

    def test_simple_macro(self) -> None:
        """Mean of per-query values."""
        assert macro_average([0.5, 0.5, 1.0]) == pytest.approx(2 / 3)

    def test_empty(self) -> None:
        """Empty list gives 0."""
        assert macro_average([]) == 0.0


class TestComputeAllMetrics:
    """Test the aggregate metric computation."""

    def test_basic_aggregation(self) -> None:
        """Test compute_all_metrics with simple inputs."""
        q_results = [
            {
                "query_id": "q1",
                "retrieved_ids": ["A", "B", "C", "D", "E"],
                "qrels": {"A": 2, "B": 1, "Z": 1},
                "relevant_ids": {"A", "B", "Z"},
                "judged_ids": {"A", "B", "Z"},
            },
        ]
        m = compute_all_metrics(q_results)
        assert m["total_queries"] == 1
        assert 0 <= m["ndcg_at_10"] <= 1
        assert 0 <= m["precision_at_10"] <= 1
        assert 0 <= m["exact_mrr_at_10"] <= 1
        assert 0 <= m["exact_success_at_10"] <= 1
        assert m["top10_judged_ratio"] >= 0
        assert m["top10_unjudged_ratio"] >= 0


class TestQrelsLoading:
    """Test qrels loading and helper functions."""

    def test_load_qrels_from_temp(self, tmp_path: Path) -> None:
        """Test loading qrels from a temp TSV file."""
        lines = [
            "id\tquery_id\tproduct_id\tlabel\n",
            "1\tQ1\tP1\tExact\n",
            "2\tQ1\tP2\tPartial\n",
            "3\tQ1\tP3\tIrrelevant\n",
            "4\tQ2\tP1\tExact\n",
            "5\tQ2\tP1\tExact\n",  # Duplicate pair → majority Exact
        ]
        path = tmp_path / "labels.tsv"
        path.write_text("".join(lines), encoding="utf-8")

        qrels = load_qrels(path)
        assert len(qrels) == 2
        assert qrels["Q1"]["P1"] == 2
        assert qrels["Q1"]["P2"] == 1
        assert qrels["Q1"]["P3"] == 0
        # Q2/P1: two Exact votes → majority Exact=2
        assert qrels["Q2"]["P1"] == 2

    def test_get_judged_products(self) -> None:
        """Test get_judged_products."""
        qrels = {"q1": {"p1": 2, "p2": 1}}
        judged = get_judged_products(qrels, "q1")
        assert judged == {"p1", "p2"}
        assert get_judged_products(qrels, "nonexistent") == set()

    def test_get_relevant_products(self) -> None:
        """Test get_relevant_products with grade filter."""
        qrels = {"q1": {"p1": 2, "p2": 1, "p3": 0}}
        assert get_relevant_products(qrels, "q1", min_grade=1) == {"p1", "p2"}
        assert get_relevant_products(qrels, "q1", min_grade=2) == {"p1"}


class TestWandsProduct:
    """Test WandsProduct adapter."""

    def test_to_search_text(self) -> None:
        """Test search text construction from WANDS fields."""
        wp = WandsProduct(
            product_id="P1",
            title="Modern Sofa",
            product_class="Furniture > Sofas",
            description="Comfortable 3-seater",
            rating=4.5,
            review_count=100,
        )
        text = wp.to_search_text()
        assert "Modern Sofa" in text
        assert "Furniture" in text
        assert "Comfortable" in text

    def test_to_dict(self) -> None:
        """Test conversion to dict (no price/brand)."""
        wp = WandsProduct("P1", "Sofa", "Furniture", "Desc", 4.0, 50)
        d = wp.to_dict()
        assert d["product_id"] == "P1"
        assert d["price"] is None
        assert d["brand"] == ""
        assert d["rating"] == 4.0


class TestRetrievalIsolation:
    """Verify retrieval never accesses qrels."""

    def test_wands_product_has_no_qrel_access(self) -> None:
        """WandsProduct has no qrels-related methods."""
        wp = WandsProduct("P1", "T", "C", "D", None, None)
        assert not hasattr(wp, "qrels")
        assert not hasattr(wp, "relevance")
        assert not hasattr(wp, "label")

    def test_generate_candidates_from_full_catalog(self, tmp_path: Path) -> None:
        """Test that build_catalog creates products without qrel info."""
        from marketlens.catalog import ProductCatalog
        from marketlens.models import Product

        # Products must not know about labels
        p = Product(product_id="test", title="Test", brand=None, price=None)
        cat = ProductCatalog([p])
        # Neither product nor catalog has qrels access
        assert not hasattr(p, "qrels")
        assert not hasattr(cat, "qrels")
