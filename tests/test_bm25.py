"""Tests for BM25 keyword retriever."""

import pytest

from marketlens.retrieval.bm25 import BM25Retriever


class TestBM25Retriever:
    """BM25Retriever tests."""

    @pytest.fixture
    def documents(self) -> list[str]:
        """Sample documents for testing."""
        return [
            "wireless noise cancelling headphones with bluetooth",
            "budget earbuds with good sound quality and long battery",
            "premium studio headphones for professional audio mixing",
            "smart speaker with voice assistant and multi-room support",
            "ai powered noise cancelling earbuds adaptive sound",
        ]

    @pytest.fixture
    def doc_ids(self) -> list[str]:
        """Document IDs matching documents."""
        return ["D001", "D002", "D003", "D004", "D005"]

    @pytest.fixture
    def fitted_bm25(self, documents: list[str], doc_ids: list[str]) -> BM25Retriever:
        """A pre-fitted BM25 retriever."""
        return BM25Retriever().fit(documents, doc_ids)

    def test_fit_empty_raises(self) -> None:
        """Test fitting on empty documents raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            BM25Retriever().fit([])

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched doc lengths raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            BM25Retriever().fit(["doc1", "doc2"], doc_ids=["id1"])

    def test_is_fitted(self, fitted_bm25: BM25Retriever) -> None:
        """Test that the retriever reports as fitted."""
        assert fitted_bm25.is_fitted

    def test_search_not_fitted_raises(self) -> None:
        """Test that searching before fitting raises RuntimeError."""
        bm25 = BM25Retriever()
        with pytest.raises(RuntimeError, match="must be fitted"):
            bm25.search("test")

    def test_exact_match_top_result(self, fitted_bm25: BM25Retriever) -> None:
        """Test that exact query matches the right document."""
        results = fitted_bm25.search("wireless noise cancelling headphones", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "D001"  # Best match for wireless headphones

    def test_partial_match(self, fitted_bm25: BM25Retriever) -> None:
        """Test partial keyword match."""
        results = fitted_bm25.search("earbuds", top_k=3)
        assert len(results) > 0
        matching_ids = {r[0] for r in results}
        assert "D002" in matching_ids or "D005" in matching_ids

    def test_empty_query(self, fitted_bm25: BM25Retriever) -> None:
        """Test empty query returns empty results."""
        results = fitted_bm25.search("", top_k=5)
        assert results == []

    def test_no_match_query(self, fitted_bm25: BM25Retriever) -> None:
        """Test query with no matching terms."""
        results = fitted_bm25.search("xyzzy_fake_term_12345", top_k=5)
        assert results == []

    def test_top_k_limits(self, fitted_bm25: BM25Retriever) -> None:
        """Test top_k limits results."""
        results = fitted_bm25.search("wireless", top_k=2)
        assert len(results) <= 2

    def test_scores_are_positive(self, fitted_bm25: BM25Retriever) -> None:
        """Test that all returned scores are positive."""
        results = fitted_bm25.search("headphones", top_k=5)
        for _doc_id, score in results:
            assert score > 0

    def test_results_sorted_by_score(self, fitted_bm25: BM25Retriever) -> None:
        """Test results are sorted descending by score."""
        results = fitted_bm25.search("audio", top_k=5)
        scores = [score for _doc_id, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_default_doc_ids(self, documents: list[str]) -> None:
        """Test that default doc IDs are string indices."""
        bm25 = BM25Retriever().fit(documents)
        results = bm25.search("headphones", top_k=1)
        if results:
            assert results[0][0].isdigit()

    def test_tokenization(self) -> None:
        """Test tokenization is case-insensitive."""
        tokens = BM25Retriever._tokenize("Wireless HEADPHONES 123")
        assert "wireless" in tokens
        assert "headphones" in tokens
        assert "123" in tokens

    @pytest.mark.parametrize("query,expected_id", [
        ("studio headphones", "D003"),
        ("voice assistant", "D004"),
        ("budget earbuds", "D002"),
        ("ai adaptive", "D005"),
    ])
    def test_query_targets_correct_doc(
        self, fitted_bm25: BM25Retriever, query: str, expected_id: str
    ) -> None:
        """Test that specific queries target the right document."""
        results = fitted_bm25.search(query, top_k=1)
        assert len(results) > 0
        assert results[0][0] == expected_id
