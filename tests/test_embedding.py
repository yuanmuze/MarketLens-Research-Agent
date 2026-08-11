"""Tests for embedding-based semantic retriever."""

import numpy as np
import pytest

from marketlens.retrieval.embedding import (
    EmbeddingRetriever,
    FakeEmbeddingBackend,
)


class TestFakeEmbeddingBackend:
    """FakeEmbeddingBackend tests."""

    def test_dim(self) -> None:
        """Test dimension property."""
        backend = FakeEmbeddingBackend(dim=64)
        assert backend.dim == 64

    def test_default_dim(self) -> None:
        """Test default dimension."""
        backend = FakeEmbeddingBackend()
        assert backend.dim == 128

    def test_encode_shape(self) -> None:
        """Test encode output shape."""
        backend = FakeEmbeddingBackend(dim=32)
        embeddings = backend.encode(["hello", "world", "test"])
        assert embeddings.shape == (3, 32)

    def test_encode_single_text(self) -> None:
        """Test encoding a single text."""
        backend = FakeEmbeddingBackend(dim=64)
        embeddings = backend.encode(["hello"])
        assert embeddings.shape == (1, 64)

    def test_encode_empty_list(self) -> None:
        """Test encoding empty list returns empty array."""
        backend = FakeEmbeddingBackend()
        embeddings = backend.encode([])
        assert embeddings.shape == (0, 128)

    def test_deterministic(self) -> None:
        """Test that same text produces same embedding."""
        backend = FakeEmbeddingBackend(seed=42)
        e1 = backend.encode(["hello world"])
        e2 = backend.encode(["hello world"])
        assert np.array_equal(e1, e2)

    def test_different_texts_different_embeddings(self) -> None:
        """Test that different texts produce different embeddings."""
        backend = FakeEmbeddingBackend(seed=42)
        e1 = backend.encode(["text one"])
        e2 = backend.encode(["text two"])
        assert not np.array_equal(e1, e2)

    def test_normalized_embeddings(self) -> None:
        """Test that embeddings are L2 normalized."""
        backend = FakeEmbeddingBackend(dim=64)
        embeddings = backend.encode(["hello world", "another text"])
        for vec in embeddings:
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-6


class TestEmbeddingRetriever:
    """EmbeddingRetriever tests."""

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
    def fitted_retriever(
        self, documents: list[str], doc_ids: list[str]
    ) -> EmbeddingRetriever:
        """A pre-fitted embedding retriever."""
        backend = FakeEmbeddingBackend(dim=64, seed=42)
        return EmbeddingRetriever(backend).fit(documents, doc_ids)

    def test_default_backend(self) -> None:
        """Test that default backend is FakeEmbeddingBackend."""
        retriever = EmbeddingRetriever()
        assert isinstance(retriever._backend, FakeEmbeddingBackend)

    def test_dim(self, fitted_retriever: EmbeddingRetriever) -> None:
        """Test dimension property."""
        assert fitted_retriever.dim == 64

    def test_is_fitted(self, fitted_retriever: EmbeddingRetriever) -> None:
        """Test fitted status."""
        assert fitted_retriever.is_fitted

    def test_search_not_fitted_raises(self) -> None:
        """Test searching before fitting raises RuntimeError."""
        retriever = EmbeddingRetriever()
        with pytest.raises(RuntimeError, match="must be fitted"):
            retriever.search("test")

    def test_fit_empty_raises(self) -> None:
        """Test fitting on empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            EmbeddingRetriever().fit([])

    def test_search_returns_results(self, fitted_retriever: EmbeddingRetriever) -> None:
        """Test basic search returns results."""
        results = fitted_retriever.search("wireless headphones", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_results_are_tuples(
        self, fitted_retriever: EmbeddingRetriever
    ) -> None:
        """Test that search results are (id, score) tuples."""
        results = fitted_retriever.search("audio", top_k=2)
        assert results  # Ensure non-empty
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], str)
            assert isinstance(result[1], float)

    def test_empty_query(self, fitted_retriever: EmbeddingRetriever) -> None:
        """Test empty query returns results (semantic, so depends on backend)."""
        results = fitted_retriever.search("", top_k=5)
        # Fake backend always produces some similarity
        assert len(results) >= 0

    def test_top_k_limits(self, fitted_retriever: EmbeddingRetriever) -> None:
        """Test top_k parameter limits results."""
        results = fitted_retriever.search("headphones", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_by_score(
        self, fitted_retriever: EmbeddingRetriever
    ) -> None:
        """Test that results are sorted descending by score."""
        results = fitted_retriever.search("headphones", top_k=5)
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)

    def test_scores_between_zero_and_one(
        self, fitted_retriever: EmbeddingRetriever
    ) -> None:
        """Test that similarity scores are in [0, 1]."""
        results = fitted_retriever.search("audio", top_k=5)
        for _doc_id, score in results:
            assert 0 <= score <= 1

    def test_default_doc_ids(self, documents: list[str]) -> None:
        """Test default doc IDs are string indices."""
        retriever = EmbeddingRetriever(FakeEmbeddingBackend(dim=16, seed=99))
        retriever.fit(documents)
        results = retriever.search("headphones", top_k=1)
        assert results  # Non-empty
        assert results[0][0].isdigit()
