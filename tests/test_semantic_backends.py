"""Tests for the active memory/pgvector semantic backend contract."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from marketlens.catalog import ProductCatalog
from marketlens.models import Product
from marketlens.retrieval.embedding import FakeEmbeddingBackend
from marketlens.retrieval.pgvector_retriever import PgVectorEmbeddingRetriever
from marketlens.retrieval.reranker import KeywordReranker
from marketlens.retrieval.semantic import (
    SemanticBackendStatus,
    SemanticBackendUnavailableError,
    SemanticRetriever,
)
from marketlens.retrieval.service import RetrievalService


class StubPgVectorRetriever:
    """Deterministic protocol implementation for service routing tests."""

    backend_name = "pgvector"
    model_name = "test-model"
    dim = 384

    def __init__(
        self,
        results: list[tuple[str, float]],
        *,
        ready: bool = True,
    ) -> None:
        self.results = results
        self.ready = ready
        self.search_calls: list[tuple[str, int, set[str] | None]] = []

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        allowed = set(candidate_ids) if candidate_ids is not None else None
        self.search_calls.append((query, top_k, allowed))
        results = self.results
        if allowed is not None:
            results = [(pid, score) for pid, score in results if pid in allowed]
        return results[:top_k]

    def status(
        self,
        expected_product_ids: Collection[str] | None = None,
    ) -> SemanticBackendStatus:
        expected = set(expected_product_ids or ())
        indexed_count = len(expected) if self.ready else max(0, len(expected) - 1)
        return SemanticBackendStatus(
            backend=self.backend_name,
            model=self.model_name,
            dimension=self.dim,
            ready=self.ready,
            indexed_count=indexed_count,
            expected_count=len(expected),
            detail="" if self.ready else "missing 1 catalog embedding",
        )


@pytest.fixture
def catalog() -> ProductCatalog:
    return ProductCatalog([
        Product(
            product_id="P1",
            title="Alpha wireless headphones",
            brand="Alpha",
            price=100.0,
            rating=4.5,
            review_count=10,
        ),
        Product(
            product_id="P2",
            title="Beta desktop speaker",
            brand="Beta",
            price=200.0,
            rating=4.0,
            review_count=20,
        ),
        Product(
            product_id="P3",
            title="Gamma budget headphones",
            brand="Gamma",
            price=50.0,
            rating=3.5,
            review_count=30,
        ),
    ])


def test_protocol_accepts_memory_and_pgvector_stub(catalog: ProductCatalog) -> None:
    memory = RetrievalService(
        catalog,
        embedding_backend=FakeEmbeddingBackend(dim=32),
    ).initialize()
    stub = StubPgVectorRetriever([("P2", 0.9)])
    assert isinstance(memory._semantic_retriever, SemanticRetriever)
    assert isinstance(stub, SemanticRetriever)


def test_invalid_semantic_backend_fails_at_construction(catalog: ProductCatalog) -> None:
    with pytest.raises(ValueError, match="semantic_backend"):
        RetrievalService(catalog, semantic_backend="invalid")  # type: ignore[arg-type]


def test_injected_backend_must_match_selector(catalog: ProductCatalog) -> None:
    stub = StubPgVectorRetriever([("P2", 0.9)])
    service = RetrievalService(catalog, semantic_retriever=stub)
    with pytest.raises(ValueError, match="does not match"):
        service.initialize()


def test_pgvector_requires_384_dimensions() -> None:
    with pytest.raises(ValueError, match="384-dimensional"):
        PgVectorEmbeddingRetriever(
            FakeEmbeddingBackend(dim=128),
            lambda: None,  # type: ignore[arg-type,return-value]
            "test-model",
        )


def test_missing_pgvector_index_fails_without_memory_fallback(
    catalog: ProductCatalog,
) -> None:
    stub = StubPgVectorRetriever([], ready=False)
    service = RetrievalService(
        catalog,
        semantic_backend="pgvector",
        semantic_retriever=stub,
    )
    with pytest.raises(SemanticBackendUnavailableError, match="missing 1"):
        service.initialize()
    assert service._memory_embedding is None


def test_pgvector_embedding_route_and_structured_filter(
    catalog: ProductCatalog,
) -> None:
    stub = StubPgVectorRetriever([("P2", 0.9), ("P1", 0.8), ("P3", 0.7)])
    service = RetrievalService(
        catalog,
        semantic_backend="pgvector",
        semantic_retriever=stub,
    ).initialize()

    output = service.search(
        "audio",
        strategy="embedding",
        top_k=1,
        max_price=120.0,
    )

    assert output.semantic_backend == "pgvector"
    assert [item.product_id for item in output.results] == ["P1"]
    assert stub.search_calls[-1][2] == {"P1", "P3"}
    assert service._memory_embedding is None


def test_hybrid_uses_pgvector_results_in_active_rrf(catalog: ProductCatalog) -> None:
    stub = StubPgVectorRetriever([("P2", 0.9), ("P1", 0.8)])
    service = RetrievalService(
        catalog,
        semantic_backend="pgvector",
        semantic_retriever=stub,
    ).initialize()

    output = service.search("headphones", strategy="hybrid", top_k=3)

    assert output.results[0].product_id == "P1"
    assert output.results[0].bm25_score is not None
    assert output.results[0].embedding_score == 0.8
    assert any(item.product_id == "P2" for item in output.results)


def test_pgvector_hybrid_can_feed_explicit_reranker(catalog: ProductCatalog) -> None:
    stub = StubPgVectorRetriever([("P2", 0.9), ("P1", 0.8)])
    service = RetrievalService(
        catalog,
        semantic_backend="pgvector",
        semantic_retriever=stub,
        reranker=KeywordReranker(),
    ).initialize()

    output = service.search("headphones", strategy="rerank", top_k=2)

    assert output.results[0].product_id == "P1"
    assert output.results[0].reranker_score is not None


def test_status_reports_selected_backend_model_dimension_and_coverage(
    catalog: ProductCatalog,
) -> None:
    stub = StubPgVectorRetriever([("P2", 0.9)])
    service = RetrievalService(
        catalog,
        semantic_backend="pgvector",
        semantic_retriever=stub,
    ).initialize()

    status = service.status()
    assert status["semantic_backend"] == "pgvector"
    assert status["embedding_model"] == "test-model"
    assert status["embedding_dim"] == 384
    assert status["semantic_index_ready"] is True
    assert status["semantic_indexed_count"] == 3
