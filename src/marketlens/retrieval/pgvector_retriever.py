"""pgvector-backed semantic retriever (reuses the same embedding backend).

Implements the same interface as EmbeddingRetriever.search() so Hybrid RRF
can combine BM25 + semantic results unchanged. Embeddings are stored in
PostgreSQL/pgvector and queried via cosine similarity.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection

from sqlalchemy.orm import Session

from marketlens.retrieval.embedding import EmbeddingBackend
from marketlens.retrieval.semantic import SemanticBackendStatus

logger = logging.getLogger(__name__)

PGVECTOR_DIMENSION = 384


class PgVectorEmbeddingRetriever:
    """Semantic retriever backed by PostgreSQL pgvector.

    Encodes queries with the SAME EmbeddingBackend used by the in-memory
    retriever (no second semantic definition), then runs a cosine
    similarity top-k query against the `product_embeddings` table.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        session_factory: Callable[[], Session],
        model_name: str,
    ) -> None:
        """Initialize with an embedding backend and session factory.

        Args:
            backend: EmbeddingBackend for encoding query text.
            session_factory: Callable returning a SQLAlchemy session.
            model_name: Embedding model identifier (filters stored vectors).
        """
        if backend.dim != PGVECTOR_DIMENSION:
            raise ValueError(
                f"pgvector requires {PGVECTOR_DIMENSION}-dimensional embeddings, "
                f"got {backend.dim}"
            )
        if not model_name.strip():
            raise ValueError("pgvector model_name must not be empty")
        self._backend = backend
        self._session_factory = session_factory
        self._model_name = model_name

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._backend.dim

    @property
    def model_name(self) -> str:
        """Embedding model identifier."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Storage/search backend name."""
        return "pgvector"

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search for semantically similar products.

        Args:
            query: Query text.
            top_k: Number of results.
            candidate_ids: Optional product IDs allowed in the result set.

        Returns:
            List of (product_id, similarity) tuples, sorted descending.
        """
        # Encode query using the same backend (consistency with in-memory).
        query_vec = self._backend.encode([query])[0].tolist()

        from marketlens.persistence.repositories import ProductEmbeddingRepository

        session = self._session_factory()
        try:
            repo = ProductEmbeddingRepository(session)
            return repo.search(
                query_vec,
                top_k=top_k,
                model_name=self._model_name,
                candidate_ids=candidate_ids,
            )
        finally:
            session.close()

    def status(
        self,
        expected_product_ids: Collection[str] | None = None,
    ) -> SemanticBackendStatus:
        """Check model, dimension, and catalog coverage in PostgreSQL."""
        expected = set(expected_product_ids or ())
        session = self._session_factory()
        try:
            from marketlens.persistence.repositories import ProductEmbeddingRepository

            repo = ProductEmbeddingRepository(session)
            index = repo.index_status(self._model_name, expected)
        finally:
            session.close()

        dimensions = index["dimensions"]
        ready = (
            index["indexed_count"] == index["expected_count"]
            and dimensions == {self.dim}
        )
        if not expected:
            ready = True
        if index["indexed_count"] != index["expected_count"]:
            detail = (
                f"missing {index['expected_count'] - index['indexed_count']} "
                "catalog embeddings"
            )
        elif dimensions != {self.dim} and expected:
            detail = f"stored dimensions {sorted(dimensions)} do not match {self.dim}"
        else:
            detail = ""
        return SemanticBackendStatus(
            backend=self.backend_name,
            model=self.model_name,
            dimension=self.dim,
            ready=ready,
            indexed_count=index["indexed_count"],
            expected_count=index["expected_count"],
            detail=detail,
        )
