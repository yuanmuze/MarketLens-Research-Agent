"""pgvector-backed semantic retriever (reuses the same embedding backend).

Implements the same interface as EmbeddingRetriever.search() so Hybrid RRF
can combine BM25 + semantic results unchanged. Embeddings are stored in
PostgreSQL/pgvector and queried via cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Callable

from marketlens.retrieval.embedding import EmbeddingBackend

logger = logging.getLogger(__name__)


class PgVectorEmbeddingRetriever:
    """Semantic retriever backed by PostgreSQL pgvector.

    Encodes queries with the SAME EmbeddingBackend used by the in-memory
    retriever (no second semantic definition), then runs a cosine
    similarity top-k query against the `product_embeddings` table.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        session_factory: Callable,
        model_name: str,
    ) -> None:
        """Initialize with an embedding backend and session factory.

        Args:
            backend: EmbeddingBackend for encoding query text.
            session_factory: Callable returning a SQLAlchemy session.
            model_name: Embedding model identifier (filters stored vectors).
        """
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

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search for semantically similar products.

        Args:
            query: Query text.
            top_k: Number of results.

        Returns:
            List of (product_id, similarity) tuples, sorted descending.
        """
        # Encode query using the same backend (consistency with in-memory).
        query_vec = self._backend.encode([query])[0].tolist()

        from marketlens.persistence.repositories import ProductEmbeddingRepository

        session = self._session_factory()
        try:
            repo = ProductEmbeddingRepository(session)
            return repo.search(query_vec, top_k=top_k, model_name=self._model_name)
        finally:
            session.close()
