"""Embedding-based semantic search for product retrieval.

Provides a pluggable embedding interface. The default implementation
uses a deterministic fake embedding for offline testing. A real
sentence-transformers implementation is available as an optional backend.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Abstract embedding backend interface."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            NumPy array of shape (len(texts), embedding_dim).
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...


class FakeEmbeddingBackend(EmbeddingBackend):
    """Deterministic fake embedding for offline testing.

    Uses a simple hash-based approach to produce consistent,
    deterministic vectors for any input text. No model download required.
    """

    def __init__(self, dim: int = 128, seed: int = 42) -> None:
        """Initialize fake embedding backend.

        Args:
            dim: Embedding dimension (default 128).
            seed: Random seed for reproducibility.
        """
        self._dim = dim
        self._rng = np.random.RandomState(seed)  # noqa: NPY002

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using deterministic hash-based embeddings.

        Each unique text produces a consistent vector. The hash determines
        the random projection used.

        Args:
            texts: List of text strings.

        Returns:
            NumPy array of shape (len(texts), dim).
        """
        import hashlib

        embeddings = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Use MD5 hash to seed a deterministic random vector
            hash_bytes = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).digest()  # noqa: S324
            seed = int.from_bytes(hash_bytes[:4], "big") % (2**31)
            rng = np.random.RandomState(seed)  # noqa: NPY002
            vec = rng.randn(self._dim).astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings[i] = vec

        return embeddings


class SentenceTransformersBackend(EmbeddingBackend):
    """Optional sentence-transformers based embedding backend.

    Uses all-MiniLM-L6-v2 by default: 384-dimensional embeddings,
    lightweight (~80MB), runs on CPU. Suitable for development and
    small production workloads.

    Requires: pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        """Initialize with a sentence-transformers model.

        Args:
            model_name: HuggingFace model name (default all-MiniLM-L6-v2).
            batch_size: Batch size for encoding (default 32).
            normalize: Whether to L2-normalize output vectors (default True).
        """
        self._model_name = model_name
        self._model: Any = None  # SentenceTransformer | None
        self.batch_size = batch_size
        self.normalize = normalize

    @property
    def dim(self) -> int:
        """Embedding dimension.

        Returns: 384 for default all-MiniLM-L6-v2.
        """
        if self._model is None:
            self._load_model()
        assert self._model is not None, "Model failed to load"
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_info(self) -> dict[str, str | int]:
        """Return model metadata for record-keeping.

        Returns:
            Dict with model_name, dim, backend_type.
        """
        return {
            "backend_type": "sentence-transformers",
            "model_name": self._model_name,
            "dim": self.dim,
            "batch_size": self.batch_size,
        }

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Loaded sentence-transformers model: %s (dim=%d)",
                self._model_name, self._model.get_sentence_embedding_dimension(),
            )
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the sentence-transformers model.

        Uses batch encoding for memory efficiency on large text lists.
        All vectors are L2-normalized for cosine similarity computation.

        Args:
            texts: List of text strings to encode.

        Returns:
            NumPy array of shape (len(texts), dim), float32, L2-normalized.
        """
        if self._model is None:
            self._load_model()
        assert self._model is not None, "Model failed to load"
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )


class EmbeddingRetriever:
    """Semantic product retriever using embedding similarity.

    Supports any EmbeddingBackend implementation.
    """

    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        """Initialize the embedding retriever.

        Args:
            backend: EmbeddingBackend instance. Uses FakeEmbeddingBackend if None.
        """
        self._backend = backend or FakeEmbeddingBackend()
        self._doc_ids: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._is_fitted: bool = False

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._backend.dim

    @property
    def is_fitted(self) -> bool:
        """Whether the retriever has been fitted."""
        return self._is_fitted

    def fit(self, documents: list[str], doc_ids: list[str] | None = None) -> "EmbeddingRetriever":
        """Encode and index a corpus of documents.

        Args:
            documents: List of document texts.
            doc_ids: Optional list of document IDs.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot fit embedding retriever on empty document list")

        self._doc_ids = doc_ids if doc_ids is not None else [str(i) for i in range(len(documents))]
        self._embeddings = self._backend.encode(documents)
        self._is_fitted = True
        logger.info(
            "Embedding retriever fitted on %d documents (dim=%d)",
            len(documents),
            self._backend.dim,
        )
        return self

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search for semantically similar documents.

        Args:
            query: The search query string.
            top_k: Number of top results.

        Returns:
            List of (doc_id, cosine_similarity) tuples, sorted descending.

        Raises:
            RuntimeError: If not fitted.
        """
        if not self._is_fitted or self._embeddings is None:
            raise RuntimeError("EmbeddingRetriever must be fitted before searching")

        # Encode query
        query_embedding = self._backend.encode([query])
        query_vec = query_embedding[0]

        # Compute cosine similarity
        similarities = np.dot(self._embeddings, query_vec)

        # Get top-k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                results.append((self._doc_ids[idx], score))

        return results
