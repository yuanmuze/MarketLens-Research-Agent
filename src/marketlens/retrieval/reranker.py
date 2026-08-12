"""Reranker interface for post-retrieval relevance scoring.

Provides four implementations:
- CrossEncoderReranker: real ms-marco-MiniLM-L-6-v2 (production)
- KeywordReranker: Jaccard similarity (lightweight fallback / testing)
- NoOpReranker: pass-through (disabled)
- FakeReranker: deterministic fake scores (unit tests)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract reranker interface for post-retrieval scoring."""

    @abstractmethod
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, document) pairs for relevance.

        Args:
            pairs: List of (query, document_text) tuples.

        Returns:
            List of relevance scores (higher = more relevant).
        """
        ...

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
        return self.__class__.__name__


class CrossEncoderReranker(Reranker):
    """Real Cross-Encoder reranker using sentence-transformers.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB). The model is
    lazily loaded on first score() call and reused across requests.

    Only re-ranks candidates (not the full catalog). Each pair is
    (query, candidate_product_text) scored independently.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        """Initialize the Cross-Encoder reranker (lazy load)."""
        self._model: Any = None  # CrossEncoder | None

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
        return f"CrossEncoder({self.MODEL_NAME})"

    def _ensure_model(self) -> None:
        """Lazy-load the CrossEncoder model.

        Raises:
            ImportError: If sentence-transformers not installed.
            OSError: If model download fails.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.MODEL_NAME)
            logger.info("CrossEncoder loaded: %s", self.MODEL_NAME)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            ) from e

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, document) pairs with Cross-Encoder.

        Args:
            pairs: List of (query, document_text) tuples.

        Returns:
            List of relevance scores (float, higher = more relevant).
        """
        self._ensure_model()
        assert self._model is not None
        # CrossEncoder.predict returns list[float] for regression
        scores = self._model.predict(pairs, show_progress_bar=False)
        if isinstance(scores, list):
            return [round(float(s), 4) for s in scores]
        import numpy as np
        if isinstance(scores, np.ndarray):
            return [round(float(s), 4) for s in scores.flatten().tolist()]
        return [0.0] * len(pairs)


class FakeReranker(Reranker):
    """Deterministic fake reranker for unit tests (no model needed).

    Uses a hash-based score so results are reproducible.
    """

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs deterministically from text hash.

        Args:
            pairs: List of (query, document_text) tuples.

        Returns:
            List of deterministic scores [0, 1].
        """
        import hashlib

        scores: list[float] = []
        for _i, (query, document) in enumerate(pairs):
            h = hashlib.md5(f"{query}|{document}".encode(), usedforsecurity=False).hexdigest()  # noqa: S324
            s = int(h[:8], 16) / 0xFFFFFFFF
            scores.append(round(s, 4))
        return scores

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
        return "FakeReranker"


class KeywordReranker(Reranker):
    """Simple keyword-overlap reranker for offline testing.

    Computes Jaccard similarity between tokenized query and document.
    No model download required. NOT a Cross-Encoder — explicitly
    labeled to avoid confusion.
    """

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
        return "KeywordReranker(Jaccard)"

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs by Jaccard similarity.

        Args:
            pairs: List of (query, document_text) tuples.

        Returns:
            List of Jaccard similarity scores [0, 1].
        """
        import re

        scores = []
        for query, document in pairs:
            query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
            doc_tokens = set(re.findall(r"[a-zA-Z0-9]+", document.lower()))

            if not query_tokens or not doc_tokens:
                scores.append(0.0)
                continue

            intersection = query_tokens & doc_tokens
            union = query_tokens | doc_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
            scores.append(round(jaccard, 4))

        return scores


class NoOpReranker(Reranker):
    """No-op reranker that passes through original ordering.

    Used when reranking is disabled.
    """

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return uniform scores (no reranking).

        Args:
            pairs: List of (query, document_text) tuples.

        Returns:
            List of 0.0 scores (all equal).
        """
        return [0.0] * len(pairs)
