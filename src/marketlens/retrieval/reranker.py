"""Reranker interface for post-retrieval relevance scoring."""

import logging
from abc import ABC, abstractmethod

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


class KeywordReranker(Reranker):
    """Simple keyword-overlap reranker for offline testing.

    Computes Jaccard similarity between tokenized query and document.
    No model download required.
    """

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
