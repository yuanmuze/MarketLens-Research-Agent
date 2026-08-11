"""BM25 keyword search for product retrieval.

Uses rank-bm25 (Okapi BM25) for keyword-based retrieval without
external API dependencies.
"""

import logging
import math

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 (Okapi BM25) keyword-based product retriever.

    Provides efficient keyword search over product text without
    requiring external search services or API keys.

    The implementation follows the standard Okapi BM25 formula:
    score(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        """Initialize the BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (default 1.5).
            b: Length normalization parameter (default 0.75).
            epsilon: IDF smoothing parameter (default 0.25).
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Internal state
        self._corpus: list[str] = []
        self._doc_ids: list[str] = []
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._doc_freqs: list[dict[str, int]] = []
        self._num_docs: int = 0
        self._is_fitted: bool = False

    def fit(self, documents: list[str], doc_ids: list[str] | None = None) -> "BM25Retriever":
        """Fit the BM25 model on a corpus of documents.

        Args:
            documents: List of document texts.
            doc_ids: Optional list of document IDs (uses index if not provided).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If documents list is empty or lengths don't match.
        """
        if not documents:
            raise ValueError("Cannot fit BM25 on empty document list")

        if doc_ids is not None and len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) must match documents length ({len(documents)})"
            )

        self._corpus = documents
        self._doc_ids = doc_ids if doc_ids is not None else [str(i) for i in range(len(documents))]
        self._num_docs = len(documents)

        # Tokenize and compute document frequencies
        tokenized = [self._tokenize(doc) for doc in documents]
        self._doc_len = [len(tokens) for tokens in tokenized]
        self._avgdl = sum(self._doc_len) / self._num_docs if self._num_docs > 0 else 0.0

        # Compute document frequencies
        self._doc_freqs = []
        df: dict[str, int] = {}
        for tokens in tokenized:
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self._doc_freqs.append(freq)
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        # Compute IDF
        self._idf = {}
        for term, freq in df.items():
            idf = math.log((self._num_docs - freq + 0.5) / (freq + 0.5) + 1.0) + self.epsilon
            self._idf[term] = max(idf, self.epsilon)

        self._is_fitted = True
        logger.info("BM25 fitted on %d documents (avgdl=%.1f)", self._num_docs, self._avgdl)
        return self

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Search for documents matching the query.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.

        Raises:
            RuntimeError: If the model hasn't been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("BM25Retriever must be fitted before searching")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i in range(self._num_docs):
            score = self._score_doc(query_tokens, i)
            if score > 0:
                scores.append((self._doc_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        """Score a single document against query tokens.

        Args:
            query_tokens: Tokenized query.
            doc_idx: Document index.

        Returns:
            BM25 score (0 if no match).
        """
        score = 0.0
        doc_len = self._doc_len[doc_idx]
        doc_freq = self._doc_freqs[doc_idx]

        for token in query_tokens:
            if token not in self._idf:
                continue
            tf = doc_freq.get(token, 0)
            if tf == 0:
                continue

            idf = self._idf[token]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
            score += idf * numerator / denominator

        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric tokens.

        Args:
            text: Input text string.

        Returns:
            List of lowercase tokens.
        """
        import re

        # Simple tokenization: lowercase, split on non-alphanumeric
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text.lower())]

    @property
    def is_fitted(self) -> bool:
        """Whether the retriever has been fitted."""
        return self._is_fitted
