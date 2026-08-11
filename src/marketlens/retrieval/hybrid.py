"""Reciprocal Rank Fusion (RRF) hybrid retrieval combining BM25 and embeddings."""

import logging

from marketlens.catalog import ProductCatalog
from marketlens.models import SearchQuery, SearchResult, UserConstraints
from marketlens.retrieval.bm25 import BM25Retriever
from marketlens.retrieval.embedding import EmbeddingRetriever
from marketlens.retrieval.reranker import NoOpReranker, Reranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid product retriever combining BM25 + embedding via RRF.

    Provides:
    - Reciprocal Rank Fusion for combining BM25 and embedding results
    - Hard constraint filtering (budget, brand, category, attributes)
    - Configurable weights for BM25 vs embedding
    - Optional reranker pass
    """

    def __init__(
        self,
        catalog: ProductCatalog,
        bm25: BM25Retriever | None = None,
        embedding: EmbeddingRetriever | None = None,
        reranker: Reranker | None = None,
        bm25_weight: float = 0.5,
        embedding_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            catalog: The product catalog to search.
            bm25: Pre-fitted BM25Retriever (auto-created if None).
            embedding: Pre-fitted EmbeddingRetriever (auto-created if None).
            reranker: Optional reranker (uses NoOpReranker if None).
            bm25_weight: Weight for BM25 in RRF (default 0.5).
            embedding_weight: Weight for embedding in RRF (default 0.5).
            rrf_k: RRF smoothing constant (default 60).
        """
        self.catalog = catalog
        self.reranker = reranker or NoOpReranker()
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.rrf_k = rrf_k
        self._bm25 = bm25
        self._embedding = embedding
        self._is_fitted = False

    def fit(self) -> "HybridRetriever":
        """Fit all retrieval components on the catalog.

        Returns:
            Self for method chaining.
        """
        products = self.catalog.get_all_products()
        if not products:
            raise ValueError("Cannot fit retriever on empty catalog")

        doc_ids = [p.product_id for p in products]
        texts = self.catalog.get_search_texts()

        # Fit BM25
        if self._bm25 is None:
            self._bm25 = BM25Retriever().fit(texts, doc_ids)

        # Fit embedding
        if self._embedding is None:
            self._embedding = EmbeddingRetriever().fit(texts, doc_ids)

        self._is_fitted = True
        logger.info("HybridRetriever fitted on %d products", len(products))
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether the retriever has been fitted."""
        return self._is_fitted

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute hybrid search with optional filtering and reranking.

        Args:
            query: The structured search query.

        Returns:
            List of SearchResults, sorted by relevance.

        Raises:
            RuntimeError: If not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("HybridRetriever must be fitted before searching")

        # Apply hard constraint filtering first (pre-filter)
        filters = query.filters
        candidate_ids = self._apply_hard_filters(filters)

        # Step 1: BM25 retrieval
        bm25_results: dict[str, float] = {}
        if query.use_bm25 and self._bm25 is not None:
            raw_bm25 = self._bm25.search(query.text, top_k=query.top_k * 2)
            bm25_results = {doc_id: score for doc_id, score in raw_bm25}
            # Filter to candidates
            bm25_results = {k: v for k, v in bm25_results.items() if k in candidate_ids}

        # Step 2: Embedding retrieval
        embedding_results: dict[str, float] = {}
        if query.use_embedding and self._embedding is not None:
            raw_emb = self._embedding.search(query.text, top_k=query.top_k * 2)
            embedding_results = {doc_id: score for doc_id, score in raw_emb}
            # Filter to candidates
            embedding_results = {k: v for k, v in embedding_results.items() if k in candidate_ids}

        # Step 3: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            bm25_results,
            embedding_results,
            k=self.rrf_k,
            w_bm25=self.bm25_weight,
            w_emb=self.embedding_weight,
        )

        # Sort by fused score
        sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)

        # Step 4: Optional reranker
        if query.use_reranker:
            sorted_ids, reranker_scores = self._apply_reranker(query.text, sorted_ids)
        else:
            reranker_scores = {}

        # Step 5: Build SearchResult objects
        results = []
        for rank, pid in enumerate(sorted_ids[:query.top_k], start=1):
            product = self.catalog.get_product(pid)
            if product is None:
                continue

            result = SearchResult(
                product=product,
                score=fused.get(pid, 0.0),
                bm25_score=bm25_results.get(pid),
                embedding_score=embedding_results.get(pid),
                reranker_score=reranker_scores.get(pid),
                rank=rank,
                source=self._determine_source(
                    pid, bm25_results, embedding_results, reranker_scores
                ),
            )
            results.append(result)

        return results

    def _apply_hard_filters(self, filters: UserConstraints | None) -> set[str]:
        """Apply hard constraint filters to get candidate product IDs.

        Args:
            filters: User constraints. If None, returns all product IDs.

        Returns:
            Set of candidate product IDs.
        """
        if filters is None:
            return set(self.catalog.get_product_ids())

        filtered = self.catalog.filter_by_constraints(
            max_budget=filters.max_budget,
            min_budget=filters.min_budget,
            brands=filters.preferred_brands if filters.preferred_brands else None,
            excluded_brands=filters.excluded_brands if filters.excluded_brands else None,
            categories=filters.categories if filters.categories else None,
            min_rating=filters.min_rating,
            min_review_count=filters.min_review_count,
            excluded_product_ids=filters.excluded_product_ids if filters.excluded_product_ids else None,
        )
        return set(filtered)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: dict[str, float],
        embedding_results: dict[str, float],
        k: int = 60,
        w_bm25: float = 0.5,
        w_emb: float = 0.5,
    ) -> dict[str, float]:
        """Fuse BM25 and embedding results using weighted RRF.

        Args:
            bm25_results: {doc_id: bm25_score} mapping.
            embedding_results: {doc_id: embedding_score} mapping.
            k: RRF smoothing constant.
            w_bm25: BM25 weight.
            w_emb: Embedding weight.

        Returns:
            {doc_id: fused_rrf_score} mapping.
        """
        fused: dict[str, float] = {}

        # Rank BM25 results by score
        bm25_ranked = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _score) in enumerate(bm25_ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w_bm25 / (k + rank)

        # Rank embedding results by score
        emb_ranked = sorted(embedding_results.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _score) in enumerate(emb_ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w_emb / (k + rank)

        return fused

    def _apply_reranker(
        self, query: str, product_ids: list[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Apply reranker to reorder results.

        Args:
            query: The search query.
            product_ids: List of product IDs in current order.

        Returns:
            Tuple of (reordered_product_ids, {doc_id: reranker_score}).
        """
        texts = []
        for pid in product_ids:
            product = self.catalog.get_product(pid)
            if product:
                texts.append(product.to_search_text())
            else:
                texts.append("")

        pairs = [(query, text) for text in texts]
        scores = self.reranker.score(pairs)

        # Pair ids with scores and re-sort
        scored = list(zip(product_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        reordered = [pid for pid, _ in scored]
        score_map = {pid: score for pid, score in scored}
        return reordered, score_map

    @staticmethod
    def _determine_source(
        pid: str,
        bm25: dict[str, float],
        embedding: dict[str, float],
        reranker: dict[str, float],
    ) -> str:
        """Determine the primary source method for a result.

        Args:
            pid: Product ID.
            bm25: BM25 score map.
            embedding: Embedding score map.
            reranker: Reranker score map.

        Returns:
            Source method string.
        """
        if reranker and pid in reranker:
            return "reranked"
        if pid in bm25 and pid in embedding:
            return "hybrid"
        if pid in bm25:
            return "bm25"
        if pid in embedding:
            return "embedding"
        return "unknown"
