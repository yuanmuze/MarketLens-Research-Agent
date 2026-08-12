"""Unified retrieval service — single entry point for 4 strategies.

Orchestrates BM25, Embedding, Hybrid RRF, and Cross-Encoder rerank
behind one consistent interface. Handles embedding caching, model
lifecycle, and structured filtering.

Design decisions:
  - RRF (not score addition) — BM25 and embedding scores have
    incomparable scales. RRF fuses by rank, which is scale-free.
  - Two-stage retrieval — BM25/embedding broaden recall, hybrid
    fuses ranks, reranker refines precision on top candidates only.
  - In-memory numpy for 2k products — no vector database needed.
    For >100k, swap to FAISS or pgvector.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from marketlens.catalog import ProductCatalog
from marketlens.retrieval.bm25 import BM25Retriever
from marketlens.retrieval.embedding import (
    EmbeddingBackend,
    EmbeddingRetriever,
    FakeEmbeddingBackend,
    SentenceTransformersBackend,
)
from marketlens.retrieval.reranker import (
    CrossEncoderReranker,
    KeywordReranker,
    Reranker,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")

# Schema version — bump when _build_search_text logic changes
TEXT_SCHEMA_VERSION = "v1"


def _compute_data_hash(data_path: Path) -> str:
    """Compute SHA256 of the data file content for cache fingerprinting.

    Args:
        data_path: Path to the products JSON file.

    Returns:
        First 16 hex chars of the SHA256 hash.
    """
    chunk_size = 1 << 20  # 1 MB chunks
    h = hashlib.sha256()
    with open(data_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class RetrievalOutput:
    """Unified output for any retrieval strategy.

    All four strategies return this exact structure.
    """

    query: str
    strategy: str  # "bm25" | "embedding" | "hybrid" | "rerank"
    total_found: int
    results: list[RetrievalResultItem]
    elapsed_ms: float
    model_used: str = ""
    embedding_dim: int = 0


@dataclass
class RetrievalResultItem:
    """A single result item, identical across strategies."""

    rank: int
    product_id: str
    title: str
    brand: str
    price: float | None
    rating: float | None
    review_count: int | None
    final_score: float
    bm25_score: float | None = None
    embedding_score: float | None = None
    reranker_score: float | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    description: str = ""
    url: str = ""

    @classmethod
    def from_product(
        cls,
        product: dict[str, Any],
        rank: int,
        final_score: float,
        *,
        bm25_score: float | None = None,
        embedding_score: float | None = None,
        reranker_score: float | None = None,
    ) -> RetrievalResultItem:
        """Build from a product dict (catalog or Pydantic model)."""
        return cls(
            rank=rank,
            product_id=str(product.get("product_id", "")),
            title=str(product.get("title", ""))[:200],
            brand=str(product.get("brand") or ""),
            price=product.get("price") if product.get("price") is not None else None,
            rating=product.get("rating") if product.get("rating") is not None else None,
            review_count=product.get("review_count") if product.get("review_count") is not None else None,
            final_score=round(final_score, 4),
            bm25_score=round(bm25_score, 4) if bm25_score is not None else None,
            embedding_score=round(embedding_score, 4) if embedding_score is not None else None,
            reranker_score=round(reranker_score, 4) if reranker_score is not None else None,
            attributes=product.get("attributes", {}),
            description=str(product.get("description") or "")[:500],
            url=str(product.get("url") or ""),
        )


def _build_search_text(product: dict[str, Any]) -> str:
    """Combine product fields into a single searchable text.

    Uses title + brand + description for a richer embedding signal
    than title alone. None values are skipped (not stringified).

    Args:
        product: Product dict.

    Returns:
        Search text string.
    """
    parts: list[str] = []
    for fname in ("title", "brand", "description"):
        val = product.get(fname)
        if val is not None and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def _embedding_cache_path(
    data_path: Path,
    model_name: str,
    product_count: int,
    embedding_dim: int,
) -> Path:
    """Derive a cache path from data content + model + schema.

    Cache is invalidated when any of these change:
      - Data file content (SHA256)
      - Model name
      - Text schema version
      - Product count
      - Embedding dimension

    Args:
        data_path: Path to the product JSON file.
        model_name: Embedding model identifier.
        product_count: Number of products.
        embedding_dim: Embedding vector dimension.

    Returns:
        Cache file path (.npy).
    """
    data_hash = _compute_data_hash(data_path)
    fingerprint = f"{data_hash}:{model_name}:{TEXT_SCHEMA_VERSION}:{product_count}:{embedding_dim}"
    key = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"embeddings_{key}.npy"


def _cache_metadata_path(embedding_path: Path) -> Path:
    """Derive metadata path from embedding cache path."""
    return embedding_path.with_suffix(".json")


class RetrievalService:
    """Unified retrieval service orchestrating all four strategies.

    Lifecycle:
      1. initialize() — build BM25 index, compute/load embedding cache
      2. search(query, strategy) — run one of four strategies
      3. search() is stateless after initialization — safe for concurrent reads

    Embeddings are cached to disk on first computation. Subsequent
    initializations load from cache (validated against data+model hash).
    """

    def __init__(
        self,
        catalog: ProductCatalog,
        *,
        data_path: Path | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        use_fake_embeddings: bool = False,
        reranker: Reranker | None = None,
    ) -> None:
        """Create the retrieval service.

        Args:
            catalog: Product catalog.
            data_path: Path to product JSON (used for cache key derivation).
            embedding_backend: Pre-constructed embedding backend.
                If None, tries real model unless use_fake_embeddings=True.
            use_fake_embeddings: Explicitly use FakeEmbeddingBackend for
                tests/demos. When False and embedding_backend is None,
                a real SentenceTransformer is required.
            reranker: Reranker instance. Defaults to CrossEncoderReranker.
                Use FakeReranker/KeywordReranker for tests.
        """
        self._catalog = catalog
        self._data_path = data_path
        self._products = catalog.get_all_products()
        self._product_dicts = [p.model_dump() for p in self._products]
        self._product_index = {str(p["product_id"]): p for p in self._product_dicts}
        self._search_texts = [_build_search_text(p) for p in self._product_dicts]

        # Backends
        self._bm25: BM25Retriever | None = None
        self._embedding: EmbeddingRetriever | None = None
        self._embedding_backend = embedding_backend
        self._use_fake_embeddings = use_fake_embeddings
        self._reranker: Reranker | None = reranker
        self._is_initialized = False
        self._embedding_cache_hit = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> RetrievalService:
        """Build indices and load/initialize embedding cache.

        Must be called once before any search(). Idempotent.

        Returns:
            Self for chaining.
        """
        if self._is_initialized:
            return self

        t0 = time.monotonic()
        n = len(self._product_dicts)
        logger.info("Initializing RetrievalService with %d products...", n)

        # BM25
        doc_ids = [str(p["product_id"]) for p in self._product_dicts]
        if n > 0:
            self._bm25 = BM25Retriever().fit(self._search_texts, doc_ids)
            logger.info("BM25 index built (%d docs)", n)
        else:
            self._bm25 = BM25Retriever().fit(["empty"], ["__empty__"])
            logger.info("BM25 index empty (0 docs)")

        # Embedding — compute or load from cache
        self._init_embeddings()
        logger.info("Embeddings ready (dim=%d)", self._embedding_backend.dim if self._embedding_backend else 0)

        elapsed = (time.monotonic() - t0) * 1000
        logger.info("RetrievalService initialized in %.0f ms", elapsed)
        self._is_initialized = True
        return self

    def _init_embeddings(self) -> None:
        """Initialize embedding backend and retriever, loading cache if available."""
        if len(self._product_dicts) == 0:
            if self._embedding_backend is None:
                self._embedding_backend = FakeEmbeddingBackend(dim=8, seed=0)
            self._embedding = EmbeddingRetriever(self._embedding_backend).fit(["empty"], ["__empty__"])
            return

        # Select backend
        if self._embedding_backend is None:
            if self._use_fake_embeddings:
                self._embedding_backend = FakeEmbeddingBackend(dim=128, seed=42)
                logger.info("Using fake embedding (explicitly requested)")
            else:
                self._embedding_backend = self._create_real_backend()

        # Try cache for real backends
        if self._data_path and isinstance(self._embedding_backend, SentenceTransformersBackend):
            self._try_load_cache()
            if self._embedding is not None:
                return

        # Compute fresh
        logger.info("Computing embeddings for %d products...", len(self._product_dicts))
        self._embedding = EmbeddingRetriever(self._embedding_backend).fit(
            self._search_texts,
            [str(p["product_id"]) for p in self._product_dicts],
        )

        # Save cache
        if self._data_path and isinstance(self._embedding_backend, SentenceTransformersBackend):
            self._save_cache()

    def _try_load_cache(self) -> None:
        """Try to load embedding cache from disk. Sets self._embedding on success."""
        if self._data_path is None or not isinstance(self._embedding_backend, SentenceTransformersBackend):
            return

        dim = self._embedding_backend.dim
        n = len(self._product_dicts)
        cache_path = _embedding_cache_path(
            self._data_path, self._embedding_backend._model_name, n, dim,
        )
        meta_path = _cache_metadata_path(cache_path)

        if not cache_path.exists() or not meta_path.exists():
            logger.info("No existing cache found at %s", cache_path)
            return

        logger.info("Loading embedding cache from %s", cache_path)
        try:
            cached = np.load(str(cache_path))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            # Validate shape, dtype, count
            errors = []
            if meta.get("count") != n:
                errors.append(f"count {meta.get('count')} != {n}")
            if cached.shape[1] != dim:
                errors.append(f"dim {cached.shape[1]} != {dim}")
            if cached.dtype != np.float32:
                errors.append(f"dtype {cached.dtype} != float32")
            if cached.shape[0] != n:
                errors.append(f"rows {cached.shape[0]} != {n}")

            if errors:
                logger.warning("Cache validation failed: %s. Recomputing.", "; ".join(errors))
                return

            self._embedding = EmbeddingRetriever(self._embedding_backend)
            self._embedding._doc_ids = [str(p["product_id"]) for p in self._product_dicts]
            self._embedding._embeddings = cached
            self._embedding._is_fitted = True
            self._embedding_cache_hit = True
            logger.info("Loaded %d cached embeddings (%d-dim)", len(cached), cached.shape[1])
        except Exception as e:
            logger.warning("Failed to load cache: %s. Recomputing.", e)

    def _save_cache(self) -> None:
        """Save computed embeddings to disk with metadata."""
        if self._data_path is None or self._embedding is None or self._embedding._embeddings is None:
            return
        if not isinstance(self._embedding_backend, SentenceTransformersBackend):
            return

        dim = self._embedding_backend.dim
        n = len(self._product_dicts)
        cache_path = _embedding_cache_path(
            self._data_path, self._embedding_backend._model_name, n, dim,
        )
        meta_path = _cache_metadata_path(cache_path)

        data_hash = _compute_data_hash(self._data_path)
        np.save(str(cache_path), self._embedding._embeddings)
        meta_path.write_text(json.dumps({
            "model_name": self._embedding_backend._model_name,
            "dim": dim,
            "count": n,
            "dtype": "float32",
            "data_sha256": data_hash,
            "text_schema_version": TEXT_SCHEMA_VERSION,
            "created": str(time.time()),
        }, indent=2), encoding="utf-8")
        logger.info("Embedding cache saved to %s", cache_path)

    @staticmethod
    def _create_real_backend() -> EmbeddingBackend:
        """Create a real SentenceTransformers backend.

        Returns:
            SentenceTransformersBackend.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            backend = SentenceTransformersBackend(
                model_name="all-MiniLM-L6-v2",
                batch_size=64,
                normalize=True,
            )
            _ = backend.dim  # Force load
            logger.info("Using real embedding: all-MiniLM-L6-v2 (384-dim)")
            return backend
        except ImportError as e:
            raise RuntimeError(
                "Real embedding requested but sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers, "
                "or pass use_fake_embeddings=True to use fake embeddings."
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"Real embedding requested but model failed to load: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10,
        candidate_k: int = 50,
        *,
        max_budget: float | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
        brand: str | None = None,
    ) -> RetrievalOutput:
        """Run a retrieval query with the specified strategy.

        Args:
            query: Natural language search query.
            strategy: One of "bm25", "embedding", "hybrid", "rerank".
            top_k: Number of results to return.
            candidate_k: Number of candidates for reranker (only for "rerank").
            max_budget: Maximum price (alias for max_price).
            min_price: Minimum price filter.
            max_price: Maximum price filter.
            min_rating: Minimum rating filter.
            brand: Brand filter.

        Returns:
            RetrievalOutput with unified result structure.

        Raises:
            RuntimeError: If service not initialized.
            ValueError: If strategy is invalid.
        """
        if not self._is_initialized:
            raise RuntimeError("RetrievalService not initialized. Call initialize() first.")

        if strategy not in ("bm25", "embedding", "hybrid", "rerank"):
            raise ValueError(f"Unknown strategy: {strategy}. Use bm25, embedding, hybrid, or rerank.")

        # Normalize budget → max_price
        effective_max_price = max_price
        if max_budget is not None:
            effective_max_price = max_budget

        t0 = time.monotonic()

        # Step 1: Collect candidate product IDs from structured filter
        candidate_ids = self._apply_structured_filter(
            min_price=min_price,
            max_price=effective_max_price,
            min_rating=min_rating,
            brand=brand,
        )

        if strategy == "bm25":
            results, scores_bm25, scores_emb = self._search_bm25(query, top_k, candidate_ids)
            strategy_name = "bm25"
        elif strategy == "embedding":
            results, scores_bm25, scores_emb = self._search_embedding(query, top_k, candidate_ids)
            strategy_name = "embedding"
        elif strategy == "hybrid":
            results, scores_bm25, scores_emb = self._search_hybrid(query, top_k, candidate_ids)
            strategy_name = "hybrid"
        else:  # rerank
            results, scores_bm25, scores_emb = self._search_rerank(query, top_k, candidate_k, candidate_ids)
            strategy_name = "rerank"

        elapsed_ms = (time.monotonic() - t0) * 1000

        model_name = ""
        emb_dim = 0
        if self._embedding_backend is not None:
            emb_dim = self._embedding_backend.dim
            if isinstance(self._embedding_backend, SentenceTransformersBackend):
                model_name = self._embedding_backend._model_name

        return RetrievalOutput(
            query=query,
            strategy=strategy_name,
            total_found=len(results),
            results=results,
            elapsed_ms=round(elapsed_ms, 2),
            model_used=model_name,
            embedding_dim=emb_dim,
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _search_bm25(
        self, query: str, top_k: int, candidate_ids: set[str],
    ) -> tuple[list[RetrievalResultItem], dict[str, float], dict[str, float]]:
        """BM25 keyword search."""
        assert self._bm25 is not None
        raw = self._bm25.search(query, top_k * 3)
        bm25_scores: dict[str, float] = {}
        items: list[RetrievalResultItem] = []
        rank = 0
        for pid, score in raw:
            if pid not in candidate_ids:
                continue
            rank += 1
            bm25_scores[pid] = score
            product = self._product_index.get(pid)
            if product is None:
                continue
            items.append(RetrievalResultItem.from_product(
                product, rank, score, bm25_score=score,
            ))
            if rank >= top_k:
                break
        return items, bm25_scores, {}

    def _search_embedding(
        self, query: str, top_k: int, candidate_ids: set[str],
    ) -> tuple[list[RetrievalResultItem], dict[str, float], dict[str, float]]:
        """Embedding semantic search."""
        assert self._embedding is not None
        raw = self._embedding.search(query, top_k * 3)
        emb_scores: dict[str, float] = {}
        items: list[RetrievalResultItem] = []
        rank = 0
        for pid, score in raw:
            if pid not in candidate_ids:
                continue
            rank += 1
            emb_scores[pid] = score
            product = self._product_index.get(pid)
            if product is None:
                continue
            items.append(RetrievalResultItem.from_product(
                product, rank, score, embedding_score=score,
            ))
            if rank >= top_k:
                break
        return items, {}, emb_scores

    def _search_hybrid(
        self, query: str, top_k: int, candidate_ids: set[str],
    ) -> tuple[list[RetrievalResultItem], dict[str, float], dict[str, float]]:
        """Hybrid RRF: fuse BM25 + Embedding ranks."""
        assert self._bm25 is not None
        assert self._embedding is not None

        # Get raw results from both
        bm25_raw = self._bm25.search(query, top_k * 4)
        emb_raw = self._embedding.search(query, top_k * 4)

        # Filter to candidates
        bm25_filtered = {pid: s for pid, s in bm25_raw if pid in candidate_ids}
        emb_filtered = {pid: s for pid, s in emb_raw if pid in candidate_ids}

        # Rank both lists
        bm25_ranked = sorted(bm25_filtered.items(), key=lambda x: x[1], reverse=True)
        emb_ranked = sorted(emb_filtered.items(), key=lambda x: x[1], reverse=True)

        # RRF fusion
        K = 60
        fused: dict[str, float] = {}
        for rank, (pid, _score) in enumerate(bm25_ranked, 1):
            fused[pid] = 1.0 / (K + rank)
        for rank, (pid, _score) in enumerate(emb_ranked, 1):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (K + rank)

        # Sort by fused score
        sorted_pids = sorted(fused.keys(), key=lambda p: fused[p], reverse=True)

        items: list[RetrievalResultItem] = []
        rank = 0
        for pid in sorted_pids:
            rank += 1
            product = self._product_index.get(pid)
            if product is None:
                continue
            items.append(RetrievalResultItem.from_product(
                product, rank, fused[pid],
                bm25_score=bm25_filtered.get(pid),
                embedding_score=emb_filtered.get(pid),
            ))
            if rank >= top_k:
                break

        return items, bm25_filtered, emb_filtered

    def _search_rerank(
        self,
        query: str,
        top_k: int,
        candidate_k: int,
        candidate_ids: set[str],
    ) -> tuple[list[RetrievalResultItem], dict[str, float], dict[str, float]]:
        """Rerank: Hybrid → Cross-Encoder rescore on top candidates."""
        # Stage 1: Hybrid retrieval
        hybrid_results, bm25_scores, emb_scores = self._search_hybrid(
            query, candidate_k, candidate_ids,
        )

        if not hybrid_results:
            return [], {}, {}

        # Stage 2: Reranker rescore
        reranker_scores = self._rerank_candidates(query, hybrid_results)

        # Re-sort by reranker score
        for item in hybrid_results:
            if item.product_id in reranker_scores:
                item.reranker_score = reranker_scores[item.product_id]
                item.final_score = item.reranker_score

        hybrid_results.sort(key=lambda x: x.final_score, reverse=True)

        # Re-rank
        for i, item in enumerate(hybrid_results[:top_k], 1):
            item.rank = i

        return hybrid_results[:top_k], bm25_scores, emb_scores

    def _rerank_candidates(
        self, query: str, candidates: list[RetrievalResultItem],
    ) -> dict[str, float]:
        """Rerank candidates using the instance's reranker backend.

        Args:
            query: The search query.
            candidates: Candidate items from hybrid retrieval.

        Returns:
            Dict of product_id → reranker_score.
        """
        if self._reranker is None:
            # Lazy init: CrossEncoder by default, Keyword for fake mode
            if self._use_fake_embeddings:
                self._reranker = KeywordReranker()
            else:
                try:
                    self._reranker = CrossEncoderReranker()
                except (ImportError, OSError) as e:
                    logger.error("CrossEncoder failed: %s. Install sentence-transformers.", e)
                    self._reranker = KeywordReranker()
                    raise RuntimeError(
                        f"CrossEncoderReranker unavailable: {e}"
                    ) from e

        pairs = [
            (query, _build_search_text(self._product_index.get(c.product_id, {})))
            for c in candidates
        ]
        scores = self._reranker.score(pairs)
        return {
            c.product_id: round(s, 4)
            for c, s in zip(candidates, scores)
        }

    # ------------------------------------------------------------------
    # Structured filtering
    # ------------------------------------------------------------------

    def _apply_structured_filter(
        self,
        *,
        min_price: float | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
        brand: str | None = None,
    ) -> set[str]:
        """Filter products by hard constraints.

        IMPORTANT: Products with missing price are excluded from
        price-filtered queries. They cannot be determined to satisfy
        or violate the constraint, so the safe choice is exclusion.

        Args:
            min_price: Minimum price (exclusive of None-price products).
            max_price: Maximum price (exclusive of None-price products).
            min_rating: Minimum rating.
            brand: Exact brand match (case-insensitive).

        Returns:
            Set of valid product IDs.
        """
        ids = set(self._product_index.keys())

        # Brand filter
        if brand:
            brand_lower = brand.strip().lower()
            ids = {
                pid for pid in ids
                if self._product_index[pid].get("brand", "").lower() == brand_lower
            }

        # Price filters — exclude None-price products
        if min_price is not None:
            ids = {
                pid for pid in ids
                if self._product_index[pid].get("price") is not None
                and self._product_index[pid]["price"] >= min_price
            }
        if max_price is not None:
            ids = {
                pid for pid in ids
                if self._product_index[pid].get("price") is not None
                and self._product_index[pid]["price"] <= max_price
            }

        # Rating filter
        if min_rating is not None:
            ids = {
                pid for pid in ids
                if self._product_index[pid].get("rating") is not None
                and self._product_index[pid]["rating"] >= min_rating
            }

        return ids

    # ------------------------------------------------------------------
    # Properties / Status
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """Whether the service is ready for queries."""
        return self._is_initialized

    @property
    def product_count(self) -> int:
        """Number of products in the catalog."""
        return len(self._product_dicts)

    @property
    def embedding_model_info(self) -> dict[str, Any]:
        """Information about the active embedding backend."""
        if self._embedding_backend is None:
            return {"type": "none", "dim": 0}
        if isinstance(self._embedding_backend, SentenceTransformersBackend):
            return self._embedding_backend.model_info
        return {"type": "fake", "dim": self._embedding_backend.dim}

    def status(self) -> dict[str, Any]:
        """Return comprehensive service status for /health endpoint.

        Returns a dict safe for API exposure (no local paths).
        """
        emb_type = "none"
        emb_model = "none"
        emb_dim = 0
        if self._embedding_backend is not None:
            emb_dim = self._embedding_backend.dim
            if isinstance(self._embedding_backend, SentenceTransformersBackend):
                emb_type = "sentence-transformers"
                emb_model = self._embedding_backend._model_name
            elif isinstance(self._embedding_backend, FakeEmbeddingBackend):
                emb_type = "fake"
                emb_model = "FakeEmbeddingBackend"

        reranker_name = "none (lazy, not yet loaded)"
        if self._reranker is not None:
            reranker_name = self._reranker.backend_name

        return {
            "retrieval_service_ready": self._is_initialized,
            "embedding_backend": emb_type,
            "embedding_model": emb_model,
            "embedding_dim": emb_dim,
            "reranker_backend": reranker_name,
            "product_count": self.product_count,
            "embedding_cache_hit": self._embedding_cache_hit,
        }
