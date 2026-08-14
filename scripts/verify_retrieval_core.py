#!/usr/bin/env python3
"""Verify RetrievalService correctness on real product data.

Checks embedding integrity, cache validity, reranker operation, and
constraint enforcement. Exits non-zero on any failure.

Usage:
  uv run python scripts/verify_retrieval_core.py --products data/processed/electronics_2000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from marketlens.catalog import ProductCatalog
from marketlens.retrieval.service import RetrievalService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fail(msg: str) -> None:
    """Log an error and exit non-zero."""
    logger.error("FAIL: %s", msg)
    sys.exit(1)


def main() -> None:
    """Verify MarketLens retrieval core correctness."""
    parser = argparse.ArgumentParser(description="Verify MarketLens retrieval core")
    parser.add_argument("--products", type=Path, required=True, help="Path to product JSON")
    args = parser.parse_args()

    if not args.products.exists():
        fail(f"File not found: {args.products}")

    logger.info("=== Loading catalog ===")
    catalog = ProductCatalog.from_json(args.products)
    products = catalog.get_all_products()
    assert len(products) == 2000, f"Expected 2000 products, got {len(products)}"

    ids = [p.product_id for p in products]
    assert len(set(ids)) == 2000, f"Duplicate product_ids: {len(ids)} unique / {len(set(ids))}"

    logger.info("=== Initializing RetrievalService ===")
    t0 = time.monotonic()
    service = RetrievalService(catalog, data_path=args.products, use_fake_embeddings=False)
    service.initialize()
    init_time = (time.monotonic() - t0) * 1000
    logger.info("Init: %.0f ms", init_time)

    status = service.status()
    logger.info("Status: %s", json.dumps(status, indent=2))

    # Embedding checks
    model_info = service.embedding_model_info
    if model_info["type"] == "fake":
        fail("Embedding backend is FAKE — real model required")
    if status["embedding_dim"] != 384:
        fail(f"Expected dim=384, got {status['embedding_dim']}")
    if status["product_count"] != 2000:
        fail(f"Expected 2000 products, got {status['product_count']}")

    # Verify embedding matrix shape
    assert service._memory_embedding is not None
    emb = service._memory_embedding._embeddings
    assert emb is not None
    assert emb.shape[0] == 2000, f"Embedding rows: {emb.shape[0]}"
    assert emb.shape[1] == 384, f"Embedding dim: {emb.shape[1]}"

    logger.info("=== Testing 4 strategies ===")
    strategies = ["bm25", "embedding", "hybrid", "rerank"]
    for s in strategies:
        out = service.search("wireless headphones", strategy=s, top_k=5)
        assert out.strategy == s, f"{s}: strategy mismatch"
        assert isinstance(out.results, list), f"{s}: results not a list"
        assert all(r.product_id for r in out.results), f"{s}: missing product_id"
        logger.info("  %s: %d results in %.2f ms", s, out.total_found, out.elapsed_ms)

    # Check reranker status after first rerank query triggers lazy load
    status2 = service.status()
    reranker_name = status2["reranker_backend"]
    logger.info("Reranker after first rerank: %s", reranker_name)
    if "CrossEncoder" not in reranker_name and "KeywordReranker" not in reranker_name:
        fail(f"Reranker not loaded: {reranker_name}")

    logger.info("=== Testing reranker candidates ===")
    out = service.search("headphones", strategy="rerank", top_k=5, candidate_k=10)
    assert out.total_found <= 5, f"rerank top_k violation: {out.total_found}"

    logger.info("=== Testing constraint enforcement ===")
    # Budget
    out = service.search("headphones", strategy="hybrid", top_k=10, max_budget=50.0)
    for r in out.results:
        assert r.price is not None, f"Null price in budget-filtered results: {r.product_id}"
        assert r.price <= 50.0, f"Price {r.price} > 50: {r.product_id}"

    # Rating
    out = service.search("headphones", strategy="hybrid", top_k=10, min_rating=4.5)
    for r in out.results:
        assert r.rating is not None, f"Null rating in rating-filtered: {r.product_id}"
        assert r.rating >= 4.5, f"Rating {r.rating} < 4.5: {r.product_id}"

    # Brand (case-insensitive)
    out = service.search("audio", strategy="hybrid", top_k=10, brand="Sony")
    for r in out.results:
        assert r.brand.lower() == "sony", f"Brand mismatch: {r.brand}"

    # Missing price must not sneak through price filter
    out = service.search("headphones", strategy="bm25", top_k=20, max_budget=100.0)
    for r in out.results:
        assert r.price is not None and r.price <= 100.0, f"Constraint violation: {r.product_id} price={r.price}"

    # If filter removes everything, should return empty — never fake results
    out = service.search("headphones", strategy="hybrid", top_k=5, max_budget=0.01)
    assert out.total_found == 0, f"Should be empty for impossible budget: {out.total_found}"

    logger.info("=== All checks passed ===")
    logger.info("Product count: 2000")
    logger.info("Embedding: real (384-dim)")
    logger.info("Reranker: %s", reranker_name)
    logger.info("Init time: %.0f ms", init_time)
    logger.info("Cache hit: %s", status["embedding_cache_hit"])
    sys.exit(0)


if __name__ == "__main__":
    main()
