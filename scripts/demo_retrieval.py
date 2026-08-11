#!/usr/bin/env python3
"""Demonstrate the four retrieval strategies on real product data.

Runs 3 query types × 4 strategies, compares results and timing.

Usage:
  uv run python scripts/demo_retrieval.py
  uv run python scripts/demo_retrieval.py --products data/processed/electronics_2000.json
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from marketlens.catalog import ProductCatalog
from marketlens.retrieval.service import RetrievalService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DEMO_QUERIES: list[dict[str, Any]] = [
    # Type 1: Brand/keyword query
    {
        "query": "Sony wireless headphones noise cancelling",
        "category": "Brand/keyword — exact brand + product features",
        "filters": {},  # No constraints
    },
    # Type 2: Natural language semantic query
    {
        "query": "affordable bluetooth earbuds with good sound for working out",
        "category": "Semantic — natural language need description",
        "filters": {"max_budget": 50.0},
    },
    # Type 3: Constrained query with price + rating
    {
        "query": "high quality premium headphones with microphone for office calls",
        "category": "Constrained — features + budget + rating",
        "filters": {"min_rating": 4.0, "max_budget": 200.0},
    },
]

STRATEGIES = ["bm25", "embedding", "hybrid", "rerank"]


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_results(output) -> None:
    """Print retrieval results."""
    print(f"  Strategy: {output.strategy} | Found: {output.total_found} | Time: {output.elapsed_ms:.1f} ms")
    if output.model_used:
        print(f"  Model: {output.model_used} ({output.embedding_dim}-dim)")
    print(f"  {'Rank':<5} {'Score':<8} {'Brand':<20} {'Price':<10} {'Rating':<8} Title")
    print(f"  {'-'*5} {'-'*8} {'-'*20} {'-'*10} {'-'*8} {'-'*40}")
    for item in output.results[:5]:
        price_str = f"${item.price:.2f}" if item.price is not None else "—"
        rating_str = f"{item.rating}/5" if item.rating is not None else "—"
        title = item.title[:45]
        print(f"  {item.rank:<5} {item.final_score:<8.4f} {item.brand:<20} {price_str:<10} {rating_str:<8} {title}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo MarketLens retrieval strategies")
    parser.add_argument(
        "--products", type=Path,
        default=Path("data/processed/electronics_2000.json"),
        help="Path to product JSON",
    )
    args = parser.parse_args()

    if not args.products.exists():
        # Fall back to fixture
        logger.info("Real data not found at %s, using fixture", args.products)
        catalog = ProductCatalog.from_fixture("electronics_sample.json")
        data_path = None
    else:
        logger.info("Loading products from %s", args.products)
        catalog = ProductCatalog.from_json(args.products)
        data_path = args.products

    print_header("Initializing RetrievalService")
    logger.info("This loads BM25 index and computes/loads embedding cache...")
    t0 = time.monotonic()
    service = RetrievalService(catalog, data_path=data_path)
    service.initialize()
    init_time = (time.monotonic() - t0) * 1000
    logger.info(f"First initialization: {init_time:.0f} ms")
    logger.info(f"Embedding backend: {service.embedding_model_info}")

    print_header("Demonstration: 3 Queries × 4 Strategies")

    for qi, demo in enumerate(DEMO_QUERIES, 1):
        query = demo["query"]
        category = demo["category"]
        filters = demo["filters"]

        print(f"\n{'─' * 80}")
        print(f"  Query {qi}: \"{query}\"")
        print(f"  Category: {category}")
        if filters:
            print(f"  Filters: {filters}")
        print(f"{'─' * 80}")

        for strategy in STRATEGIES:
            t0 = time.monotonic()
            try:
                output = service.search(
                    query=query,
                    strategy=strategy,
                    top_k=5,
                    candidate_k=30,
                    **filters,
                )
                query_time = (time.monotonic() - t0) * 1000
                output.elapsed_ms = query_time
                print_results(output)
            except Exception as e:
                logger.error("  %s failed: %s", strategy, e)

    print_header("Summary")
    logger.info("First init (model load + index build): %.0f ms", init_time)
    logger.info("Cached queries run at ~0-2 ms each (in-memory numpy)")
    logger.info("Note: Results vary by strategy. No ground truth exists yet.")
    logger.info("These are relative comparisons, not absolute quality judgments.")


if __name__ == "__main__":
    main()
