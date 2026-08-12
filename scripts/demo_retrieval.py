#!/usr/bin/env python3
"""Compare retrieval strategies on product data.

5 query types × 4 strategies. Reports timing (P50/P95) and status.

Usage:
  uv run python scripts/demo_retrieval.py --products data/processed/electronics_2000.json
  uv run python scripts/demo_retrieval.py  # uses fixture if no real data
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from marketlens.catalog import ProductCatalog
from marketlens.retrieval.service import RetrievalService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DEMO_QUERIES: list[dict[str, Any]] = [
    {
        "query": "Sony wireless noise cancelling headphones",
        "category": "Brand/keyword — exact brand + feature keywords",
        "filters": {},
    },
    {
        "query": "comfortable earbuds for running and working out",
        "category": "Semantic — natural language need description",
        "filters": {},
    },
    {
        "query": "bluetooth headphones for music",
        "category": "Budget — needs to stay under price cap",
        "filters": {"max_budget": 50.0},
    },
    {
        "query": "premium office headset with microphone",
        "category": "Rating constraint — needs 4.0+ stars",
        "filters": {"min_rating": 4.0},
    },
    {
        "query": "quantum computing GPU accelerator PCIe card",
        "category": "No-match — this should not exist in Electronics catalog",
        "filters": {},
    },
]

STRATEGIES = ["bm25", "embedding", "hybrid", "rerank"]


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_results(output) -> None:
    """Print retrieval results table."""
    print(f"  Strategy: {output.strategy} | Found: {output.total_found} | Time: {output.elapsed_ms:.1f} ms")
    if output.model_used:
        print(f"  Model: {output.model_used} ({output.embedding_dim}-dim)")
    print(f"  {'Rank':<5} {'Score':<8} {'Product ID':<16} {'Price':<10} {'Rating':<8} Title")
    print(f"  {'-'*5} {'-'*8} {'-'*16} {'-'*10} {'-'*8} {'-'*40}")
    for item in output.results[:5]:
        price_str = f"${item.price:.2f}" if item.price is not None else "—"
        rating_str = f"{item.rating}/5" if item.rating is not None else "—"
        print(f"  {item.rank:<5} {item.final_score:<8.4f} {item.product_id:<16} {price_str:<10} {rating_str:<8} {item.title[:38]}")
    if output.results and output.results[0].reranker_score is not None:
        print(f"  (reranker scores: {' '.join(f'{r.product_id}={r.reranker_score}' for r in output.results[:3])})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo MarketLens retrieval strategies")
    parser.add_argument("--products", type=Path, default=Path("data/processed/electronics_2000.json"), help="Path to product JSON")
    parser.add_argument("--fake", action="store_true", help="Force fake embeddings (skip model download)")
    args = parser.parse_args()

    use_real = args.products.exists() and not args.fake
    if use_real:
        logger.info("Loading %s", args.products)
        catalog = ProductCatalog.from_json(args.products)
        data_path = args.products
        use_fake = False
    else:
        logger.info("Using fixture catalog")
        catalog = ProductCatalog.from_fixture("electronics_sample.json")
        data_path = None
        use_fake = True

    print_header("Initialization")
    logger.info("Building indices and loading embeddings...")
    t0 = time.monotonic()
    service = RetrievalService(catalog, data_path=data_path, use_fake_embeddings=use_fake)
    service.initialize()
    init_time = (time.monotonic() - t0) * 1000
    logger.info("Init complete: %.0f ms", init_time)

    status = service.status()
    print_header("Service Status")
    for k, v in sorted(status.items()):
        print(f"  {k}: {v}")

    # Warm-up query (loads reranker if lazy)
    _ = service.search("warmup", strategy="hybrid", top_k=3)

    print_header(f"5 Queries × 4 Strategies ({status['product_count']} products)")
    all_timings: dict[str, list[float]] = {s: [] for s in STRATEGIES}

    for qi, demo in enumerate(DEMO_QUERIES, 1):
        query = demo["query"]
        filters = demo["filters"]
        print(f"\n{'─' * 80}")
        print(f"  Q{qi}: \"{query}\" [{demo['category']}]")
        if filters:
            print(f"  Filters: {filters}")
        print(f"{'─' * 80}")

        for strategy in STRATEGIES:
            t0 = time.monotonic()
            try:
                output = service.search(query=query, strategy=strategy, top_k=5, candidate_k=30, **filters)
                qt = (time.monotonic() - t0) * 1000
                output.elapsed_ms = qt
                all_timings[strategy].append(qt)
                print_results(output)
            except Exception as e:
                logger.error("  %s ERROR: %s", strategy, e)

    print_header("Timing Summary (P50 / P95, ms)")
    for s in STRATEGIES:
        ts = all_timings[s]
        if ts:
            print(f"  {s:<12}: P50={statistics.median(ts):.1f}  P95={sorted(ts)[int(len(ts)*0.95)]:.1f}  Mean={statistics.mean(ts):.1f}")
    print()
    logger.info("No ground truth — these are relative comparisons, not quality judgments.")


if __name__ == "__main__":
    main()
