#!/usr/bin/env python3
"""Run the frozen WANDS test split across retrieval backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from marketlens.catalog import ProductCatalog
from marketlens.evaluation.metrics import (
    compute_all_metrics,
    macro_average,
    recall_at_k,
)
from marketlens.evaluation.wands import (
    WandsProduct,
    WandsQuery,
    get_judged_products,
    get_relevant_products,
    load_products,
    load_qrels,
    load_queries,
)
from marketlens.models import Product
from marketlens.retrieval.embedding import SentenceTransformersBackend
from marketlens.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

WANDS_DIR = Path("data/external/wands")
OUTPUT_DIR = Path("data/evaluation/wands_backends")
MANIFEST_PATH = Path("benchmarks/manifests/wands.json")
MODEL_NAME = "all-MiniLM-L6-v2"
SEED = 42
TOP_K = 10
CANDIDATE_K = 50
STRATEGIES = (
    "popularity",
    "bm25",
    "semantic_memory",
    "semantic_pgvector",
    "hybrid_memory",
    "hybrid_pgvector",
    "quality_pgvector",
)


def split_query_ids(query_ids: list[str], seed: int = SEED) -> dict[str, list[str]]:
    """Return deterministic 60/20/20 query-ID splits."""
    ordered = sorted(query_ids)
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * 0.60)
    validation_end = train_end + int(len(ordered) * 0.20)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def _catalog(products: list[WandsProduct]) -> ProductCatalog:
    return ProductCatalog([
        Product(
            product_id=item.product_id,
            title=item.title,
            brand=None,
            price=None,
            rating=item.rating,
            review_count=item.review_count,
            description=item.description,
            attributes={"product_class": item.product_class},
        )
        for item in products
    ])


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def _relevant_mrr(retrieved: list[str], qrels: dict[str, int], k: int = 10) -> float:
    for rank, product_id in enumerate(retrieved[:k], 1):
        if qrels.get(product_id, 0) >= 1:
            return 1.0 / rank
    return 0.0


def _evaluate(
    name: str,
    queries: list[WandsQuery],
    qrels: dict[str, dict[str, int]],
    search: Callable[[str], list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_results: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures: dict[str, int] = {}
    for index, query in enumerate(queries, 1):
        started = time.perf_counter()
        failure: str | None = None
        try:
            retrieved = search(query.query_text)
            if not retrieved:
                failure = "no_results"
        except Exception as exc:  # preserve failures in final output
            retrieved = []
            failure = type(exc).__name__
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        if failure:
            failures[failure] = failures.get(failure, 0) + 1
        query_qrels = qrels.get(query.query_id, {})
        result = {
            "query_id": query.query_id,
            "retrieved_ids": retrieved,
            "qrels": query_qrels,
            "relevant_ids": get_relevant_products(qrels, query.query_id),
            "judged_ids": get_judged_products(qrels, query.query_id),
        }
        query_results.append(result)
        runs.append({
            "query_id": query.query_id,
            "query": query.query_text,
            "query_class": query.query_class,
            "strategy": name,
            "product_ids": retrieved,
            "latency_ms": round(elapsed_ms, 3),
            "failure": failure,
        })
        if index % 20 == 0:
            logger.info("[%s] %d/%d", name, index, len(queries))

    metrics: dict[str, Any] = dict(compute_all_metrics(query_results))
    # This run intentionally retrieves TOP_K=10. The shared evaluation helper
    # exposes recall_at_50, which would be mislabeled for these truncated runs.
    metrics.pop("recall_at_50", None)
    metrics.update({
        "recall_at_10": macro_average([
            recall_at_k(item["retrieved_ids"], item["relevant_ids"], 10)
            for item in query_results
        ]),
        "mrr_at_10_relevant": macro_average([
            _relevant_mrr(item["retrieved_ids"], item["qrels"], 10)
            for item in query_results
        ]),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "mean": statistics.mean(latencies) if latencies else 0.0,
        },
        "failures": failures,
    })
    return metrics, runs


def _database_url() -> str:
    raw = os.environ.get("MARKETLENS_DATABASE_URL", "")
    if not raw:
        raise RuntimeError("MARKETLENS_DATABASE_URL is required")
    url = make_url(raw)
    if not url.database or "test" not in url.database.lower():
        raise RuntimeError("refusing a database whose name lacks 'test'")
    return raw


def _load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen = manifest["frozen_config"]
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
    actual_hash = hashlib.sha256(canonical.encode()).hexdigest()
    if actual_hash != manifest["config_hash"]:
        raise RuntimeError("WANDS manifest config hash mismatch")
    return manifest


def main() -> None:
    """Evaluate one frozen WANDS split without overwriting prior evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Validation is for smoke checks; test is the one-shot final evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit validation queries for smoke checks; forbidden for the test split.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.split == "test" and args.limit is not None:
        parser.error("--limit is forbidden for the frozen test split")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    # Final evaluation must be reproducible from the already-frozen snapshots.
    # Missing local files are a hard failure, not an implicit network download.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    manifest = _load_manifest()
    frozen = manifest["frozen_config"]
    if (
        frozen["seed"] != SEED
        or frozen["top_k"] != TOP_K
        or frozen["candidate_k"] != CANDIDATE_K
    ):
        raise RuntimeError("script constants disagree with frozen manifest")

    products = load_products(WANDS_DIR / "product.csv")
    all_queries = load_queries(WANDS_DIR / "query.csv")
    splits = split_query_ids([query.query_id for query in all_queries])
    split_hash = hashlib.sha256(
        json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if split_hash != frozen["wands"]["split_hash"]:
        raise RuntimeError("WANDS split hash differs from frozen manifest")
    query_map = {query.query_id: query for query in all_queries}
    selected_ids = splits[args.split]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    selected_queries = [query_map[query_id] for query_id in selected_ids]

    catalog = _catalog(products)
    embedding_backend = SentenceTransformersBackend(MODEL_NAME)
    memory = RetrievalService(
        catalog,
        data_path=WANDS_DIR / "product.csv",
        embedding_backend=embedding_backend,
        semantic_backend="memory",
    ).initialize()
    engine = create_engine(_database_url())
    factory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    pgvector = RetrievalService(
        catalog,
        data_path=WANDS_DIR / "product.csv",
        embedding_backend=embedding_backend,
        semantic_backend="pgvector",
        session_factory=factory,
        embedding_model_name=MODEL_NAME,
    ).initialize()

    popularity_ids = [
        product.product_id
        for product in sorted(
            catalog.get_all_products(),
            key=lambda item: (
                -(item.review_count or 0),
                -(item.rating or 0.0),
                item.product_id,
            ),
        )[:TOP_K]
    ]
    searchers: dict[str, Callable[[str], list[str]]] = {
        "popularity": lambda _query: popularity_ids,
        "bm25": lambda query: [
            item.product_id
            for item in memory.search(query, strategy="bm25", top_k=TOP_K).results
        ],
        "semantic_memory": lambda query: [
            item.product_id
            for item in memory.search(query, strategy="embedding", top_k=TOP_K).results
        ],
        "semantic_pgvector": lambda query: [
            item.product_id
            for item in pgvector.search(query, strategy="embedding", top_k=TOP_K).results
        ],
        "hybrid_memory": lambda query: [
            item.product_id
            for item in memory.search(query, strategy="hybrid", top_k=TOP_K).results
        ],
        "hybrid_pgvector": lambda query: [
            item.product_id
            for item in pgvector.search(query, strategy="hybrid", top_k=TOP_K).results
        ],
        "quality_pgvector": lambda query: [
            item.product_id
            for item in pgvector.search(
                query,
                strategy="rerank",
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
            ).results
        ],
    }

    # Warm with a train query only. Labels are not loaded until after warm-up.
    warm_query = query_map[splits["train"][0]].query_text
    for strategy in STRATEGIES[1:]:
        searchers[strategy](warm_query)

    qrels = load_qrels(WANDS_DIR / "label.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split.json").write_text(
        json.dumps({"seed": SEED, "split_hash": split_hash, "splits": splits}, indent=2),
        encoding="utf-8",
    )
    summary: dict[str, Any] = {}
    all_runs: list[dict[str, Any]] = []
    try:
        for strategy in STRATEGIES:
            logger.info(
                "Evaluating %s on %d frozen %s queries",
                strategy,
                len(selected_queries),
                args.split,
            )
            metrics, runs = _evaluate(
                strategy,
                selected_queries,
                qrels,
                searchers[strategy],
            )
            summary[strategy] = metrics
            all_runs.extend(runs)
            (args.output_dir / f"metrics_{strategy}.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
        with (args.output_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
            for run in all_runs:
                handle.write(json.dumps(run, ensure_ascii=False) + "\n")
        result = {
            "generated_at": datetime.now(UTC).isoformat(),
            "config_hash": manifest["config_hash"],
            "split": args.split,
            "query_count": len(selected_queries),
            "strategies": summary,
            "not_run": {
                "marketlens_agent": "not run — no real LLM was authorized",
                "without_validator": "not run — no real LLM was authorized",
                "always_agent_vs_routed": "not run — no real LLM was authorized",
                "tokens_cost": "not measured — no real LLM was used",
            },
            "not_applicable": {
                "constraint_validity": (
                    "WANDS has no price or brand constraints; no synthetic fields were added"
                ),
                "unsupported_claim_rate": (
                    "retrieval-only strategies generate no natural-language claims"
                ),
            },
            "historical_holdout_limitation": (
                "An earlier evaluation used all 480 WANDS queries before this split was defined; "
                "the frozen test run is not used for retuning, but is not a "
                "historically untouched holdout."
            ),
        }
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
