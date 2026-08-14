#!/usr/bin/env python3
"""Evaluate the frozen ESCI subset across real Phase 8 retrieval backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from marketlens.catalog import ProductCatalog
from marketlens.evaluation.metrics import ndcg_at_k, recall_at_k
from marketlens.retrieval.embedding import SentenceTransformersBackend
from marketlens.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed/esci")
SUBSET_MANIFEST = Path("data/manifests/esci_subset.json")
RUN_MANIFEST = Path("reports/phase8_esci_run_manifest.json")
OUTPUT_DIR = Path("data/evaluation/phase8_esci")
MODEL_NAME = "all-MiniLM-L6-v2"
SEED = 20_260_814
TOP_K = 10
CANDIDATE_K = 50
GRADE = {"E": 3, "S": 2, "C": 1, "I": 0}
STRATEGIES = (
    "popularity",
    "bm25",
    "semantic_memory",
    "semantic_pgvector",
    "hybrid_memory",
    "hybrid_pgvector",
    "quality_pgvector",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_subset() -> dict[str, Any]:
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    if manifest["selection"]["seed"] != SEED:
        raise RuntimeError("ESCI seed differs from frozen subset manifest")
    for metadata in manifest["files"].values():
        path = Path(metadata["path"])
        if path.stat().st_size != metadata["size_bytes"]:
            raise RuntimeError(f"ESCI derived size mismatch: {path}")
        if _sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"ESCI derived SHA-256 mismatch: {path}")
    return manifest


def _load_run_manifest(required: bool) -> dict[str, Any] | None:
    if not RUN_MANIFEST.is_file():
        if required:
            raise RuntimeError("frozen ESCI run manifest is required for test")
        return None
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    frozen = manifest["frozen_config"]
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
    actual_hash = hashlib.sha256(canonical.encode()).hexdigest()
    if actual_hash != manifest["config_hash"]:
        raise RuntimeError("ESCI run manifest config hash mismatch")
    if (
        frozen["seed"] != SEED
        or frozen["top_k"] != TOP_K
        or frozen["candidate_k"] != CANDIDATE_K
    ):
        raise RuntimeError("script constants disagree with ESCI run manifest")
    return manifest


def _database_url() -> str:
    raw = os.environ.get("MARKETLENS_DATABASE_URL", "")
    if not raw:
        raise RuntimeError("MARKETLENS_DATABASE_URL is required")
    url = make_url(raw)
    if not url.database or "test" not in url.database.lower():
        raise RuntimeError("refusing a database whose name lacks 'test'")
    return raw


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _reciprocal_rank(
    retrieved: list[str],
    qrels: dict[str, int],
    accepted_grades: set[int],
) -> float:
    for rank, product_id in enumerate(retrieved[:TOP_K], 1):
        if qrels.get(product_id, 0) in accepted_grades:
            return 1.0 / rank
    return 0.0


def _evaluate(
    name: str,
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, int]],
    search: Callable[[str], list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    latencies: list[float] = []
    failures: dict[str, int] = {}
    recalls: list[float] = []
    relevant_mrrs: list[float] = []
    exact_mrrs: list[float] = []
    ndcgs: list[float] = []
    judged_ratios: list[float] = []
    runs: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for index, query in enumerate(queries, 1):
        started = time.perf_counter()
        failure: str | None = None
        try:
            retrieved = search(query["query"])
            if not retrieved:
                failure = "no_results"
        except Exception as exc:
            retrieved = []
            failure = type(exc).__name__
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        if failure:
            failures[failure] = failures.get(failure, 0) + 1
        query_qrels = qrels[query["query_id"]]
        relevant = {
            product_id
            for product_id, grade in query_qrels.items()
            if grade >= GRADE["S"]
        }
        recall = recall_at_k(retrieved, relevant, TOP_K)
        relevant_mrr = _reciprocal_rank(
            retrieved,
            query_qrels,
            {GRADE["E"], GRADE["S"]},
        )
        exact_mrr = _reciprocal_rank(retrieved, query_qrels, {GRADE["E"]})
        ndcg = ndcg_at_k(retrieved, query_qrels, TOP_K)
        judged = len(set(retrieved[:TOP_K]) & set(query_qrels)) / TOP_K
        recalls.append(recall)
        relevant_mrrs.append(relevant_mrr)
        exact_mrrs.append(exact_mrr)
        ndcgs.append(ndcg)
        judged_ratios.append(judged)
        runs.append({
            "query_id": query["query_id"],
            "strategy": name,
            "product_ids": retrieved,
            "latency_ms": round(elapsed_ms, 3),
            "failure": failure,
        })
        per_query.append({
            "query_id": query["query_id"],
            "query": query["query"],
            "ndcg_at_10": ndcg,
            "recall_at_10": recall,
            "exact_mrr_at_10": exact_mrr,
            "retrieved_ids": retrieved,
            "judged_count": len(query_qrels),
            "relevant_count": len(relevant),
            "failure": failure,
        })
        if index % 20 == 0:
            logger.info("[%s] %d/%d", name, index, len(queries))
    metrics = {
        "ndcg_at_10": statistics.mean(ndcgs) if ndcgs else 0.0,
        "mrr_at_10_relevant_es": (
            statistics.mean(relevant_mrrs) if relevant_mrrs else 0.0
        ),
        "exact_mrr_at_10": statistics.mean(exact_mrrs) if exact_mrrs else 0.0,
        "recall_at_10_es": statistics.mean(recalls) if recalls else 0.0,
        "top10_judged_ratio": (
            statistics.mean(judged_ratios) if judged_ratios else 0.0
        ),
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "failures": failures,
        "query_count": len(queries),
    }
    return metrics, runs, per_query


def _backend_parity(
    runs: list[dict[str, Any]],
    left: str,
    right: str,
) -> dict[str, float | int]:
    by_key = {(run["strategy"], run["query_id"]): run for run in runs}
    query_ids = sorted({run["query_id"] for run in runs if run["strategy"] == left})
    exact = 0
    overlaps: list[float] = []
    for query_id in query_ids:
        left_ids = by_key[(left, query_id)]["product_ids"]
        right_ids = by_key[(right, query_id)]["product_ids"]
        if left_ids == right_ids:
            exact += 1
        overlaps.append(len(set(left_ids) & set(right_ids)) / TOP_K)
    return {
        "query_count": len(query_ids),
        "exact_ranking_matches": exact,
        "exact_ranking_match_rate": exact / len(query_ids) if query_ids else 0.0,
        "mean_top10_overlap": statistics.mean(overlaps) if overlaps else 0.0,
    }


def main() -> None:
    """Evaluate validation or the one-shot frozen official-test subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.split == "test" and args.limit is not None:
        parser.error("--limit is forbidden for the frozen test split")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.output_dir}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    subset_manifest = _validate_subset()
    run_manifest = _load_run_manifest(required=args.split == "test")
    catalog = ProductCatalog.from_json(DATA_DIR / "catalog.json")
    all_queries = json.loads((DATA_DIR / "queries.json").read_text(encoding="utf-8"))
    selected_queries = [
        query for query in all_queries if query["derived_split"] == args.split
    ]
    if args.limit is not None:
        selected_queries = selected_queries[: args.limit]

    embedding_backend = SentenceTransformersBackend(MODEL_NAME)
    memory = RetrievalService(
        catalog,
        data_path=DATA_DIR / "catalog.json",
        embedding_backend=embedding_backend,
        semantic_backend="memory",
        embedding_model_name=MODEL_NAME,
    ).initialize()
    engine = create_engine(_database_url())
    factory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    pgvector = RetrievalService(
        catalog,
        data_path=DATA_DIR / "catalog.json",
        embedding_backend=embedding_backend,
        semantic_backend="pgvector",
        session_factory=factory,
        embedding_model_name=MODEL_NAME,
    ).initialize()

    popularity_ids = [
        product.product_id
        for product in sorted(catalog.get_all_products(), key=lambda item: item.product_id)[
            :TOP_K
        ]
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

    # Warm only with a train query. The qrels file is not opened until afterward.
    warm_query = next(
        query["query"] for query in all_queries if query["derived_split"] == "train"
    )
    for strategy in STRATEGIES[1:]:
        searchers[strategy](warm_query)

    raw_qrels = json.loads((DATA_DIR / "qrels.json").read_text(encoding="utf-8"))
    qrels = {
        query_id: {product_id: GRADE[label] for product_id, label in values.items()}
        for query_id, values in raw_qrels.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {}
    all_runs: list[dict[str, Any]] = []
    quality_per_query: list[dict[str, Any]] = []
    try:
        for strategy in STRATEGIES:
            logger.info(
                "Evaluating %s on %d frozen %s queries",
                strategy,
                len(selected_queries),
                args.split,
            )
            metrics, runs, per_query = _evaluate(
                strategy,
                selected_queries,
                qrels,
                searchers[strategy],
            )
            summary[strategy] = metrics
            all_runs.extend(runs)
            if strategy == "quality_pgvector":
                quality_per_query = per_query
            (args.output_dir / f"metrics_{strategy}.json").write_text(
                json.dumps(metrics, indent=2) + "\n",
                encoding="utf-8",
            )
        parity = {
            "semantic_memory_vs_pgvector": _backend_parity(
                all_runs,
                "semantic_memory",
                "semantic_pgvector",
            ),
            "hybrid_memory_vs_pgvector": _backend_parity(
                all_runs,
                "hybrid_memory",
                "hybrid_pgvector",
            ),
        }
        worst_cases = sorted(
            quality_per_query,
            key=lambda item: (item["ndcg_at_10"], item["query_id"]),
        )[:5]
        with (args.output_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
            for run in all_runs:
                handle.write(json.dumps(run, ensure_ascii=False) + "\n")
        result = {
            "generated_at": datetime.now(UTC).isoformat(),
            "config_hash": run_manifest["config_hash"] if run_manifest else None,
            "subset_manifest_sha256": _sha256(SUBSET_MANIFEST),
            "dataset": subset_manifest["dataset"],
            "split": args.split,
            "query_count": len(selected_queries),
            "catalog_unique_products": len(catalog),
            "strategies": summary,
            "backend_parity": parity,
            "pgvector": {
                "mode": "exact cosine",
                "approximate": (
                    "not run separately: the repository query's deterministic "
                    "secondary product_id sort uses Seq Scan + Sort even though "
                    "frozen migration 0002 defines an HNSW index"
                ),
            },
            "relevance_mapping": {
                "graded_ndcg": GRADE,
                "recall_and_relevant_mrr": "E or S",
                "exact_mrr": "E only",
            },
            "execution_failures": sum(
                sum(metrics["failures"].values()) for metrics in summary.values()
            ),
            "predefined_lowest_quality_cases": worst_cases,
            "benchmark_claim": subset_manifest["benchmark_claim"],
            "real_llm_used": False,
        }
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
