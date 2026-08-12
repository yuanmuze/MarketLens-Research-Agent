#!/usr/bin/env python3
"""Evaluate four retrieval strategies on the WANDS benchmark.

Runs BM25, Embedding, Hybrid RRF, and Cross-Encoder Rerank on all
WANDS queries against the full product corpus. Outputs per-query
results (JSONL), aggregated metrics (JSON), and metadata.

Fixed parameters:
  top_k=10, candidate_k=50, RRF k=60, seed=42

Usage:
  uv run python scripts/evaluate_wands.py
  uv run python scripts/evaluate_wands.py --resume   # skip completed queries
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from marketlens.catalog import ProductCatalog
from marketlens.evaluation.metrics import compute_all_metrics
from marketlens.evaluation.wands import (
    WandsProduct,
    get_judged_products,
    get_relevant_products,
    load_products,
    load_qrels,
    load_queries,
)
from marketlens.models import Product
from marketlens.retrieval.service import RetrievalService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WANDS_DIR = Path("data/external/wands")
OUTPUT_DIR = Path("data/evaluation/wands")
STRATEGIES = ["bm25", "embedding", "hybrid", "rerank"]
TOP_K = 10
CANDIDATE_K = 50
SEED = 42


def build_catalog(products: list[WandsProduct]) -> ProductCatalog:
    """Build a ProductCatalog from WANDS products.

    WANDS has no price/brand. The catalog populates only what's available.
    The resulting catalog is fed to RetrievalService — it never sees qrels.
    """
    catalog_products = []
    for wp in products:
        catalog_products.append(Product(
            product_id=wp.product_id,
            title=wp.title,
            brand=None,
            price=None,
            rating=wp.rating,
            review_count=wp.review_count,
            description=wp.description,
            attributes={"product_class": wp.product_class},
        ))
    return ProductCatalog(catalog_products)


def evaluate_strategy(
    service: RetrievalService,
    strategy: str,
    queries: list,
    qrels: dict[str, dict[str, int]],
    output_path: Path,
    resume: bool = False,
) -> list[dict]:
    """Run one strategy on all queries, writing results as JSONL.

    Args:
        service: Initialized RetrievalService.
        strategy: Strategy name.
        queries: WANDS query list.
        qrels: Query relevance labels.
        output_path: Path to write JSONL runs file.
        resume: Skip queries already in output_path.

    Returns:
        List of per-query result dicts for metric computation.
    """
    # Check resume
    completed_ids: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["query_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        logger.info("Resume: %d queries already completed", len(completed_ids))

    mode = "a" if resume else "w"
    fh = open(output_path, mode, encoding="utf-8")  # noqa: SIM115

    query_results: list[dict] = []
    latencies: list[float] = []
    total = len(queries)

    for i, q in enumerate(queries):
        qid = q.query_id

        # Warm-up: skip first query timing (cold reranker load for rerank strategy)
        is_warm = i > 0 or (strategy != "rerank")

        if qid in completed_ids:
            continue

        t0 = time.monotonic()
        try:
            output = service.search(
                query=q.query_text,
                strategy=strategy,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
            )
        except Exception as e:
            logger.error("Query %s failed: %s", qid, e)
            output = None

        elapsed = (time.monotonic() - t0) * 1000
        if is_warm:
            latencies.append(elapsed)

        # Build per-result lines
        retrieved_ids = []
        if output and output.results:
            for item in output.results:
                judged = 0
                rel_grade = 0
                if qid in qrels and item.product_id in qrels[qid]:
                    judged = 1
                    rel_grade = qrels[qid][item.product_id]

                line = {
                    "query_id": qid,
                    "query": q.query_text[:200],
                    "query_class": q.query_class,
                    "strategy": strategy,
                    "rank": item.rank,
                    "product_id": item.product_id,
                    "retrieval_score": item.final_score,
                    "rerank_score": item.reranker_score,
                    "latency_ms": round(elapsed, 2),
                    "judged": judged,
                    "relevance": rel_grade,
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                retrieved_ids.append(item.product_id)
        else:
            # Empty result → write one row
            fh.write(json.dumps({
                "query_id": qid,
                "query": q.query_text[:200],
                "query_class": q.query_class,
                "strategy": strategy,
                "rank": 0,
                "product_id": "",
                "retrieval_score": 0.0,
                "rerank_score": None,
                "latency_ms": round(elapsed, 2),
                "judged": 0,
                "relevance": 0,
            }, ensure_ascii=False) + "\n")

        fh.flush()

        # Per-query metric inputs
        judged_ids = get_judged_products(qrels, qid)
        relevant_ids = get_relevant_products(qrels, qid)
        query_qrels = qrels.get(qid, {})

        query_results.append({
            "query_id": qid,
            "retrieved_ids": retrieved_ids,
            "qrels": query_qrels,
            "relevant_ids": relevant_ids,
            "judged_ids": judged_ids,
        })

        if (i + 1) % 50 == 0:
            logger.info("  [%s] %d/%d queries done", strategy, i + 1, total)

    fh.close()

    # Log timing
    if latencies:
        lat_sorted = sorted(latencies)
        logger.info("  [%s] P50=%.1fms P95=%.1fms (%d warm queries)",
                     strategy, lat_sorted[len(lat_sorted) // 2],
                     lat_sorted[int(len(lat_sorted) * 0.95)], len(latencies))

    return query_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strategies on WANDS")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files")
    parser.add_argument("--strategy", choices=STRATEGIES, default=None, help="Run single strategy only")
    parser.add_argument("--fake-embeddings", action="store_true", help="Use fake embeddings (fast, for testing)")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else STRATEGIES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load WANDS data
    logger.info("=== Loading WANDS data ===")
    wands_products = load_products(WANDS_DIR / "product.csv")
    wands_queries = load_queries(WANDS_DIR / "query.csv")
    qrels = load_qrels(WANDS_DIR / "label.csv")

    logger.info("Products: %d, Queries: %d, Qrels queries: %d",
                len(wands_products), len(wands_queries), len(qrels))

    # Build catalog and service
    logger.info("=== Building catalog and service ===")
    t0 = time.monotonic()
    catalog = build_catalog(wands_products)
    catalog_build_time = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    service = RetrievalService(
        catalog,
        data_path=WANDS_DIR / "product.csv",
        use_fake_embeddings=args.fake_embeddings,
    )
    service.initialize()
    init_time = (time.monotonic() - t0) * 1000

    status = service.status()
    logger.info("Service status: %s", json.dumps(status, indent=2))
    logger.info("Catalog build: %.0fms, Service init: %.0fms", catalog_build_time, init_time)

    # Collect source metadata
    source_manifest = {}
    manifest_path = WANDS_DIR / "source.json"
    if manifest_path.exists():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Evaluate each strategy
    logger.info("=== Evaluating %d strategies on %d queries ===", len(strategies), len(wands_queries))
    all_query_results: dict[str, list[dict]] = {}

    for strategy in strategies:
        logger.info("--- %s ---", strategy)
        runs_path = OUTPUT_DIR / f"runs_{strategy}.jsonl"
        qr = evaluate_strategy(
            service, strategy, wands_queries, qrels,
            output_path=runs_path, resume=args.resume,
        )
        all_query_results[strategy] = qr

        # Compute metrics
        metrics = compute_all_metrics(qr)
        logger.info("  nDCG@10=%.4f, Prec@10=%.4f, MRR=%.4f, Success@10=%.4f",
                     metrics["ndcg_at_10"], metrics["precision_at_10"],
                     metrics["exact_mrr_at_10"], metrics["exact_success_at_10"])

        # Save metrics
        metrics_path = OUTPUT_DIR / f"metrics_{strategy}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Aggregate per strategy
    summary: dict[str, dict[str, float]] = {}
    for s in strategies:
        if s in all_query_results:
            summary[s] = compute_all_metrics(all_query_results[s])

    # Rerank diagnostics
    if "rerank" in all_query_results and "hybrid" in all_query_results:
        _rerank_diagnostics(all_query_results, qrels)

    # Latency: collect from all strategies
    latency_summary: dict[str, dict[str, float]] = {}
    for s in strategies:
        runs_path = OUTPUT_DIR / f"runs_{s}.jsonl"
        if runs_path.exists():
            lats = []
            seen_qids: set[str] = set()
            with open(runs_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    qid = r["query_id"]
                    if qid not in seen_qids:
                        seen_qids.add(qid)
                        lats.append(r["latency_ms"])
            if lats:
                sl = sorted(lats)
                latency_summary[s] = {"p50": sl[len(sl)//2], "p95": sl[int(len(sl)*0.95)], "mean": statistics.mean(lats)}

    # Save metadata
    try:
        import subprocess
        current_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        current_sha = "unknown"

    metadata = {
        "wands_source": source_manifest.get("repo_url", "unknown"),
        "wands_commit": source_manifest.get("repo_commit_sha", "unknown"),
        "wands_file_sha256": source_manifest.get("files", {}),
        "marketlens_commit": current_sha,
        "embedding_model": status.get("embedding_model", "unknown"),
        "reranker_model": status.get("reranker_backend", "unknown"),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_k": 60,
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_build_ms": catalog_build_time,
        "service_init_ms": init_time,
        "product_count": len(wands_products),
        "query_count": len(wands_queries),
        "label_count": sum(len(v) for v in qrels.values()),
        "fake_embeddings": args.fake_embeddings,
    }
    meta_path = OUTPUT_DIR / "metadata_v1.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save latency summary
    latency_path = OUTPUT_DIR / "latency_v1.json"
    latency_path.write_text(json.dumps(latency_summary, indent=2), encoding="utf-8")

    logger.info("=== Evaluation Complete ===")
    for s in strategies:
        if s in summary:
            m = summary[s]
            logger.info("%s: nDCG@10=%.4f Prec@10=%.4f MRR=%.4f Success@10=%.4f",
                         s, m["ndcg_at_10"], m["precision_at_10"], m["exact_mrr_at_10"], m["exact_success_at_10"])


def _rerank_diagnostics(all_results: dict, qrels: dict) -> None:
    """Compute rerank-specific diagnostics comparing hybrid vs rerank."""
    hybrid_results = {r["query_id"]: r for r in all_results["hybrid"]}
    rerank_results = {r["query_id"]: r for r in all_results["rerank"]}

    improved = 0
    same = 0
    degraded = 0
    exact_in_hybrid_candidates = 0

    for qid, hr in hybrid_results.items():
        rr = rerank_results.get(qid)
        if rr is None:
            continue

        h_ndcg = None
        r_ndcg = None
        from marketlens.evaluation.metrics import ndcg_at_k
        h_ndcg = ndcg_at_k(hr["retrieved_ids"], hr["qrels"], 10)
        r_ndcg = ndcg_at_k(rr["retrieved_ids"], rr["qrels"], 10)

        if r_ndcg > h_ndcg + 0.001:
            improved += 1
        elif r_ndcg < h_ndcg - 0.001:
            degraded += 1
        else:
            same += 1

        # Exact in hybrid top 50 relevant
        relevant = get_relevant_products(qrels, qid)
        hybrid_top = hr["retrieved_ids"][:CANDIDATE_K]
        if relevant & set(hybrid_top):
            exact_in_hybrid_candidates += 1

    total = len(hybrid_results)
    logger.info("=== Rerank Diagnostics ===")
    logger.info("  Hybrid→Rerank nDCG@10: improved=%d, same=%d, degraded=%d (of %d)",
                 improved, same, degraded, total)
    logger.info("  Hybrid top-%d contains relevant: %d/%d (%.1f%%)",
                 CANDIDATE_K, exact_in_hybrid_candidates, total,
                 100 * exact_in_hybrid_candidates / total if total else 0)


if __name__ == "__main__":
    main()
