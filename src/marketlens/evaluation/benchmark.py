"""Reproducible evaluation benchmarks for MarketLens retrieval and agent.

All metrics are computed from actual execution results.
Fixture benchmarks are explicitly labeled as such.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """A single evaluation query with ground truth."""

    query_id: str
    query_text: str
    category: str  # exact_match, synonym, budget, multi_constraint, no_result, contradiction, insufficient_evidence
    relevant_product_ids: list[str]  # Ground truth relevant product IDs
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    expect_results: bool = True
    notes: str = ""


@dataclass
class QueryResult:
    """Result of evaluating one query."""

    query_id: str
    retrieved_ids: list[str]  # Product IDs in retrieval order
    recall_at_10: float
    ndcg_at_10: float
    constraints_satisfied: bool
    result_count: int
    duration_ms: float
    errors: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""

    title: str
    total_queries: int
    completed_queries: int
    failed_queries: int

    avg_recall_at_10: float
    avg_ndcg_at_10: float
    constraint_satisfaction_rate: float
    task_completion_rate: float
    avg_latency_ms: float

    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    query_results: list[QueryResult] = field(default_factory=list)
    is_fixture_data: bool = True


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """Compute Recall@K.

    Args:
        retrieved_ids: Product IDs in retrieval order.
        relevant_ids: Ground truth relevant product IDs.
        k: Cutoff rank.

    Returns:
        Recall@K value [0, 1].
    """
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def compute_ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 = relevant, 0 = not relevant).

    Args:
        retrieved_ids: Product IDs in retrieval order.
        relevant_ids: Ground truth relevant product IDs.
        k: Cutoff rank.

    Returns:
        nDCG@K value [0, 1].
    """
    import math

    if not relevant_ids:
        return 1.0

    relevant_set = set(relevant_ids)
    k = min(k, len(retrieved_ids))

    # DCG
    dcg = 0.0
    for i in range(k):
        if retrieved_ids[i] in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0 for rank 0

    # IDCG (ideal ranking: all relevant docs first)
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_hard_constraint_rate(results: list[QueryResult]) -> float:
    """Compute the hard constraint satisfaction rate.

    Args:
        results: List of query results.

    Returns:
        Rate [0, 1].
    """
    if not results:
        return 0.0
    satisfied = sum(1 for r in results if r.constraints_satisfied)
    return satisfied / len(results)


def compute_task_completion_rate(results: list[QueryResult]) -> float:
    """Compute task completion rate (queries with at least one result).

    Args:
        results: List of query results.

    Returns:
        Rate [0, 1].
    """
    if not results:
        return 0.0
    completed = sum(1 for r in results if r.result_count > 0)
    return completed / len(results)


def run_evaluation(
    queries: list[EvaluationQuery],
    search_fn: Callable[[str, int], list[str]],
    top_k: int = 10,
) -> EvaluationReport:
    """Run a complete evaluation across all queries.

    Args:
        queries: List of evaluation queries.
        search_fn: Function (query_text, top_k) -> list[product_id].
        top_k: Number of results to evaluate.

    Returns:
        EvaluationReport with all metrics.
    """
    query_results: list[QueryResult] = []

    for q in queries:
        errors: list[str] = []
        t0 = time.monotonic()

        try:
            retrieved = search_fn(q.query_text, top_k)
        except Exception as e:
            retrieved = []
            errors.append(str(e))

        elapsed_ms = (time.monotonic() - t0) * 1000

        recall = compute_recall_at_k(retrieved, q.relevant_product_ids, top_k)
        ndcg = compute_ndcg_at_k(retrieved, q.relevant_product_ids, top_k)

        # Constraint satisfaction: if expecting results, check we got some
        if q.expect_results:
            constraints_ok = len(retrieved) > 0 or not q.relevant_product_ids
        else:
            constraints_ok = len(retrieved) == 0  # Expecting no results

        query_results.append(QueryResult(
            query_id=q.query_id,
            retrieved_ids=retrieved,
            recall_at_10=recall,
            ndcg_at_10=ndcg,
            constraints_satisfied=constraints_ok,
            result_count=len(retrieved),
            duration_ms=elapsed_ms,
            errors=errors,
        ))

    # Aggregate metrics
    completed = [r for r in query_results if not r.errors]
    failed = [r for r in query_results if r.errors]

    avg_recall = sum(r.recall_at_10 for r in query_results) / len(query_results) if query_results else 0.0
    avg_ndcg = sum(r.ndcg_at_10 for r in query_results) / len(query_results) if query_results else 0.0
    constraint_rate = compute_hard_constraint_rate(query_results)
    completion_rate = compute_task_completion_rate(query_results)
    avg_latency = sum(r.duration_ms for r in query_results) / len(query_results) if query_results else 0.0

    # Per-category breakdown
    per_category: dict[str, dict[str, float]] = {}
    categories = set(q.category for q in queries)
    for cat in categories:
        cat_results = [r for r, q in zip(query_results, queries) if q.category == cat]
        if cat_results:
            per_category[cat] = {
                "count": float(len(cat_results)),
                "avg_recall": sum(r.recall_at_10 for r in cat_results) / len(cat_results),
                "avg_ndcg": sum(r.ndcg_at_10 for r in cat_results) / len(cat_results),
                "avg_latency_ms": sum(r.duration_ms for r in cat_results) / len(cat_results),
            }

    report = EvaluationReport(
        title="MarketLens Evaluation — Fixture Benchmark",
        total_queries=len(queries),
        completed_queries=len(completed),
        failed_queries=len(failed),
        avg_recall_at_10=round(avg_recall, 4),
        avg_ndcg_at_10=round(avg_ndcg, 4),
        constraint_satisfaction_rate=round(constraint_rate, 4),
        task_completion_rate=round(completion_rate, 4),
        avg_latency_ms=round(avg_latency, 2),
        per_category=per_category,
        query_results=query_results,
        is_fixture_data=True,
    )

    return report


def print_report(report: EvaluationReport) -> None:
    """Print an evaluation report to console.

    Args:
        report: The evaluation report.
    """
    print(f"\n{'='*60}")
    print(f"  {report.title}")
    print(f"{'='*60}")
    if report.is_fixture_data:
        print("  ⚠ FIXTURE BENCHMARK — not real dataset results")
    print(f"  Queries: {report.total_queries} total, "
          f"{report.completed_queries} completed, "
          f"{report.failed_queries} failed")
    print(f"  Recall@10:      {report.avg_recall_at_10:.4f}")
    print(f"  nDCG@10:        {report.avg_ndcg_at_10:.4f}")
    print(f"  Constraint Sat: {report.constraint_satisfaction_rate:.4f}")
    print(f"  Task Complete:  {report.task_completion_rate:.4f}")
    print(f"  Avg Latency:    {report.avg_latency_ms:.2f} ms")

    if report.per_category:
        print("\n  Per Category:")
        for cat, metrics in sorted(report.per_category.items()):
            print(f"    {cat}: Recall={metrics['avg_recall']:.4f}, "
                  f"nDCG={metrics['avg_ndcg']:.4f}, "
                  f"Latency={metrics['avg_latency_ms']:.2f}ms")
    print(f"{'='*60}\n")


def compare_retrievers(
    queries: list[EvaluationQuery],
    bm25_fn: Callable[[str, int], list[str]],
    embedding_fn: Callable[[str, int], list[str]],
    hybrid_fn: Callable[[str, int], list[str]],
    top_k: int = 10,
) -> dict[str, EvaluationReport]:
    """Compare BM25, embedding, and hybrid retrieval.

    Args:
        queries: Evaluation queries.
        bm25_fn: BM25 search function.
        embedding_fn: Embedding search function.
        hybrid_fn: Hybrid search function.
        top_k: Results cutoff.

    Returns:
        Dict of method name to EvaluationReport.
    """
    reports = {}
    for name, fn in [("bm25", bm25_fn), ("embedding", embedding_fn), ("hybrid", hybrid_fn)]:
        logger.info("Evaluating %s retrieval...", name)
        report = run_evaluation(queries, fn, top_k)
        report.title = f"MarketLens — {name.upper()} Retrieval (Fixture)"
        reports[name] = report
        print_report(report)
    return reports
