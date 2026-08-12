"""Information retrieval metrics for WANDS evaluation.

All metrics computed per-query, then macro-averaged across queries.
Multi-grade relevance: Exact=2, Partial=1, Irrelevant=0.

Definitions:
- nDCG@k: multi-grade (0/1/2) DCG / IDCG
- Precision@k: relevance >= 1 counts as relevant
- Exact MRR@10: reciprocal rank of first Exact=2 result
- Exact Success@10: whether top-10 contains at least one Exact=2
- Recall@k: fraction of known relevant products found
"""

from __future__ import annotations

import math
from typing import Any


def ndcg_at_k(
    retrieved_ids: list[str],
    qrels: dict[str, int],
    k: int,
) -> float:
    """Compute nDCG@k with graded relevance (0/1/2).

    Args:
        retrieved_ids: Ranked list of product IDs.
        qrels: dict[product_id, relevance_grade].
        k: Cutoff rank.

    Returns:
        nDCG@k [0, 1].
    """
    k = min(k, len(retrieved_ids))
    dcg = 0.0
    for i in range(k):
        grade = qrels.get(retrieved_ids[i], 0)
        dcg += (2 ** grade - 1) / math.log2(i + 2)

    # Ideal DCG: sort all judged items by grade descending
    ideal_grades = sorted(qrels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(ideal_grades):
        idcg += (2 ** grade - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    retrieved_ids: list[str],
    qrels: dict[str, int],
    k: int,
    min_grade: int = 1,
) -> float:
    """Compute Precision@k (binary: relevance >= min_grade).

    Args:
        retrieved_ids: Ranked product IDs.
        qrels: dict[product_id, relevance_grade].
        k: Cutoff.
        min_grade: Minimum grade to count as relevant.

    Returns:
        Precision@k [0, 1].
    """
    k = min(k, len(retrieved_ids))
    if k == 0:
        return 0.0
    relevant = sum(1 for pid in retrieved_ids[:k] if qrels.get(pid, 0) >= min_grade)
    return relevant / k


def exact_mrr_at_10(
    retrieved_ids: list[str],
    qrels: dict[str, int],
) -> float:
    """Compute MRR@10 for Exact=2 matches.

    Args:
        retrieved_ids: Ranked product IDs.
        qrels: dict[product_id, relevance_grade].

    Returns:
        Reciprocal rank of first Exact match [0, 1].
    """
    for rank, pid in enumerate(retrieved_ids[:10], 1):
        if qrels.get(pid, 0) == 2:
            return 1.0 / rank
    return 0.0


def exact_success_at_10(
    retrieved_ids: list[str],
    qrels: dict[str, int],
) -> int:
    """Check if top-10 contains an Exact=2 match.

    Args:
        retrieved_ids: Ranked product IDs.
        qrels: dict[product_id, relevance_grade].

    Returns:
        1 if Exact found in top-10, 0 otherwise.
    """
    return 1 if any(qrels.get(pid, 0) == 2 for pid in retrieved_ids[:10]) else 0


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Recall@k against known relevant products.

    Args:
        retrieved_ids: Ranked product IDs.
        relevant_ids: Set of known relevant product IDs.
        k: Cutoff.

    Returns:
        Recall@k [0, 1].
    """
    if not relevant_ids:
        return 1.0
    k = min(k, len(retrieved_ids))
    found = relevant_ids & set(retrieved_ids[:k])
    return len(found) / len(relevant_ids)


def judged_ratio_at_k(
    retrieved_ids: list[str],
    judged_ids: set[str],
    k: int,
) -> dict[str, float]:
    """Compute judged and unjudged ratios at k.

    Args:
        retrieved_ids: Ranked product IDs.
        judged_ids: Set of judged product IDs for this query.
        k: Cutoff.

    Returns:
        dict with "judged_ratio" and "unjudged_ratio".
    """
    k = min(k, len(retrieved_ids))
    if k == 0:
        return {"judged_ratio": 0.0, "unjudged_ratio": 0.0}
    top = retrieved_ids[:k]
    judged = sum(1 for pid in top if pid in judged_ids)
    return {
        "judged_ratio": judged / k,
        "unjudged_ratio": (k - judged) / k,
    }


def macro_average(values: list[float]) -> float:
    """Compute macro average (mean of per-query values)."""
    return sum(values) / len(values) if values else 0.0


def compute_all_metrics(
    query_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute all metrics from per-query results.

    Args:
        query_results: List of dicts with keys:
            retrieved_ids, qrels, relevant_ids, judged_ids.

    Returns:
        Dict of macro-averaged metrics.
    """
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    prec5: list[float] = []
    prec10: list[float] = []
    mrr: list[float] = []
    success: list[float] = []
    recall50: list[float] = []
    judged10: list[float] = []
    unjudged10: list[float] = []
    short_results = 0

    for qr in query_results:
        retrieved = qr["retrieved_ids"]
        qrels = qr["qrels"]
        relevant = qr.get("relevant_ids", set())
        judged = qr.get("judged_ids", set())

        ndcg5.append(ndcg_at_k(retrieved, qrels, 5))
        ndcg10.append(ndcg_at_k(retrieved, qrels, 10))
        prec5.append(precision_at_k(retrieved, qrels, 5))
        prec10.append(precision_at_k(retrieved, qrels, 10))
        mrr.append(exact_mrr_at_10(retrieved, qrels))
        success.append(exact_success_at_10(retrieved, qrels))
        recall50.append(recall_at_k(retrieved, relevant, 50))

        jr = judged_ratio_at_k(retrieved, judged, 10)
        judged10.append(jr["judged_ratio"])
        unjudged10.append(jr["unjudged_ratio"])

        if len(retrieved) < 10:
            short_results += 1

    return {
        "ndcg_at_5": macro_average(ndcg5),
        "ndcg_at_10": macro_average(ndcg10),
        "precision_at_5": macro_average(prec5),
        "precision_at_10": macro_average(prec10),
        "exact_mrr_at_10": macro_average(mrr),
        "exact_success_at_10": macro_average(success),
        "recall_at_50": macro_average(recall50),
        "top10_judged_ratio": macro_average(judged10),
        "top10_unjudged_ratio": macro_average(unjudged10),
        "queries_with_short_results": short_results,
        "total_queries": len(query_results),
    }
