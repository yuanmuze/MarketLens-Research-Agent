"""Retrieval strategy comparison: BM25, Embedding, Hybrid RRF, Hybrid+Reranker.

Provides a unified framework for running side-by-side comparisons of all
four retrieval strategies on the same data, queries, and constraints.

Uses RealDataLoader for Amazon data (falls back to fixture when unavailable).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from marketlens.catalog import ProductCatalog
from marketlens.models import SearchQuery, UserConstraints
from marketlens.retrieval.bm25 import BM25Retriever
from marketlens.retrieval.embedding import (
    EmbeddingBackend,
    EmbeddingRetriever,
    FakeEmbeddingBackend,
    SentenceTransformersBackend,
)
from marketlens.retrieval.hybrid import HybridRetriever
from marketlens.retrieval.reranker import KeywordReranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation query set (50+ queries, auto-curated, synthetic)
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""

    query_id: str
    query_text: str
    category: str
    constraints: dict[str, Any] = field(default_factory=dict)
    relevant_product_ids: list[str] = field(default_factory=list)
    relevance_grades: dict[str, int] = field(default_factory=dict)  # pid -> 0|1|2|3
    label_source: str = "synthetic"
    review_status: str = "pending"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "category": self.category,
            "constraints": self.constraints,
            "relevant_product_ids": self.relevant_product_ids,
            "relevance_grades": self.relevance_grades,
            "label_source": self.label_source,
            "review_status": self.review_status,
            "notes": self.notes,
        }


def build_eval_queries(catalog: ProductCatalog | None = None) -> list[EvalQuery]:
    """Build 50+ evaluation queries from catalog products.

    Labels are auto-curated (synthetic). review_status is "pending" until
    human review. All queries flagged as label_source="synthetic" or
    "auto_curated".

    Args:
        catalog: Optional ProductCatalog to derive queries from.

    Returns:
        List of 50+ EvalQuery objects.
    """
    queries: list[EvalQuery] = []

    if catalog is None or len(catalog) == 0:
        # Use generic audio/electronics queries
        products_dict = {}
    else:
        products_dict = {p.product_id: p for p in catalog.get_all_products()}

    # --- Category: exact_match (8 queries) ---
    exact_pids = list(products_dict.keys())[:8]
    if exact_pids:
        for i, pid in enumerate(exact_pids, 1):
            p = products_dict[pid]
            queries.append(EvalQuery(
                query_id=f"exact-{i:03d}",
                query_text=p.title,
                category="exact_match",
                relevant_product_ids=[pid],
                relevance_grades={pid: 3},
                label_source="auto_curated",
                review_status="pending",
                notes=f"Exact title match: {p.title[:80]}",
            ))
    else:
        # Generic exact match queries when no catalog available
        generic_exact = [
            "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            "Bose QuietComfort Ultra Wireless Headphones",
            "Apple AirPods Pro 2nd Generation",
            "Samsung Galaxy Buds3 Pro",
            "Sennheiser Momentum 4 Wireless Headphones",
            "JBL Tour One M2 Wireless Headphones",
            "Anker Soundcore Space A40 Earbuds",
            "Google Pixel Buds Pro",
        ]
        for i, title in enumerate(generic_exact, 1):
            queries.append(EvalQuery(
                query_id=f"exact-{i:03d}",
                query_text=title,
                category="exact_match",
                relevant_product_ids=[],
                label_source="synthetic",
                review_status="pending",
                notes=f"Exact title match (generic, no catalog): {title[:80]}",
            ))

    # --- Category: synonym (8 queries) ---
    synonym_map = [
        ("noise cancelling wireless over-ear headphones", "headphones with ANC and bluetooth"),
        ("true wireless in-ear earbuds for calls", "TWS earbuds with microphone"),
        ("over-ear studio monitoring headphones", "professional reference headphones wired"),
        ("sports earbuds waterproof sweat resistant", "running earphones IPX rated"),
        ("budget friendly wireless earbuds long battery", "cheap bluetooth earphones long playtime"),
        ("premium audiophile headphones high fidelity", "high-end studio quality sound headphones"),
        ("smart speaker with voice assistant", "AI speaker with built-in microphone"),
        ("gaming headset with microphone low latency", "esports headphones with mic wireless"),
    ]
    for i, (query_text, _notes) in enumerate(synonym_map, 1):
        queries.append(EvalQuery(
            query_id=f"syn-{i:03d}",
            query_text=query_text,
            category="synonym",
            relevant_product_ids=[],
            label_source="auto_curated",
            review_status="pending",
            notes=_notes,
        ))

    # --- Category: brand_filter (8 queries) ---
    brands = ["Sony", "Bose", "Apple", "Samsung", "Sennheiser", "JBL", "Anker", "Google"]
    for i, brand in enumerate(brands, 1):
        queries.append(EvalQuery(
            query_id=f"brand-{i:03d}",
            query_text=f"{brand} wireless audio",
            category="brand_filter",
            constraints={"preferred_brands": [brand]},
            relevant_product_ids=[],
            label_source="auto_curated",
            review_status="pending",
            notes=f"Brand filter: {brand}",
        ))

    # --- Category: budget (8 queries) ---
    budget_tiers = [
        ("wireless earbuds under $50", 50.0),
        ("headphones under $100", 100.0),
        ("noise cancelling headphones under $200", 200.0),
        ("premium wireless headphones under $300", 300.0),
        ("best earbuds under $150", 150.0),
        ("affordable ANC headphones under $250", 250.0),
        ("flagship headphones under $400", 400.0),
        ("budget gaming headset under $80", 80.0),
    ]
    for i, (query_text, budget) in enumerate(budget_tiers, 1):
        queries.append(EvalQuery(
            query_id=f"budget-{i:03d}",
            query_text=query_text,
            category="budget",
            constraints={"max_budget": budget},
            relevant_product_ids=[],
            label_source="auto_curated",
            review_status="pending",
            notes=f"Max budget: ${budget:.2f}",
        ))

    # --- Category: multi_constraint (8 queries) ---
    multi_queries: list[tuple[str, dict[str, Any]]] = [
        ("Sony noise cancelling headphones under $350 with 30h+ battery", {"max_budget": 350.0, "preferred_brands": ["Sony"]}),
        ("high rated bluetooth headphones with spatial audio ANC", {"min_rating": 4.3}),
        ("Apple wireless earbuds with ANC under $300", {"max_budget": 300.0, "preferred_brands": ["Apple"]}),
        ("Bose premium headphones under $500 with best ANC", {"max_budget": 500.0, "preferred_brands": ["Bose"]}),
        ("Samsung earbuds with long battery life under $200", {"max_budget": 200.0, "preferred_brands": ["Samsung"]}),
        ("waterproof sports earbuds under $150 with 30h+ battery", {"max_budget": 150.0}),
        ("studio headphones with 4.5+ rating under $400", {"min_rating": 4.5, "max_budget": 400.0}),
        ("wireless ANC over-ear headphones 4.0+ rating under $250", {"min_rating": 4.0, "max_budget": 250.0}),
    ]
    for i, (query_text, constraints) in enumerate(multi_queries, 1):
        queries.append(EvalQuery(
            query_id=f"multi-{i:03d}",
            query_text=query_text,
            category="multi_constraint",
            constraints=constraints,
            relevant_product_ids=[],
            label_source="auto_curated",
            review_status="pending",
            notes=f"Multi-constraint: {constraints}",
        ))

    # --- Category: attribute (6 queries) ---
    attr_queries: list[tuple[str, dict[str, Any]]] = [
        ("black wireless earbuds bluetooth 5.3", {}),
        ("headphones with 40h+ battery life", {}),
        ("waterproof IPX7 earbuds for sports", {}),
        ("headphones with USB-C charging and multipoint", {}),
        ("earbuds with wireless charging case", {}),
        ("lightweight headphones under 200g", {}),
    ]
    for i, (query_text, constraints) in enumerate(attr_queries, 1):
        queries.append(EvalQuery(
            query_id=f"attr-{i:03d}",
            query_text=query_text,
            category="attribute",
            constraints=constraints,
            relevant_product_ids=[],
            label_source="auto_curated",
            review_status="pending",
            notes="Attribute-focused query",
        ))

    # --- Category: no_result (6 queries) ---
    no_result_queries = [
        "wireless headphones under $5",
        "Nintendo Switch OLED pro gaming headphones",
        "professional DJ mixer controller headphones",
        "medical grade hearing aid bluetooth",
        "military spec tactical communication headset",
        "aerospace grade noise cancelling aviation headset",
    ]
    for i, query_text in enumerate(no_result_queries, 1):
        queries.append(EvalQuery(
            query_id=f"none-{i:03d}",
            query_text=query_text,
            category="no_result",
            constraints={},
            relevant_product_ids=[],
            label_source="auto_curated",
            review_status="pending",
            notes="Expected: no results",
        ))

    # --- Category: contradiction (4 queries) ---
    contradiction_queries = [
        ("best $500 headphones under $100", {"max_budget": 100.0}),
        ("premium audiophile headphones under $20", {"max_budget": 20.0}),
        ("flagship ANC headphones under $30 with 50h battery", {"max_budget": 30.0}),
        ("professional studio monitors under $10", {"max_budget": 10.0}),
    ]
    for i, (query_text, constraints) in enumerate(contradiction_queries, 1):
        queries.append(EvalQuery(
            query_id=f"contra-{i:03d}",
            query_text=query_text,
            category="contradiction",
            constraints=constraints,
            relevant_product_ids=[],
            label_source="auto_curated",
            review_status="pending",
            notes="Contradictory constraints",
        ))

    # --- Category: insufficient_evidence (4 queries) ---
    insuf_queries = [
        "best mid-range open-back planar magnetic headphones",
        "tube amplifier compatible high impedance headphones",
        "bone conduction swimming headphones waterproof",
        "biometric heart rate monitoring earbuds",
    ]
    for i, query_text in enumerate(insuf_queries, 1):
        queries.append(EvalQuery(
            query_id=f"insuf-{i:03d}",
            query_text=query_text,
            category="insufficient_evidence",
            constraints={},
            relevant_product_ids=[],
            label_source="auto_curated",
            review_status="pending",
            notes="Specialized category, likely not in catalog",
        ))

    return queries


# ---------------------------------------------------------------------------
# Comparison framework
# ---------------------------------------------------------------------------

@dataclass
class RetrievalTiming:
    """Timing stats for a retrieval run."""

    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_latencies(cls, latencies_ms: list[float]) -> RetrievalTiming:
        """Compute timing percentiles from raw latencies."""
        if not latencies_ms:
            return cls()
        arr = np.array(latencies_ms)
        return cls(
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            mean_ms=float(np.mean(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
        )


@dataclass
class RetrievalResult:
    """Result from one retrieval strategy on one query."""

    query_id: str
    strategy: str  # bm25, embedding, hybrid, hybrid_rerank
    product_ids: list[str]
    scores: list[float]
    duration_ms: float
    error: str | None = None


@dataclass
class StrategyReport:
    """Aggregated report for one retrieval strategy."""

    strategy: str
    total_queries: int
    success_count: int
    error_count: int
    recall_at_10: float
    ndcg_at_10: float
    constraint_satisfaction_rate: float
    no_result_correct_rate: float
    timing: RetrievalTiming
    results: list[RetrievalResult] = field(default_factory=list)


def compute_strategy_report(
    strategy: str,
    queries: list[EvalQuery],
    search_fn: Callable[[str, int, dict[str, Any] | None], tuple[list[str], list[float]]],
    top_k: int = 10,
) -> StrategyReport:
    """Run all queries through one retrieval strategy and compute metrics.

    Args:
        strategy: Strategy name.
        queries: Evaluation queries.
        search_fn: (query_text, top_k, constraints_dict) -> (product_ids, scores).
        top_k: Results cutoff.

    Returns:
        StrategyReport with all metrics.
    """
    from marketlens.evaluation.benchmark import compute_ndcg_at_k, compute_recall_at_k

    results: list[RetrievalResult] = []
    latencies: list[float] = []
    constraint_failures = 0
    no_result_correct = 0
    no_result_total = 0

    for q in queries:
        t0 = time.monotonic()
        error = None
        try:
            pids, scores = search_fn(q.query_text, top_k, q.constraints)
        except Exception as e:
            pids, scores = [], []
            error = str(e)

        elapsed_ms = (time.monotonic() - t0) * 1000
        latencies.append(elapsed_ms)

        results.append(RetrievalResult(
            query_id=q.query_id,
            strategy=strategy,
            product_ids=pids,
            scores=scores,
            duration_ms=elapsed_ms,
            error=error,
        ))

        # Track constraint/no-result stats
        if q.constraints and len(q.constraints) > 0:
            # Check if constraints were respected (simple check)
            if error is None and len(pids) == 0:
                constraint_failures += 1

        if q.category == "no_result" or q.category == "contradiction":
            no_result_total += 1
            if len(pids) == 0:
                no_result_correct += 1

    # Compute aggregated metrics
    success_results = [r for r in results if r.error is None]
    avg_recall = 0.0
    avg_ndcg = 0.0
    n = 0

    for q, r in zip(queries, results):
        if r.error is not None:
            continue
        if q.relevant_product_ids:
            avg_recall += compute_recall_at_k(r.product_ids, q.relevant_product_ids, top_k)
            avg_ndcg += compute_ndcg_at_k(r.product_ids, q.relevant_product_ids, top_k)
            n += 1

    if n > 0:
        avg_recall /= n
        avg_ndcg /= n

    constraint_rate = 1.0 - (constraint_failures / len(queries)) if queries else 1.0
    nr_rate = no_result_correct / no_result_total if no_result_total > 0 else 1.0

    return StrategyReport(
        strategy=strategy,
        total_queries=len(queries),
        success_count=len(success_results),
        error_count=len(results) - len(success_results),
        recall_at_10=round(avg_recall, 4),
        ndcg_at_10=round(avg_ndcg, 4),
        constraint_satisfaction_rate=round(constraint_rate, 4),
        no_result_correct_rate=round(nr_rate, 4),
        timing=RetrievalTiming.from_latencies(latencies),
        results=results,
    )


def run_full_comparison(
    catalog: ProductCatalog,
    queries: list[EvalQuery],
    use_real_embeddings: bool = False,
    top_k: int = 10,
) -> dict[str, StrategyReport]:
    """Run BM25, Embedding, Hybrid RRF, and Hybrid+Reranker comparison.

    Args:
        catalog: Product catalog.
        queries: Evaluation queries.
        use_real_embeddings: If True, try sentence-transformers. Falls back to fake.
        top_k: Results cutoff.

    Returns:
        Dict of strategy name -> StrategyReport.
    """
    logger.info("Running retrieval comparison on %d queries...", len(queries))

    # Set up backends
    texts = catalog.get_search_texts()
    doc_ids = catalog.get_product_ids()

    # BM25
    bm25 = BM25Retriever().fit(texts, doc_ids)

    # Embedding backend
    if use_real_embeddings:
        try:
            emb_backend: EmbeddingBackend = SentenceTransformersBackend(batch_size=32)
            logger.info("Using sentence-transformers backend")
        except ImportError:
            logger.warning("sentence-transformers not available, using fake backend")
            emb_backend = FakeEmbeddingBackend(dim=128, seed=42)
    else:
        emb_backend = FakeEmbeddingBackend(dim=128, seed=42)

    emb_retriever = EmbeddingRetriever(emb_backend).fit(texts, doc_ids)

    # Hybrid (without reranker)
    hybrid = HybridRetriever(catalog, bm25=bm25, embedding=emb_retriever).fit()

    # Hybrid with reranker
    hybrid_rerank = HybridRetriever(
        catalog, bm25=bm25, embedding=emb_retriever,
        reranker=KeywordReranker(),
    ).fit()

    # Search functions
    def bm25_search(query: str, k: int, constraints: dict | None) -> tuple[list[str], list[float]]:
        cat_filtered = _apply_constraints(catalog, constraints)
        raw = bm25.search(query, k * 2)
        results = [(pid, s) for pid, s in raw if pid in cat_filtered][:k]
        return ([pid for pid, _ in results], [s for _, s in results])

    def emb_search(query: str, k: int, constraints: dict | None) -> tuple[list[str], list[float]]:
        cat_filtered = _apply_constraints(catalog, constraints)
        raw = emb_retriever.search(query, k * 2)
        results = [(pid, s) for pid, s in raw if pid in cat_filtered][:k]
        return ([pid for pid, _ in results], [s for _, s in results])

    def hybrid_search(query: str, k: int, constraints: dict | None) -> tuple[list[str], list[float]]:
        filters = _build_constraints(constraints)
        sq = SearchQuery(text=query, top_k=k, filters=filters, use_reranker=False)
        results = hybrid.search(sq)
        return ([r.product.product_id for r in results], [r.score for r in results])

    def hybrid_rerank_search(query: str, k: int, constraints: dict | None) -> tuple[list[str], list[float]]:
        filters = _build_constraints(constraints)
        sq = SearchQuery(text=query, top_k=k, filters=filters, use_reranker=True)
        results = hybrid_rerank.search(sq)
        return ([r.product.product_id for r in results], [r.score for r in results])

    # Run comparison
    reports: dict[str, StrategyReport] = {}
    for name, fn in [
        ("bm25", bm25_search),
        ("embedding", emb_search),
        ("hybrid", hybrid_search),
        ("hybrid_rerank", hybrid_rerank_search),
    ]:
        logger.info("  Running %s...", name)
        reports[name] = compute_strategy_report(name, queries, fn, top_k)

    return reports


def _apply_constraints(
    catalog: ProductCatalog, constraints: dict[str, Any] | None
) -> set[str]:
    """Apply hard constraints and return set of allowed product IDs."""
    if not constraints:
        return set(catalog.get_product_ids())

    return set(catalog.filter_by_constraints(
        max_budget=constraints.get("max_budget"),
        min_budget=constraints.get("min_budget"),
        brands=constraints.get("preferred_brands"),
        excluded_brands=constraints.get("excluded_brands"),
        min_rating=constraints.get("min_rating"),
        min_review_count=constraints.get("min_review_count"),
    ))


def _build_constraints(constraints: dict[str, Any] | None) -> UserConstraints | None:
    """Build UserConstraints from dict."""
    if not constraints:
        return None
    return UserConstraints(
        max_budget=constraints.get("max_budget"),
        min_budget=constraints.get("min_budget"),
        preferred_brands=constraints.get("preferred_brands", []),
        excluded_brands=constraints.get("excluded_brands", []),
        min_rating=constraints.get("min_rating"),
        min_review_count=constraints.get("min_review_count"),
    )


def save_comparison_results(
    reports: dict[str, StrategyReport],
    queries: list[EvalQuery],
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save comparison results to disk.

    Args:
        reports: Strategy reports.
        queries: Evaluation queries.
        output_dir: Directory to save to.
        config: Optional run configuration.

    Returns:
        Path to the summaries file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-strategy results
    for name, report in reports.items():
        path = output_dir / f"results_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": report.strategy,
                "total_queries": report.total_queries,
                "recall_at_10": report.recall_at_10,
                "ndcg_at_10": report.ndcg_at_10,
                "constraint_satisfaction_rate": report.constraint_satisfaction_rate,
                "no_result_correct_rate": report.no_result_correct_rate,
                "timing": {
                    "p50_ms": report.timing.p50_ms,
                    "p95_ms": report.timing.p95_ms,
                    "mean_ms": report.timing.mean_ms,
                    "min_ms": report.timing.min_ms,
                    "max_ms": report.timing.max_ms,
                },
                "per_query": [
                    {
                        "query_id": r.query_id,
                        "product_ids": r.product_ids[:10],
                        "scores": r.scores[:10],
                        "duration_ms": r.duration_ms,
                        "error": r.error,
                    }
                    for r in report.results
                ],
            }, f, indent=2)

    # Save summary
    summary_path = output_dir / "comparison_summary.json"
    strategies_data: dict[str, dict[str, Any]] = {}
    for name, report in reports.items():
        strategies_data[name] = {
            "recall_at_10": report.recall_at_10,
            "ndcg_at_10": report.ndcg_at_10,
            "constraint_satisfaction_rate": report.constraint_satisfaction_rate,
            "no_result_correct_rate": report.no_result_correct_rate,
            "p50_ms": report.timing.p50_ms,
            "p95_ms": report.timing.p95_ms,
            "mean_ms": report.timing.mean_ms,
            "success_count": report.success_count,
            "error_count": report.error_count,
        }

    summary: dict[str, Any] = {
        "run_config": config or {},
        "timestamp": datetime.now(UTC).isoformat(),
        "query_count": len(queries),
        "query_categories": list(set(q.category for q in queries)),
        "strategies": strategies_data,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save query definitions
    queries_path = output_dir / "eval_queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump([q.to_dict() for q in queries], f, indent=2)

    logger.info("Results saved to %s", output_dir)
    return summary_path


def generate_markdown_report(
    reports: dict[str, StrategyReport],
    queries: list[EvalQuery],
    config: dict[str, Any] | None = None,
) -> str:
    """Generate a Markdown summary of the comparison results.

    Args:
        reports: Strategy reports.
        queries: Evaluation queries.
        config: Run configuration metadata.

    Returns:
        Markdown report string.
    """
    lines = [
        "# MarketLens Retrieval Strategy Comparison",
        "",
        f"**Run date**: {datetime.now(UTC).isoformat()}",
        f"**Query count**: {len(queries)}",
        "**Top-K**: 10",
    ]
    if config:
        lines.append(f"**Data**: {config.get('data_source', 'fixture')}")
        lines.append(f"**Embedding**: {config.get('embedding_backend', 'fake')}")
        lines.append(f"**Model**: {config.get('model_name', 'N/A')}")
        lines.append(f"**Seed**: {config.get('seed', 'N/A')}")
        lines.append(f"**Python**: {config.get('python_version', 'N/A')}")
        lines.append(f"**OS**: {config.get('os', 'N/A')}")

    lines += [
        "",
        "## Results Summary",
        "",
        "| Strategy | Recall@10 | nDCG@10 | Constraint% | NoResult% | P50 (ms) | P95 (ms) | Mean (ms) |",
        "|----------|-----------|---------|-------------|-----------|----------|----------|-----------|",
    ]

    for name in ["bm25", "embedding", "hybrid", "hybrid_rerank"]:
        r = reports.get(name)
        if r is None:
            continue
        lines.append(
            f"| {name} | {r.recall_at_10:.4f} | {r.ndcg_at_10:.4f} | "
            f"{r.constraint_satisfaction_rate:.4f} | {r.no_result_correct_rate:.4f} | "
            f"{r.timing.p50_ms:.1f} | {r.timing.p95_ms:.1f} | {r.timing.mean_ms:.1f} |"
        )

    lines += [
        "",
        "## Query Categories",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    from collections import Counter
    cats = Counter(q.category for q in queries)
    for cat, count in sorted(cats.items()):
        lines.append(f"| {cat} | {count} |")

    lines += [
        "",
        "## Label Disclaimer",
        "",
        "⚠ **All queries in this evaluation are auto-curated (synthetic).**",
        "Relevant product IDs are derived from catalog metadata, not human judgment.",
        "`review_status` is `pending` until manually reviewed.",
        "These results represent relative comparisons between retrieval strategies,",
        "not absolute quality judgments.",
        "",
        "See `docs/evaluation.md` for metric definitions and evidence rules.",
        "",
        "---",
        "*Generated by MarketLens Evaluation Framework*",
    ]

    return "\n".join(lines)
