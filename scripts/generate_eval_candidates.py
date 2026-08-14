#!/usr/bin/env python3
"""Generate 50 candidate evaluation queries from real product data.

Produces auto-curated queries across 5 categories (keyword, semantic,
attribute, combined, hard). All queries reference only products that
exist in the input data. No LLM API calls. Fully reproducible with seed=42.

Output:
  data/eval/eval_candidates.jsonl  — one JSON object per line
  data/eval/eval_review.csv        — CSV for manual review
  reports/eval_candidate_summary.md — Review guidelines and summary

Usage:
  uv run python scripts/generate_eval_candidates.py
  uv run python scripts/generate_eval_candidates.py --products data/processed/electronics_2000.json --count 50 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PRODUCTS = Path("data/processed/electronics_2000.json")
DEFAULT_EVAL_DIR = Path("data/eval")
REPORTS_DIR = Path("reports")
DEFAULT_COUNT = 50
DEFAULT_SEED = 42

QUERY_TYPES = ["keyword", "semantic", "attribute", "combined", "hard"]


@dataclass
class EvalCandidate:
    """A candidate eval query awaiting human review."""

    query_id: str
    query: str
    query_type: str  # keyword, semantic, attribute, combined, hard
    expected_product_ids: list[str]
    expected_constraints: dict[str, Any] = field(default_factory=dict)
    source_product_ids: list[str] = field(default_factory=list)
    generation_reason: str = ""
    reviewer_status: str = "pending"  # pending, approved, revise, rejected
    reviewer_notes: str = ""


def load_products(path: Path) -> list[dict[str, Any]]:
    """Read the product JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        List of product dicts.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array")
    return data


def build_index(products: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index products by product_id for fast lookup.

    Args:
        products: Product list.

    Returns:
        Dict of product_id -> product.
    """
    return {p["product_id"]: p for p in products}


def pick_products(
    products: list[dict[str, Any]],
    rng: random.Random,
    n: int,
    *,
    min_rating: float | None = None,
    min_reviews: int | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_title_len: int = 0,
    with_brand: bool = False,
    with_description: bool = False,
) -> list[dict[str, Any]]:
    """Pick n diverse products matching optional filters.

    Args:
        products: Full product list.
        rng: Seeded Random instance.
        n: Number of products to pick.
        min_rating: Minimum rating.
        min_reviews: Minimum review count.
        min_price: Minimum price.
        max_price: Maximum price.
        min_title_len: Minimum title length in chars.
        with_brand: Only products with a brand.
        with_description: Only products with description.

    Returns:
        Up to n matching products.
    """
    candidates = []
    for p in products:
        if min_rating is not None and (p.get("rating") or 0) < min_rating:
            continue
        if min_reviews is not None and (p.get("review_count") or 0) < min_reviews:
            continue
        price = p.get("price")
        if price is None:
            continue
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        if min_title_len and len(p.get("title", "")) < min_title_len:
            continue
        if with_brand and not p.get("brand"):
            continue
        if with_description and not p.get("description"):
            continue
        candidates.append(p)

    if len(candidates) <= n:
        return candidates
    return rng.sample(candidates, n)


def _format_price(p: dict[str, Any]) -> str:
    """Format price for display."""
    pr = p.get("price")
    return f"${pr:.2f}" if pr is not None else "?"


def generate_keyword_queries(
    products: list[dict[str, Any]], rng: random.Random, n: int,
) -> list[EvalCandidate]:
    """Generate keyword queries: concrete brand, model, or product names.

    Uses actual brand+title fragments from real products.
    """
    candidates: list[EvalCandidate] = []

    # Pick products with brands and meaningful titles
    pool = pick_products(
        products, rng, n * 3,
        min_title_len=30, with_brand=True, min_reviews=5,
    )

    seen_ids: set[str] = set()
    count = 0
    for p in pool:
        if count >= n:
            break
        pid = p["product_id"]
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        # Build keyword-style query from brand + key terms
        title = p.get("title", "")
        brand = p.get("brand", "")
        # Take first 3-4 significant words from title (after brand)
        words = [w for w in title.split() if len(w) > 2 and w.lower() != brand.lower()]
        key_terms = " ".join(words[:4])

        if brand and key_terms:
            query = f"{brand} {key_terms}"
        else:
            query = key_terms if key_terms else title[:80]

        candidates.append(EvalCandidate(
            query_id=f"kw-{count + 1:03d}",
            query=query,
            query_type="keyword",
            expected_product_ids=[pid],
            source_product_ids=[pid],
            generation_reason=f"Branded product search from '{brand}' with key terms '{key_terms[:60]}'",
            reviewer_status="pending",
        ))
        count += 1

    return candidates


def generate_semantic_queries(
    products: list[dict[str, Any]], rng: random.Random, n: int,
) -> list[EvalCandidate]:
    """Generate semantic queries: natural language descriptions, not copy-paste titles.

    Uses templates with product attributes to create realistic user queries.
    """
    templates = [
        ("wireless", "wireless {category} with long battery life", ["wireless", "bluetooth", "earbuds", "headphones", "speaker"]),
        ("noise", "noise cancelling {category} for travel", ["noise", "cancelling", "headphones", "earbuds", "anc"]),
        ("audio quality", "best sounding {category} for music", ["headphones", "earbuds", "speaker", "audio", "sound"]),
        ("budget", "affordable {category} with good reviews", ["headphones", "earbuds", "speaker", "charger", "cable"]),
        ("premium", "high quality {category} worth the money", ["headphones", "earbuds", "speaker", "audio"]),
        ("portable", "compact {category} easy to carry around", ["earbuds", "charger", "speaker", "headphones"]),
        ("sports", "{category} suitable for workouts and running", ["earbuds", "headphones"]),
        ("calling", "{category} with clear microphone for phone calls", ["earbuds", "headphones", "headset"]),
        ("gaming", "immersive {category} for gaming", ["headphones", "headset", "earbuds"]),
        ("home", "good {category} for home office use", ["headphones", "speaker", "earbuds", "webcam"]),
    ]

    candidates: list[EvalCandidate] = []
    rng.shuffle(templates)

    count = 0
    for _tag, tmpl, keywords in templates:
        if count >= n:
            break
        # Find products matching some keywords
        matching = [
            p for p in products
            if any(kw in (p.get("title", "") + " " + (p.get("description") or "")).lower() for kw in keywords)
            and p.get("rating") and p["rating"] >= 3.5
        ]
        if len(matching) < 2:
            continue

        selected = rng.sample(matching, min(3, len(matching)))
        query = tmpl.replace("{category}", rng.choice(keywords[2:] if len(keywords) > 2 else keywords))

        candidates.append(EvalCandidate(
            query_id=f"sem-{count + 1:03d}",
            query=query,
            query_type="semantic",
            expected_product_ids=[p["product_id"] for p in selected],
            source_product_ids=[p["product_id"] for p in selected],
            generation_reason=f"Natural language '{_tag}' query with keyword match on: {', '.join(keywords)}",
            reviewer_status="pending",
        ))
        count += 1

    return candidates


def generate_attribute_queries(
    products: list[dict[str, Any]], rng: random.Random, n: int,
) -> list[EvalCandidate]:
    """Generate attribute-constrained queries: price, brand, rating, or product attributes.

    Programmatically verifies that constraints match the target products.
    """
    candidates: list[EvalCandidate] = []

    # Pattern 1: Brand filter
    brand_pool = pick_products(products, rng, n, with_brand=True, min_reviews=10)
    brands_seen: set[str] = set()
    for p in brand_pool:
        b = p.get("brand", "")
        if not b or b in brands_seen:
            continue
        brands_seen.add(b)
        candidates.append(EvalCandidate(
            query_id=f"attr-{len(candidates) + 1:03d}",
            query=f"{b} audio device",
            query_type="attribute",
            expected_product_ids=[p["product_id"]],
            expected_constraints={"preferred_brands": [b]},
            source_product_ids=[p["product_id"]],
            generation_reason=f"Brand filter: {b}",
            reviewer_status="pending",
        ))
        if len(candidates) >= n:
            break

    # Pattern 2: Max budget with verification
    budget_tiers = [20.0, 50.0, 100.0, 200.0]
    for budget in budget_tiers:
        if len(candidates) >= n:
            break
        budget_pool = pick_products(products, rng, 20, max_price=budget, min_reviews=5)
        if not budget_pool:
            continue
        p = rng.choice(budget_pool[:5])
        price = p.get("price", 0)
        if price is not None and price <= budget:
            candidates.append(EvalCandidate(
                query_id=f"attr-{len(candidates) + 1:03d}",
                query=f"electronics under ${budget:.0f}",
                query_type="attribute",
                expected_product_ids=[p["product_id"]],
                expected_constraints={"max_budget": budget},
                source_product_ids=[p["product_id"]],
                generation_reason=f"Max budget ${budget:.0f}, product price=${price:.2f}",
                reviewer_status="pending",
            ))

    # Pattern 3: Min rating
    for min_r in [4.0, 4.5]:
        if len(candidates) >= n:
            break
        pool = pick_products(products, rng, 10, min_rating=min_r, min_reviews=50)
        if not pool:
            continue
        p = rng.choice(pool[:3])
        candidates.append(EvalCandidate(
            query_id=f"attr-{len(candidates) + 1:03d}",
            query=f"highly rated electronics {min_r}+ stars",
            query_type="attribute",
            expected_product_ids=[p["product_id"]],
            expected_constraints={"min_rating": min_r},
            source_product_ids=[p["product_id"]],
            generation_reason=f"Min rating {min_r}, product rating={p.get('rating')}",
            reviewer_status="pending",
        ))

    return candidates[:n]


def generate_combined_queries(
    products: list[dict[str, Any]], rng: random.Random, n: int,
    product_index: dict[str, dict[str, Any]],
) -> list[EvalCandidate]:
    """Generate combined queries: product need + hard constraints together."""
    candidates: list[EvalCandidate] = []

    combos = [
        ("budget + brand", lambda p: {
            "max_budget": (p.get("price") or 0) * 1.2,
            "preferred_brands": [p.get("brand", "")] if p.get("brand") else [],
        }, "budget-friendly {brand} product under ${budget:.0f}"),
        ("rating + reviews", lambda p: {
            "min_rating": max(3.5, (p.get("rating") or 4.0) - 0.3),
        }, "well-reviewed product with many ratings"),
        ("brand + rating", lambda p: {
            "preferred_brands": [p.get("brand", "")] if p.get("brand") else [],
            "min_rating": max(3.5, (p.get("rating") or 4.0) - 0.2),
        }, "top-rated {brand} electronics"),
        ("budget + rating", lambda p: {
            "max_budget": (p.get("price") or 0) * 1.3,
            "min_rating": max(3.5, (p.get("rating") or 4.0) - 0.2),
        }, "quality electronics under ${budget:.0f} with good reviews"),
    ]

    pool = pick_products(products, rng, n * 3, min_reviews=10, with_brand=True)
    rng.shuffle(pool)
    rng.shuffle(combos)

    seen_ids: set[str] = set()
    count = 0
    for p in pool:
        if count >= n:
            break
        pid = p["product_id"]
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        combo = combos[count % len(combos)]
        constraints = combo[1](p)
        # Filter empty brand lists
        cleaned_constraints = {k: v for k, v in constraints.items() if v and v != []}
        if not cleaned_constraints:
            continue

        candidates.append(EvalCandidate(
            query_id=f"comb-{count + 1:03d}",
            query=f"{p.get('brand', '')} {p.get('title', '')[:6]}".strip()[:120],
            query_type="combined",
            expected_product_ids=[pid],
            expected_constraints=cleaned_constraints,
            source_product_ids=[pid],
            generation_reason=f"Combined constraints: {cleaned_constraints}",
            reviewer_status="pending",
        ))
        count += 1

    return candidates


def generate_hard_queries(
    products: list[dict[str, Any]], rng: random.Random, n: int,
    product_index: dict[str, dict[str, Any]],
) -> list[EvalCandidate]:
    """Generate hard queries: ambiguous phrasing, overlapping products, or no clear match."""
    candidates: list[EvalCandidate] = []

    # Pattern 1: Vague queries — multiple valid matches
    vague_templates = [
        "replacement charger for my device",
        "cable that works with everything",
        "something to improve my audio setup",
        "electronic accessory for travel",
        "adapter for home and office",
    ]
    for i, query in enumerate(vague_templates):
        matching = [
            p for p in products
            if any(kw in (p.get("title", "")).lower()
                   for kw in query.lower().split() if len(kw) > 3)
        ]
        matching = matching[:5]
        candidates.append(EvalCandidate(
            query_id=f"hard-{i + 1:03d}",
            query=query,
            query_type="hard",
            expected_product_ids=[p["product_id"] for p in matching] if matching else [],
            source_product_ids=[p["product_id"] for p in matching] if matching else [],
            generation_reason="Vague query with multiple possible valid matches",
            reviewer_status="pending",
        ))

    # Pattern 2: No obvious match — should return empty
    no_match_templates: list[tuple[str, list[str]]] = [
        ("pro audio mixer with XLR phantom power", []),
        ("quantum computing development kit USB-C", []),
        ("satellite TV receiver 8K HDR", []),
        ("medical grade EEG headset bluetooth", []),
        ("diesel generator transfer switch automatic", []),
    ]
    for j, (query, _) in enumerate(no_match_templates):
        candidates.append(EvalCandidate(
            query_id=f"hard-{len(vague_templates) + j + 1:03d}",
            query=query,
            query_type="hard",
            expected_product_ids=[],  # no_answer_candidate
            source_product_ids=[],
            generation_reason="No expected match in Electronics catalog — no_answer_candidate",
            reviewer_status="pending",
            reviewer_notes="Verify no relevant products exist in catalog",
        ))

    # Pattern 3: Contradictory constraints
    contra_templates = [
        ("premium flagship headphones under $15", {"max_budget": 15.0}),
        ("professional studio monitors under $20 with 1000+ reviews", {"max_budget": 20.0}),
    ]
    for k, (query, constraints) in enumerate(contra_templates):
        candidates.append(EvalCandidate(
            query_id=f"hard-{len(vague_templates) + len(no_match_templates) + k + 1:03d}",
            query=query,
            query_type="hard",
            expected_product_ids=[],
            expected_constraints=constraints,
            source_product_ids=[],
            generation_reason="Contradictory constraints — no_answer_candidate",
            reviewer_status="pending",
            reviewer_notes="Verify impossible to satisfy",
        ))

    return candidates[:n]


def generate_candidates(
    products: list[dict[str, Any]],
    total: int = 50,
    seed: int = 42,
) -> list[EvalCandidate]:
    """Generate eval candidates covering all 5 query types.

    Args:
        products: Product list.
        total: Total number of queries to generate.
        seed: Random seed.

    Returns:
        List of EvalCandidate objects.
    """
    rng = random.Random(seed)
    product_index = build_index(products)

    # Distribute across types
    per_type = max(6, total // 5)
    # Adjust so sum equals total
    distribution = [
        ("keyword", per_type),
        ("semantic", per_type),
        ("attribute", per_type),
        ("combined", per_type),
        ("hard", total - 4 * per_type),
    ]

    all_candidates: list[EvalCandidate] = []

    for qtype, count in distribution:
        logger.info("Generating %d %s queries...", count, qtype)
        if qtype == "keyword":
            cands = generate_keyword_queries(products, rng, count)
        elif qtype == "semantic":
            cands = generate_semantic_queries(products, rng, count)
        elif qtype == "attribute":
            cands = generate_attribute_queries(products, rng, count)
        elif qtype == "combined":
            cands = generate_combined_queries(products, rng, count, product_index)
        else:
            cands = generate_hard_queries(products, rng, count, product_index)

        # Assign sequential IDs
        for c in cands:
            all_candidates.append(c)

    # Re-number sequentially
    for i, c in enumerate(all_candidates, 1):
        c.query_id = f"{c.query_type[:4]}-{i:03d}"

    logger.info("Generated %d total candidates", len(all_candidates))
    return all_candidates[:total]


def validate_candidates(
    candidates: list[EvalCandidate],
    product_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate all candidates against the product index.

    Args:
        candidates: Generated candidates.
        product_index: Product ID -> product mapping.

    Returns:
        Validation result dict.
    """
    issues: list[str] = []
    seen_ids: set[str] = set()

    for c in candidates:
        # Unique query_id
        if c.query_id in seen_ids:
            issues.append(f"Duplicate query_id: {c.query_id}")
        seen_ids.add(c.query_id)

        # Non-empty query
        if not c.query or not c.query.strip():
            issues.append(f"Empty query: {c.query_id}")

        # Referenced products exist
        for pid in c.expected_product_ids:
            if pid not in product_index:
                issues.append(f"Invalid product_id in {c.query_id}: {pid}")

        for pid in c.source_product_ids:
            if pid not in product_index:
                issues.append(f"Invalid source_product_id in {c.query_id}: {pid}")

        # Type must be valid
        if c.query_type not in QUERY_TYPES:
            issues.append(f"Invalid query_type in {c.query_id}: {c.query_type}")

        # no-answer must not have expected_product_ids
        empty_expected = not c.expected_product_ids
        is_no_answer = "no_answer" in (c.reviewer_notes or "") or "no_answer" in c.generation_reason
        if is_no_answer and not empty_expected:
            issues.append(f"no_answer candidate {c.query_id} has expected_product_ids={c.expected_product_ids}")

        # Numeric constraint verification
        if c.expected_constraints:
            for pid in c.expected_product_ids:
                p = product_index.get(pid)
                if p is None:
                    continue
                if "max_budget" in c.expected_constraints:
                    price = p.get("price")
                    if price is not None and price > c.expected_constraints["max_budget"]:
                        issues.append(
                            f"Budget violation in {c.query_id}: {pid} price ${price:.2f} > ${c.expected_constraints['max_budget']:.2f}"
                        )
                if "min_rating" in c.expected_constraints:
                    rating = p.get("rating")
                    if rating is not None and rating < c.expected_constraints["min_rating"]:
                        issues.append(
                            f"Rating violation in {c.query_id}: {pid} rating {rating} < {c.expected_constraints['min_rating']}"
                        )

    return {
        "total_candidates": len(candidates),
        "issues_found": len(issues),
        "issues": issues,
        "valid": len(issues) == 0,
    }


def write_outputs(
    candidates: list[EvalCandidate],
    eval_dir: Path,
    reports_dir: Path,
    seed: int,
    source_path: str,
) -> None:
    """Write JSONL, CSV, and Markdown summary.

    Args:
        candidates: All candidates.
        eval_dir: Output directory for data files.
        reports_dir: Output directory for reports.
        seed: Random seed used.
        source_path: Path to input product data.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # JSONL
    jl_path = eval_dir / "eval_candidates.jsonl"
    with open(jl_path, "w", encoding="utf-8") as f:
        for c in candidates:
            d = {
                "query_id": c.query_id,
                "query": c.query,
                "query_type": c.query_type,
                "expected_product_ids": c.expected_product_ids,
                "expected_constraints": c.expected_constraints,
                "source_product_ids": c.source_product_ids,
                "generation_reason": c.generation_reason,
                "reviewer_status": c.reviewer_status,
                "reviewer_notes": c.reviewer_notes,
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("JSONL written to %s", jl_path)

    # CSV
    csv_path = eval_dir / "eval_review.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "query_type", "query", "expected_product_ids",
            "constraints", "generation_reason", "reviewer_status", "reviewer_notes",
        ])
        for c in candidates:
            writer.writerow([
                c.query_id, c.query_type, c.query,
                "|".join(c.expected_product_ids),
                json.dumps(c.expected_constraints) if c.expected_constraints else "",
                c.generation_reason,
                c.reviewer_status,
                c.reviewer_notes,
            ])
    logger.info("CSV written to %s", csv_path)

    # Markdown summary
    md_path = reports_dir / "eval_candidate_summary.md"
    type_counts: dict[str, int] = defaultdict(int)
    for c in candidates:
        type_counts[c.query_type] += 1

    no_answer = sum(1 for c in candidates if not c.expected_product_ids)

    lines = [
        "# Eval Candidate Summary",
        "",
        f"**Generated**: {datetime.now(UTC).isoformat()}",
        f"**Source data**: `{source_path}`",
        f"**Seed**: {seed}",
        f"**Total candidates**: {len(candidates)}",
        "",
        "## Query Type Distribution",
        "",
        "| Type | Count |",
        "|------|-------|",
    ]
    for qt in QUERY_TYPES:
        lines.append(f"| {qt} | {type_counts.get(qt, 0)} |")
    lines += [
        "",
        f"## No-answer Candidates: {no_answer}",
        "",
    ]

    lines += [
        "## Human Review Instructions",
        "",
        "### How to check if a query is natural",
        "- Would a real user type this into a search box or product filter?",
        "- Is the language natural (not just concatenated keywords or copied titles)?",
        "- If it sounds robotic or forced, mark as `revise` with suggested rewording.",
        "",
        "### How to check if expected_product_ids are correct",
        "- Look up each expected product in the source data.",
        "- Does the product genuinely match the query intent?",
        "- Does it satisfy all named constraints (brand, budget, rating)?",
        "- If the match is weak or wrong, remove that product_id or mark as `revise`.",
        "",
        "### How to fill reviewer_status",
        "- `approved`: Query is natural AND expected products are correct.",
        "- `revise`: Query OR expected products need minor changes (note what).",
        "- `rejected`: Query is bad (unrealistic, duplicates another, or products are wrong).",
        "",
        "### Why human review is mandatory",
        "- These queries are **auto-curated** from product metadata templates.",
        "- Expected product IDs are **programmatically assigned**, not judged by a person.",
        "- Without review, relevance labels are not trustworthy (circular evaluation risk).",
        "- Only after human review can this be called a 'gold' evaluation set.",
        "",
        "### Freezing the evaluation set",
        "1. Open `data/eval/eval_review.csv` in a spreadsheet editor.",
        "2. Review each row. Set `reviewer_status` to approved/revise/rejected.",
        "3. Add notes in `reviewer_notes` for revises and rejections.",
        "4. Save the CSV.",
        "5. Run the freeze script (future step) or manually filter to `approved` rows.",
        "6. Filtered approved rows become `data/eval/eval_queries.jsonl` — the frozen benchmark.",
        "",
        "---",
        f"*Generated by MarketLens eval candidate generator (seed={seed})*",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown written to %s", md_path)


def main() -> None:
    """Entry point for eval candidate generation."""
    parser = argparse.ArgumentParser(
        description="Generate candidate eval queries from product data"
    )
    parser.add_argument(
        "--products", type=Path, default=DEFAULT_PRODUCTS,
        help=f"Path to product JSON (default: {DEFAULT_PRODUCTS})",
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"Number of candidates to generate (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    if not args.products.exists():
        logger.error("Products file not found: %s", args.products)
        sys.exit(1)

    logger.info("Loading products from %s...", args.products)
    products = load_products(args.products)
    product_index = build_index(products)
    logger.info("Loaded %d products (%d unique IDs)", len(products), len(product_index))

    logger.info("Generating %d candidates (seed=%d)...", args.count, args.seed)
    candidates = generate_candidates(products, total=args.count, seed=args.seed)

    logger.info("Validating candidates...")
    validation = validate_candidates(candidates, product_index)
    if not validation["valid"]:
        logger.warning("Validation found %d issues:", validation["issues_found"])
        for issue in validation["issues"][:10]:
            logger.warning("  - %s", issue)
    else:
        logger.info("All candidates validated successfully")

    logger.info("Writing outputs...")
    write_outputs(
        candidates,
        DEFAULT_EVAL_DIR,
        REPORTS_DIR,
        seed=args.seed,
        source_path=str(args.products),
    )

    logger.info("=== Generation Summary ===")
    logger.info("  Total candidates: %d", len(candidates))
    type_counts: dict[str, int] = defaultdict(int)
    for c in candidates:
        type_counts[c.query_type] += 1
    for qt in QUERY_TYPES:
        logger.info("  %s: %d", qt, type_counts.get(qt, 0))


if __name__ == "__main__":
    main()
