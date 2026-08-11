#!/usr/bin/env python3
"""Audit real MarketLens product data for quality issues.

Reads the cleaned product JSON, computes statistics, and writes
both a JSON report and a Markdown report.

Usage:
  uv run python scripts/audit_electronics_data.py
  uv run python scripts/audit_electronics_data.py --input data/processed/electronics_2000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/processed/electronics_2000.json")
REPORTS_DIR = Path("reports")


def load_products(path: Path) -> list[dict[str, Any]]:
    """Load product list from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        List of product dicts.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of products")
    return data


def compute_missing_rate(products: list[dict[str, Any]], field: str) -> float:
    """Fraction of products where field is None, empty, or missing.

    Args:
        products: Product list.
        field: Field name.

    Returns:
        Missing rate [0, 1].
    """
    if not products:
        return 0.0
    count = sum(
        1 for p in products
        if field not in p or p[field] is None or p[field] == ""
    )
    return count / len(products)


def compute_stats(values: list[Any]) -> dict[str, float]:
    """Compute distribution stats for numeric values.

    Args:
        values: List of numeric values (may contain None).

    Returns:
        Dict with count, null_count, min, max, mean, p50, p95.
    """
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return {
            "count": len(values), "null_count": len(values),
            "min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0,
        }
    cleaned_sorted = sorted(cleaned)
    n = len(cleaned_sorted)
    return {
        "count": len(values),
        "null_count": len(values) - len(cleaned),
        "min": cleaned_sorted[0],
        "max": cleaned_sorted[-1],
        "mean": sum(cleaned_sorted) / n,
        "p50": cleaned_sorted[n // 2],
        "p95": cleaned_sorted[int(n * 0.95)],
    }


def run_audit(input_path: Path) -> dict[str, Any]:
    """Run full audit on the product data.

    Args:
        input_path: Path to the product JSON.

    Returns:
        Audit report dict.
    """
    products = load_products(input_path)

    # Basic counts
    total = len(products)
    unique_ids = len({p["product_id"] for p in products})

    # Missing rates
    missing = {}
    for field in ["title", "description", "brand", "price", "rating", "review_count"]:
        missing[field] = round(compute_missing_rate(products, field), 4)

    # Numeric distributions
    prices = [p.get("price") for p in products]
    ratings = [p.get("rating") for p in products]
    reviews = [p.get("review_count") for p in products]

    price_stats = compute_stats(prices)
    rating_stats = compute_stats(ratings)
    review_stats = compute_stats(reviews)

    # Title analysis
    titles = [p.get("title", "") for p in products]
    title_dupes = sum(1 for t, c in Counter(titles).items() if c > 1)
    title_lengths = [len(t) for t in titles]
    title_len_stats = compute_stats(title_lengths)

    # Brand analysis
    brands = [p.get("brand") for p in products if p.get("brand")]
    brand_counts = Counter(brands)
    top_brands = brand_counts.most_common(20)

    # Suspicious titles (too short, special chars, generic)
    suspicious: list[dict[str, Any]] = []
    for p in products:
        t = p.get("title", "")
        reasons = []
        if len(t) < 10:
            reasons.append("very_short")
        if t.count("Generic") > 0 and "Replacement" in t:
            reasons.append("generic_replacement")
        if len(t) > 300:
            reasons.append("very_long")
        if not t:
            reasons.append("empty")
        if reasons:
            suspicious.append({"product_id": p["product_id"], "title": t[:200], "reasons": reasons})

    # Sample products
    sample = products[:10]

    report = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "basic_counts": {
            "total_products": total,
            "unique_product_ids": unique_ids,
            "ids_are_unique": unique_ids == total,
        },
        "missing_rates": missing,
        "distributions": {
            "price": price_stats,
            "rating": rating_stats,
            "review_count": review_stats,
            "title_length": title_len_stats,
        },
        "title_analysis": {
            "title_duplicate_count": title_dupes,
            "distinct_titles": len(set(titles)),
        },
        "brand_analysis": {
            "distinct_brands": len(brand_counts),
            "top_brands": [{"brand": b, "count": c} for b, c in top_brands],
        },
        "suspicious_titles_count": len(suspicious),
        "suspicious_titles": suspicious[:30],
        "sample_products": sample[:10],
    }
    return report


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    """Write audit results as Markdown.

    Args:
        report: Audit report dict.
        path: Output markdown file path.
    """
    bc = report["basic_counts"]
    mr = report["missing_rates"]
    dist = report["distributions"]
    ta = report["title_analysis"]
    ba = report["brand_analysis"]

    lines = [
        "# MarketLens Data Quality Report",
        "",
        f"**Generated**: {report['audit_timestamp']}",
        f"**Source**: `{report['input_file']}`",
        "",
        "## Basic Counts",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total products | {bc['total_products']} |",
        f"| Unique product IDs | {bc['unique_product_ids']} |",
        f"| IDs all unique | {bc['ids_are_unique']} |",
        "",
        "## Field Completeness",
        "",
        "| Field | Missing Rate |",
        "|-------|-------------|",
    ]
    for field, rate in mr.items():
        lines.append(f"| {field} | {rate:.2%} |")

    lines += [
        "",
        "## Distributions",
        "",
        "### Price (USD)",
        "",
        _format_dist_table(dist["price"]),
        "",
        "### Rating",
        "",
        _format_dist_table(dist["rating"]),
        "",
        "### Review Count",
        "",
        _format_dist_table(dist["review_count"]),
        "",
        "### Title Length (characters)",
        "",
        _format_dist_table(dist["title_length"]),
        "",
        "## Title Analysis",
        "",
        f"- Duplicate titles: {ta['title_duplicate_count']}",
        f"- Distinct titles: {ta['distinct_titles']}",
        "",
        "## Top 20 Brands",
        "",
        "| Brand | Count |",
        "|-------|-------|",
    ]
    for entry in ba["top_brands"]:
        lines.append(f"| {entry['brand']} | {entry['count']} |")

    lines += [
        "",
        f"## Suspicious Titles ({report['suspicious_titles_count']} total, showing first 30)",
        "",
    ]
    for s in report["suspicious_titles"][:30]:
        lines.append(f"- `{s['product_id']}`: {s['title'][:100]} ({', '.join(s['reasons'])})")

    lines += [
        "",
        "## Sample Products (first 10)",
        "",
    ]
    for sp in report["sample_products"]:
        lines.append(f"- **{sp['product_id']}** | {sp.get('brand','?')} | ${sp.get('price','?')} | {sp['title'][:80]}")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_dist_table(stats: dict[str, float]) -> str:
    """Format a distribution stats dict as a Markdown table row."""
    return (
        f"| Count | {stats['count']} |\n"
        f"| Null | {stats['null_count']} |\n"
        f"| Min | {stats['min']:.2f} |\n"
        f"| Max | {stats['max']:.2f} |\n"
        f"| Mean | {stats['mean']:.2f} |\n"
        f"| P50 | {stats['p50']:.2f} |\n"
        f"| P95 | {stats['p95']:.2f} |"
    )


def main() -> None:
    """Entry point for data quality audit."""
    parser = argparse.ArgumentParser(description="Audit MarketLens product data quality")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Path to product JSON (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    logger.info("Auditing %s...", args.input)
    report = run_audit(args.input)

    # Write JSON report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "data_quality_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON report written to %s", json_path)

    # Write Markdown report
    md_path = REPORTS_DIR / "data_quality_report.md"
    write_markdown_report(report, md_path)
    logger.info("Markdown report written to %s", md_path)

    # Summary to stdout
    logger.info("=== Audit Summary ===")
    logger.info("  Products: %d (unique IDs: %s)", report["basic_counts"]["total_products"], report["basic_counts"]["unique_product_ids"])
    logger.info("  Missing rates: title=%.1f%%, price=%.1f%%, rating=%.1f%%", report["missing_rates"]["title"] * 100, report["missing_rates"]["price"] * 100, report["missing_rates"]["rating"] * 100)
    logger.info("  Brands: %d distinct", report["brand_analysis"]["distinct_brands"])
    logger.info("  Suspicious titles: %d", report["suspicious_titles_count"])


if __name__ == "__main__":
    main()
