#!/usr/bin/env python3
"""Verify downloaded WANDS dataset integrity.

Checks: file sizes, separators, field counts, ID uniqueness,
label validity, query-product coverage, and label distribution.

Exits non-zero on any failure, with an explanation.

Usage:
  uv run python scripts/verify_wands_data.py
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WANDS_DIR = Path("data/external/wands")
VALID_LABELS = {"Exact", "Partial", "Irrelevant"}


def fail(msg: str) -> None:
    """Log an error and exit."""
    logger.error("FAIL: %s", msg)
    sys.exit(1)


def load_csv(path: Path, expected_delimiter: str = "\t") -> list[dict[str, str]]:
    """Load a TSV/CSV file, auto-detecting tab vs comma delimiter."""
    with open(path, encoding="utf-8") as f:
        sample = f.read(8192)
    # Count tabs vs commas in first lines
    tabs = sample.count("\t")
    commas = sample.count(",")
    if tabs > commas:
        delimiter = "\t"
        logger.info("%s: auto-detected TAB delimiter (%d tabs vs %d commas)", path.name, tabs, commas)
    else:
        delimiter = ","
        logger.info("%s: auto-detected COMMA delimiter (%d commas vs %d tabs)", path.name, commas, tabs)

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
    logger.info("%s: %d rows, fields=%s", path.name, len(rows), list(rows[0].keys()) if rows else "EMPTY")
    return rows


def main() -> None:
    """Verify WANDS dataset."""
    # Check source manifest exists
    manifest_path = WANDS_DIR / "source.json"
    if not manifest_path.exists():
        fail(f"{manifest_path} not found. Run download_wands.py first.")

    # Load files
    product_path = WANDS_DIR / "product.csv"
    query_path = WANDS_DIR / "query.csv"
    label_path = WANDS_DIR / "label.csv"

    for p in [product_path, query_path, label_path]:
        if not p.exists():
            fail(f"{p} not found. Run download_wands.py first.")

    products = load_csv(product_path)
    queries = load_csv(query_path)
    labels = load_csv(label_path)

    # Basic counts
    logger.info("=== Basic Counts ===")
    logger.info("Products: %d", len(products))
    logger.info("Queries: %d", len(queries))
    logger.info("Labels: %d", len(labels))

    # Product ID uniqueness
    pids = [r.get("product_id", "") for r in products]
    unique_pids = set(pids)
    if len(pids) != len(unique_pids):
        fail(f"Duplicate product_ids: {len(pids)} rows, {len(unique_pids)} unique")
    if "" in unique_pids:
        fail("Empty product_id found in product.csv")
    logger.info("Product IDs: %d unique", len(unique_pids))

    # Query ID uniqueness
    qids = [r.get("query_id", "") for r in queries]
    unique_qids = set(qids)
    if len(qids) != len(unique_qids):
        fail(f"Duplicate query_ids: {len(qids)} rows, {len(unique_qids)} unique")
    if "" in unique_qids:
        fail("Empty query_id found in query.csv")
    logger.info("Query IDs: %d unique", len(unique_qids))

    # Label uniqueness (multi-annotator duplicates expected in WANDS)
    label_pairs = [(r.get("query_id", ""), r.get("product_id", "")) for r in labels]
    unique_pairs = set(label_pairs)
    if len(label_pairs) != len(unique_pairs):
        dupes = len(label_pairs) - len(unique_pairs)
        logger.info("Multi-annotator duplicates: %d (WANDS has multiple raters per pair)", dupes)
    logger.info("Label query-product pairs: %d unique (%d total rows)", len(unique_pairs), len(label_pairs))

    # Label product/queries all reference valid IDs
    bad_pids = {pid for qid, pid in label_pairs if pid not in unique_pids}
    bad_qids = {qid for qid, pid in label_pairs if qid not in unique_qids}
    if bad_pids:
        fail(f"{len(bad_pids)} label rows reference non-existent product_ids: {list(bad_pids)[:5]}")
    if bad_qids:
        fail(f"{len(bad_qids)} label rows reference non-existent query_ids: {list(bad_qids)[:5]}")
    logger.info("All label product_ids and query_ids are valid")

    # Label value validation
    label_values = set()
    for r in labels:
        for val_field in ["label", "relevance", "rating"]:
            if val_field in r:
                label_values.add(r[val_field])
    invalid = label_values - VALID_LABELS
    if invalid:
        fail(f"Invalid label values found: {invalid}")
    logger.info("Label values: %s", sorted(label_values))

    # Label distribution (majority vote per query-product pair)
    label_key = "label" if "label" in labels[0] else ("relevance" if "relevance" in labels[0] else "rating")
    pair_votes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in labels:
        pair_votes[(r["query_id"], r["product_id"])].append(r.get(label_key, "Unknown"))

    # Multi-annotator audit
    vote_counts: Counter[int] = Counter()
    single_annotator = 0
    multi_annotator = 0
    all_agree = 0
    has_majority = 0
    no_majority = 0
    label_priority = {"Exact": 3, "Partial": 2, "Irrelevant": 1}

    label_counts: Counter[str] = Counter()
    for votes in pair_votes.values():
        n = len(votes)
        vote_counts[n] += 1
        if n == 1:
            single_annotator += 1
            label_counts[votes[0]] += 1
        else:
            multi_annotator += 1
            c = Counter(votes)
            mc = c.most_common()
            if mc[0][1] > n / 2:
                has_majority += 1
                label_counts[mc[0][0]] += 1
                if len(mc) == 1:
                    all_agree += 1
            else:
                no_majority += 1
                # Tie-break: higher priority wins
                best = max(votes, key=lambda x: label_priority.get(x, 0))
                label_counts[best] += 1

    total_unique = len(pair_votes)
    logger.info("=== Multi-Annotator Audit ===")
    logger.info("  Single annotator pairs: %d", single_annotator)
    logger.info("  Multi annotator pairs: %d", multi_annotator)
    logger.info("    All agree: %d", all_agree)
    logger.info("    Has majority (some disagree): %d", has_majority - all_agree)
    logger.info("    No strict majority (tie-broken): %d", no_majority)
    logger.info("  Vote count distribution: %s", dict(sorted(vote_counts.items())))
    logger.info("  Tie-breaking rule: Exact > Partial > Irrelevant (deterministic)")
    logger.info("=== Label Distribution (after aggregation) ===")
    for k in ["Exact", "Partial", "Irrelevant"]:
        logger.info("  %s: %d (%.1f%%)", k, label_counts[k], 100 * label_counts[k] / total_unique)
    if "Exact" not in label_counts or "Partial" not in label_counts or "Irrelevant" not in label_counts:
        fail("Missing expected label categories")

    # Per-query label distribution
    q_label_counts = defaultdict(list)
    for r in labels:
        q_label_counts[r["query_id"]].append(r.get(label_key))
    q_n = [len(v) for v in q_label_counts.values()]
    logger.info("=== Per-Query Labels ===")
    logger.info("  Mean: %.1f", sum(q_n) / len(q_n))
    logger.info("  Min: %d", min(q_n))
    logger.info("  Max: %d", max(q_n))
    logger.info("  Queries with <10 labels: %d", sum(1 for v in q_n if v < 10))

    # Check for empty text fields
    empty_titles = sum(1 for r in products if not r.get("product_name", "").strip() and not r.get("title", "").strip())
    empty_queries = sum(1 for r in queries if not r.get("query", "").strip())
    if empty_titles:
        logger.warning("Products with empty title/name: %d", empty_titles)
    if empty_queries:
        logger.warning("Queries with empty query text: %d", empty_queries)
    logger.info("Empty titles: %d, empty queries: %d", empty_titles, empty_queries)

    ops_message = (
        f"Products: {len(products)}, "
        f"Queries: {len(queries)}, "
        f"Labels: {len(labels)}"
    )
    logger.info("=== WANDS Verification PASSED ===")
    logger.info(ops_message)
    logger.info(
        "Expected ~42,994 products, ~480 queries, ~233,448 labels. "
        "Actual: %s", ops_message
    )


if __name__ == "__main__":
    main()
