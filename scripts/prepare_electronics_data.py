#!/usr/bin/env python3
"""Prepare a small sample of Amazon Reviews 2023 Electronics metadata.

Streams from the official UCSD JSONL source via HuggingFace datasets
(json builder, no remote script execution) or reads from a local file.
Produces a validated, deduplicated JSON file suitable for
ProductCatalog.from_json().

Compatible with datasets >= 5.0 (no trust_remote_code, no dataset scripts).

Usage:
  # Default: ~2000 products, seed 42
  uv run python scripts/prepare_electronics_data.py

  # Custom size
  uv run python scripts/prepare_electronics_data.py --max-products 5000 --seed 123

  # Dry run (validate only, no output)
  uv run python scripts/prepare_electronics_data.py --dry-run

  # From local file
  uv run python scripts/prepare_electronics_data.py --local-file /path/to/data.jsonl

Output:
  data/processed/electronics_products.json
  data/processed/electronics_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constant seed for reproducibility
DEFAULT_SEED = 42
DEFAULT_MAX_PRODUCTS = 2000
MAX_PRODUCTS_LIMIT = 5000
SHUFFLE_BUFFER_SIZE = 10000

# Official UCSD URL for Amazon Reviews 2023 Electronics metadata
# Source: https://mcauleylab.ucsd.edu/public_datasets/
OFFICIAL_METADATA_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories/meta_Electronics.jsonl.gz"
)

# --- Field mapping from Amazon metadata to Product model ---
# Amazon metadata schema (v2023):
#   parent_asin, title, price, rating_number, average_rating,
#   store, main_category, description, features, details, images, ...
REQUIRED_FIELDS = {"parent_asin", "title"}
OPTIONAL_FIELDS = {
    "price", "average_rating", "rating_number",
    "store", "main_category", "description", "features", "details", "images",
}


def clean_price(raw: Any) -> float | None:
    """Convert raw price to float. Returns None for missing/invalid values."""
    if raw is None:
        return None
    try:
        val = float(raw)
        if val < 0:
            return None
        return round(val, 2)
    except (ValueError, TypeError):
        if isinstance(raw, str):
            # Try parsing "$XX.XX" format
            try:
                return round(float(raw.replace("$", "").replace(",", "").strip()), 2)
            except (ValueError, TypeError):
                pass
        return None


def clean_rating(raw: Any) -> float | None:
    """Convert raw rating to float 0-5. Returns None for invalid values."""
    if raw is None:
        return None
    try:
        val = float(raw)
        if 0 <= val <= 5:
            return round(val, 1)
        return None
    except (ValueError, TypeError):
        return None


def clean_review_count(raw: Any) -> int | None:
    """Convert raw rating_number to int. Returns None for invalid."""
    if raw is None:
        return None
    try:
        val = int(raw)
        return val if val >= 0 else None
    except (ValueError, TypeError):
        return None


def build_product(item: dict[str, Any]) -> dict[str, Any]:
    """Convert an Amazon metadata row to a MarketLens Product dict.

    Args:
        item: Raw metadata row from Amazon Reviews 2023.

    Returns:
        Product-compatible dict, or empty dict if required fields missing.
    """
    parent_asin = str(item.get("parent_asin", "")).strip()
    title = str(item.get("title", "")).strip()

    if not parent_asin or not title:
        return {}

    # Build attributes from features/details
    attributes: dict[str, str] = {}
    features = item.get("features")
    if isinstance(features, list):
        for feat in features[:10]:
            if isinstance(feat, str) and feat.strip():
                attributes[f"feature_{len(attributes)}"] = feat.strip()[:200]
    details = item.get("details")
    if isinstance(details, dict):
        for k, v in details.items():
            if isinstance(k, str) and isinstance(v, str):
                attributes[str(k).strip()[:100]] = str(v).strip()[:200]

    # Build description from description + features
    description = str(item.get("description") or "")
    if not description and isinstance(features, list):
        description = " | ".join(
            f for f in features[:5] if isinstance(f, str) and f.strip()
        )
    description = description[:1000] if description else ""

    # Handle images
    images: list[str] = []
    raw_images = item.get("images")
    if isinstance(raw_images, list):
        images = [
            str(img) for img in raw_images[:5]
            if isinstance(img, str) and img.startswith("http")
        ]

    return {
        "product_id": parent_asin,
        "title": title[:500],
        "brand": str(item.get("store", "")).strip()[:200] or None,
        "category": "electronics",
        "price": clean_price(item.get("price")),
        "rating": clean_rating(item.get("average_rating")),
        "review_count": clean_review_count(item.get("rating_number")),
        "attributes": attributes,
        "description": description,
        "images": images,
        "url": f"https://amazon.com/dp/{parent_asin}",
    }


def generate_manifest(
    args: argparse.Namespace,
    raw_count: int,
    cleaned_count: int,
    skip_stats: dict[str, int],
    output_path: Path,
    elapsed_s: float,
) -> dict[str, Any]:
    """Generate a data manifest with provenance info."""
    sha256 = ""
    if output_path.exists():
        sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

    # Detect datasets version if available
    datasets_version = "unknown"
    try:
        import datasets
        datasets_version = datasets.__version__
    except ImportError:
        pass

    return {
        "source": "Amazon Reviews 2023 Electronics metadata",
        "source_url": OFFICIAL_METADATA_URL,
        "source_type": "UCSD official (.jsonl.gz) via HuggingFace datasets json builder",
        "category": "Electronics",
        "datasets_version": datasets_version,
        "streaming": True,
        "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_products_target": args.max_products,
        "raw_products_read": raw_count,
        "cleaned_products_output": cleaned_count,
        "skip_reasons": skip_stats,
        "output_file": str(output_path),
        "output_sha256": sha256,
        "python_version": sys.version,
        "processing_time_s": round(elapsed_s, 2),
    }


def _stream_jsonl_via_datasets(
    max_products: int, seed: int,
) -> list[dict[str, Any]]:
    """Stream via datasets json builder (preferred path)."""
    from datasets import load_dataset

    logger.info(
        "Trying datasets json builder (url=%s, max=%d, seed=%d, buffer=%d)...",
        OFFICIAL_METADATA_URL, max_products, seed, SHUFFLE_BUFFER_SIZE,
    )

    ds = load_dataset(
        "json",
        data_files={"train": OFFICIAL_METADATA_URL},
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE)

    items: list[dict[str, Any]] = []
    for i, item in enumerate(ds):
        if isinstance(item, dict) and ("parent_asin" in item or "title" in item):
            items.append(item)
        if len(items) >= max_products:
            break
        if (i + 1) % 500 == 0:
            logger.info("  Streamed %d items...", i + 1)
    return items


def _stream_jsonl_direct(
    max_products: int, seed: int,
) -> list[dict[str, Any]]:
    """Stream via direct HTTP + gzip, parsing JSON lines manually.

    Avoids datasets schema inference entirely. Successfully handles
    mixed-type fields (null vs struct) in the official JSONL.
    """
    import gzip
    import random
    from io import BytesIO

    try:
        import httpx
    except ImportError:
        logger.error("httpx is required for direct streaming. Install with: pip install httpx")
        return []

    logger.info(
        "Streaming directly from %s (max=%d, seed=%d)...",
        OFFICIAL_METADATA_URL, max_products, seed,
    )

    items: list[dict[str, Any]] = []
    try:
        with httpx.stream("GET", OFFICIAL_METADATA_URL, timeout=60.0, follow_redirects=True) as resp:
            resp.raise_for_status()
            # Wrap in BytesIO for gzip streaming decompression
            decompressor = gzip.GzipFile(fileobj=BytesIO(resp.read()))
            for line in decompressor:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and ("parent_asin" in item or "title" in item):
                        items.append(item)
                except json.JSONDecodeError:
                    continue
                if len(items) >= max_products * 3:  # Read more for shuffle
                    break
    except Exception as e:
        logger.warning("Direct streaming failed: %s", e)

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(items)
    result = items[:max_products]
    logger.info("Loaded %d raw items via direct streaming", len(result))
    return result


def load_from_huggingface(max_products: int, seed: int) -> list[dict[str, Any]]:
    """Stream product metadata from the official UCSD JSONL source.

    Tries two approaches (in order):
    1. datasets json builder (streaming, no trust_remote_code)
    2. Direct HTTP + gzip + manual JSON parsing

    Streaming ensures only the required number of lines are decompressed
    and parsed — the full ~10 GB archive is never downloaded.

    Args:
        max_products: Maximum number of products to extract.
        seed: Random seed for shuffling.

    Returns:
        List of raw metadata dicts.
    """
    try:
        import datasets  # noqa: F401
    except ImportError:
        logger.error(
            "datasets library not installed. Install with: pip install datasets"
        )
        return []

    # Approach 1: datasets json builder
    try:
        result = _stream_jsonl_via_datasets(max_products, seed)
        if result:
            logger.info("Loaded %d raw items via datasets json builder", len(result))
            return result
    except Exception as e:
        logger.warning("datasets json builder failed: %s", e)

    # Approach 2: direct HTTP + gzip (fallback for schema issues)
    logger.info("Falling back to direct HTTP + gzip streaming...")
    result = _stream_jsonl_direct(max_products, seed)
    if result:
        return result

    logger.error(
        "All streaming approaches failed. Download manually from\n"
        "  %s\n"
        "and use --local-file <path>",
        OFFICIAL_METADATA_URL,
    )
    return []


def load_from_local(path: Path, max_products: int, seed: int) -> list[dict[str, Any]]:
    """Load product metadata from a local JSONL file.

    Args:
        path: Path to local JSONL file.
        max_products: Maximum number of products.
        seed: Random seed for shuffling.

    Returns:
        List of raw metadata dicts.
    """
    import random

    logger.info("Loading from local file: %s", path)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
            if len(items) >= max_products * 10:  # Read more than needed for shuffle
                break

    random.seed(seed)
    random.shuffle(items)
    return items[:max_products]


def main() -> None:
    """Entry point for the Amazon Electronics data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Prepare Amazon Electronics product data for MarketLens"
    )
    parser.add_argument(
        "--max-products", type=int, default=DEFAULT_MAX_PRODUCTS,
        help=f"Maximum products to output (default: {DEFAULT_MAX_PRODUCTS}, max: {MAX_PRODUCTS_LIMIT})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--local-file", type=Path, default=None,
        help="Path to local JSONL file (if not using HuggingFace)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/electronics_products.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate pipeline without writing output",
    )
    args = parser.parse_args()

    # Enforce limits
    if args.max_products < 1:
        logger.error("max-products must be >= 1")
        sys.exit(1)
    if args.max_products > MAX_PRODUCTS_LIMIT:
        logger.warning(
            "Capping max_products at %d (requested %d)",
            MAX_PRODUCTS_LIMIT, args.max_products,
        )
        args.max_products = MAX_PRODUCTS_LIMIT

    t0 = time.monotonic()

    # Step 1: Load data
    if args.local_file:
        raw_items = load_from_local(args.local_file, args.max_products, args.seed)
    else:
        raw_items = load_from_huggingface(args.max_products, args.seed)

    raw_count = len(raw_items)
    if raw_count == 0:
        logger.error("No data loaded. Try --local-file or install 'datasets' package.")
        sys.exit(1)

    # Step 2: Clean and validate
    products: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    skip_stats: dict[str, int] = {
        "missing_required_fields": 0,
        "duplicate_id": 0,
        "empty_title": 0,
        "invalid_price": 0,
        "total_skipped": 0,
    }

    for item in raw_items:
        product = build_product(item)
        if not product:
            skip_stats["missing_required_fields"] += 1
            skip_stats["total_skipped"] += 1
            continue

        pid = product["product_id"]
        if pid in seen_ids:
            skip_stats["duplicate_id"] += 1
            skip_stats["total_skipped"] += 1
            continue

        if not product["title"]:
            skip_stats["empty_title"] += 1
            skip_stats["total_skipped"] += 1
            continue

        seen_ids.add(pid)
        products.append(product)

        if len(products) >= args.max_products:
            break

    # Step 3: Write output
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        logger.info(
            "Wrote %d products to %s (%.1f KB)",
            len(products),
            args.output,
            args.output.stat().st_size / 1024,
        )

        # Generate manifest
        elapsed = time.monotonic() - t0
        manifest = generate_manifest(
            args, raw_count, len(products), skip_stats, args.output, elapsed,
        )
        manifest_path = args.output.parent / "electronics_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info("Manifest written to %s", manifest_path)

        # Summary
        logger.info("=== Pipeline Summary ===")
        logger.info("  Raw items read:     %d", raw_count)
        logger.info("  Valid products:     %d", len(products))
        logger.info("  Duplicates skipped: %d", skip_stats["duplicate_id"])
        logger.info("  Missing fields:     %d", skip_stats["missing_required_fields"])
        logger.info("  Total skipped:      %d", skip_stats["total_skipped"])
        logger.info("  Processing time:    %.1f s", elapsed)
        logger.info("  Output file:        %s", args.output)
        logger.info("  Seed:               %d", args.seed)
    else:
        elapsed = time.monotonic() - t0
        logger.info("=== DRY RUN ===")
        logger.info("  Would write %d products to %s", len(products), args.output)
        logger.info("  Raw: %d, Skipped: %d, Time: %.1fs", raw_count, skip_stats["total_skipped"], elapsed)


if __name__ == "__main__":
    main()
