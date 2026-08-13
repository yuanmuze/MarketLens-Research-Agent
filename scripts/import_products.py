#!/usr/bin/env python3
"""Import products from JSON to PostgreSQL with idempotent upsert.

Usage:
  uv run python scripts/import_products.py --input data/processed/electronics_2000.json
  uv run python scripts/import_products.py --input <file> --dry-run

Requires MARKETLENS_DATABASE_URL pointing to a PostgreSQL database
(or SQLite for local testing). Requires alembic upgrade head first.

Idempotent: re-running the same file does NOT duplicate rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Import product JSON into the database."""
    parser = argparse.ArgumentParser(description="Import products to PostgreSQL")
    parser.add_argument("--input", type=Path, required=True, help="Product JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        raw = json.load(f)

    from marketlens.models import Product
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import ProductRepository

    # Validate all products first (single invalid row should not silently fail)
    products: list[Product] = []
    invalid = 0
    for i, item in enumerate(raw):
        try:
            products.append(Product(**item))
        except Exception as e:
            invalid += 1
            logger.warning("Invalid product at index %d: %s", i, e)

    logger.info("Valid: %d, Invalid: %d (of %d total)", len(products), invalid, len(raw))

    if args.dry_run:
        logger.info("Dry run — no data written.")
        return

    with session_scope() as session:
        repo = ProductRepository(session)
        result = repo.upsert_many(products)
        session.commit()

    logger.info(
        "Import complete: inserted=%d, updated=%d, unchanged=%d, failed=%d",
        result["inserted"], result["updated"], result["unchanged"], result["failed"],
    )


if __name__ == "__main__":
    main()
