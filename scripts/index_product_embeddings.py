#!/usr/bin/env python3
"""Validate a real embedding cache and index it into PostgreSQL/pgvector.

The API never generates a full catalog index during startup. Run this command
after migrations and product import, before enabling the pgvector semantic
backend.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marketlens.retrieval.pgvector_retriever import PGVECTOR_DIMENSION
from marketlens.retrieval.service import (
    _cache_metadata_path,
    _compute_data_hash,
    _embedding_cache_path,
)

logger = logging.getLogger(__name__)

DEFAULT_PRODUCTS = Path("data/processed/electronics_2000.json")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EXPECTED_COUNT = 2000


@dataclass(frozen=True)
class ValidatedIndexInputs:
    """Validated product IDs, vector matrix, and cache metadata."""

    product_ids: list[str]
    vectors: np.ndarray
    metadata: dict[str, Any]
    cache_path: Path


def _load_products(path: Path, expected_count: int) -> list[str]:
    """Load and validate ordered, unique product IDs."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("products input must be a JSON list")
    product_ids = [str(item.get("product_id", "")) for item in raw]
    if len(product_ids) != expected_count:
        raise ValueError(
            f"expected {expected_count} products, found {len(product_ids)}"
        )
    if any(not product_id for product_id in product_ids):
        raise ValueError("products input contains an empty product_id")
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("products input contains duplicate product_id values")
    return product_ids


def validate_index_inputs(
    products_path: Path,
    cache_path: Path,
    model_name: str,
    expected_count: int,
) -> ValidatedIndexInputs:
    """Validate source hash, paired metadata, dtype, shape, model, and dimension."""
    if not products_path.is_file():
        raise FileNotFoundError(f"products file not found: {products_path}")
    if not cache_path.is_file():
        raise FileNotFoundError(f"embedding cache not found: {cache_path}")
    metadata_path = _cache_metadata_path(cache_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"embedding metadata not found: {metadata_path}")

    product_ids = _load_products(products_path, expected_count)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_hash = _compute_data_hash(products_path)

    expected_metadata = {
        "model_name": model_name,
        "dim": PGVECTOR_DIMENSION,
        "count": expected_count,
        "dtype": "float32",
        "data_sha256": source_hash,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r}, expected {value!r}"
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError("embedding metadata mismatch: " + "; ".join(mismatches))

    expected_cache = _embedding_cache_path(
        products_path,
        model_name,
        expected_count,
        PGVECTOR_DIMENSION,
    )
    if cache_path.name != expected_cache.name:
        raise ValueError(
            f"cache fingerprint mismatch: expected paired cache {expected_cache.name}"
        )

    vectors = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (expected_count, PGVECTOR_DIMENSION):
        raise ValueError(
            f"embedding shape {vectors.shape} != "
            f"({expected_count}, {PGVECTOR_DIMENSION})"
        )
    if vectors.dtype != np.float32:
        raise ValueError(f"embedding dtype {vectors.dtype} != float32")

    return ValidatedIndexInputs(
        product_ids=product_ids,
        vectors=vectors,
        metadata=metadata,
        cache_path=cache_path,
    )


def index_embeddings(
    inputs: ValidatedIndexInputs,
    model_name: str,
    batch_size: int,
) -> dict[str, int]:
    """Write vectors in batches and commit once, then verify index coverage."""
    from marketlens.persistence.engine import get_engine, session_scope
    from marketlens.persistence.models import ProductRecord
    from marketlens.persistence.repositories import ProductEmbeddingRepository

    engine = get_engine()
    if engine.dialect.name != "postgresql":
        raise RuntimeError("embedding indexing requires PostgreSQL with pgvector")

    totals = {"inserted": 0, "updated": 0, "unchanged": 0}
    with session_scope() as session:
        existing_ids = {
            str(product_id)
            for (product_id,) in session.query(ProductRecord.product_id).filter(
                ProductRecord.product_id.in_(inputs.product_ids)
            ).all()
        }
        missing_products = set(inputs.product_ids) - existing_ids
        if missing_products:
            raise ValueError(
                f"database is missing {len(missing_products)} products; import products first"
            )

        repo = ProductEmbeddingRepository(session)
        for start in range(0, len(inputs.product_ids), batch_size):
            end = min(start + batch_size, len(inputs.product_ids))
            result = repo.upsert_many(
                inputs.product_ids[start:end],
                inputs.vectors[start:end].tolist(),
                model_name,
                PGVECTOR_DIMENSION,
            )
            for key in totals:
                totals[key] += result[key]
        session.commit()

    with session_scope() as session:
        status = ProductEmbeddingRepository(session).index_status(
            model_name,
            inputs.product_ids,
        )
    if status["indexed_count"] != status["expected_count"]:
        raise RuntimeError(
            "embedding index verification failed: "
            f"{status['indexed_count']}/{status['expected_count']} indexed"
        )
    if status["dimensions"] != {PGVECTOR_DIMENSION}:
        raise RuntimeError(
            f"embedding index dimensions are {sorted(status['dimensions'])}, "
            f"expected {PGVECTOR_DIMENSION}"
        )
    totals["indexed_count"] = status["indexed_count"]
    return totals


def main() -> None:
    """Validate inputs and explicitly build the pgvector index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    cache_path = args.cache or _embedding_cache_path(
        args.products,
        args.model,
        args.expected_count,
        PGVECTOR_DIMENSION,
    )
    inputs = validate_index_inputs(
        args.products,
        cache_path,
        args.model,
        args.expected_count,
    )
    logger.info(
        "Validated embedding cache: products=%d model=%s dim=%d source_hash=%s",
        len(inputs.product_ids),
        args.model,
        PGVECTOR_DIMENSION,
        inputs.metadata["data_sha256"],
    )
    if args.validate_only:
        return

    result = index_embeddings(inputs, args.model, args.batch_size)
    logger.info(
        "pgvector index complete: inserted=%d updated=%d unchanged=%d indexed=%d",
        result["inserted"],
        result["updated"],
        result["unchanged"],
        result["indexed_count"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
