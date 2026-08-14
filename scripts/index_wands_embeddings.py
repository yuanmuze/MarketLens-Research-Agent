#!/usr/bin/env python3
"""Import WANDS products and explicitly index its validated 384-dim cache.

This script intentionally reads only WANDS product data and embedding cache
metadata. It never opens query or label files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from marketlens.evaluation.wands import WandsProduct, load_products
from marketlens.models import Product
from marketlens.persistence.repositories import (
    ProductEmbeddingRepository,
    ProductRepository,
)
from marketlens.retrieval.pgvector_retriever import PGVECTOR_DIMENSION
from marketlens.retrieval.service import (
    _cache_metadata_path,
    _compute_data_hash,
    _embedding_cache_path,
)

logger = logging.getLogger(__name__)

DEFAULT_PRODUCTS = Path("data/external/wands/product.csv")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COUNT = 42_994


def _to_product(item: WandsProduct) -> Product:
    """Map only fields present in WANDS to the catalog model."""
    return Product(
        product_id=item.product_id,
        title=item.title,
        brand=None,
        price=None,
        rating=item.rating,
        review_count=item.review_count,
        description=item.description,
        attributes={"product_class": item.product_class},
    )


def _validated_inputs(
    products_path: Path,
    cache_path: Path,
    model_name: str,
    expected_count: int,
) -> tuple[list[Product], np.ndarray, dict[str, object]]:
    """Validate ordered IDs, source hash, paired metadata, shape and values."""
    products = [_to_product(item) for item in load_products(products_path)]
    product_ids = [product.product_id for product in products]
    if len(products) != expected_count:
        raise ValueError(f"expected {expected_count} products, found {len(products)}")
    if len(set(product_ids)) != expected_count:
        raise ValueError("WANDS product IDs are not unique")

    metadata_path = _cache_metadata_path(cache_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("paired WANDS embedding cache and metadata are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "model_name": model_name,
        "dim": PGVECTOR_DIMENSION,
        "count": expected_count,
        "dtype": "float32",
        "data_sha256": _compute_data_hash(products_path),
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
        raise ValueError(f"cache fingerprint mismatch: expected {expected_cache.name}")

    vectors = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (expected_count, PGVECTOR_DIMENSION):
        raise ValueError(f"unexpected embedding shape: {vectors.shape}")
    if vectors.dtype != np.float32:
        raise ValueError(f"unexpected embedding dtype: {vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding cache contains NaN or infinity")
    return products, vectors, metadata


def _test_database_url() -> str:
    """Return the configured PostgreSQL test URL without logging credentials."""
    raw = os.environ.get("MARKETLENS_DATABASE_URL", "")
    if not raw:
        raise RuntimeError("MARKETLENS_DATABASE_URL is required")
    url = make_url(raw)
    if not url.drivername.startswith("postgresql"):
        raise RuntimeError("WANDS indexing requires PostgreSQL")
    if not url.database or "test" not in url.database.lower():
        raise RuntimeError("refusing to index a database whose name lacks 'test'")
    return raw


def index_wands(
    products: list[Product],
    vectors: np.ndarray,
    model_name: str,
    batch_size: int,
) -> dict[str, int]:
    """Import products and embeddings in one transaction, then verify coverage."""
    engine = create_engine(_test_database_url())
    factory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    totals = {
        "products_inserted": 0,
        "products_updated": 0,
        "products_unchanged": 0,
        "embeddings_inserted": 0,
        "embeddings_updated": 0,
        "embeddings_unchanged": 0,
    }
    try:
        with factory() as session:
            product_repo = ProductRepository(session)
            embedding_repo = ProductEmbeddingRepository(session)
            for start in range(0, len(products), batch_size):
                end = min(start + batch_size, len(products))
                product_result = product_repo.upsert_many(products[start:end])
                session.flush()
                embedding_result = embedding_repo.upsert_many(
                    [product.product_id for product in products[start:end]],
                    vectors[start:end].tolist(),
                    model_name,
                    PGVECTOR_DIMENSION,
                )
                for key in ("inserted", "updated", "unchanged"):
                    totals[f"products_{key}"] += product_result[key]
                    totals[f"embeddings_{key}"] += embedding_result[key]
                logger.info("Indexed WANDS batch %d/%d", end, len(products))
            session.commit()

        product_ids = [product.product_id for product in products]
        with factory() as session:
            product_count = ProductRepository(session).count()
            status = ProductEmbeddingRepository(session).index_status(
                model_name,
                product_ids,
            )
        if product_count != len(products):
            raise RuntimeError(f"product verification failed: {product_count}")
        if status["indexed_count"] != len(products):
            raise RuntimeError(f"embedding verification failed: {status}")
        if status["dimensions"] != {PGVECTOR_DIMENSION}:
            raise RuntimeError(f"embedding dimension verification failed: {status}")
        totals["product_count"] = product_count
        totals["indexed_count"] = status["indexed_count"]
        return totals
    finally:
        engine.dispose()


def main() -> None:
    """Validate and index WANDS product embeddings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.expected_count < 1 or args.batch_size < 1:
        parser.error("counts and batch size must be positive")
    cache_path = args.cache or _embedding_cache_path(
        args.products,
        args.model,
        args.expected_count,
        PGVECTOR_DIMENSION,
    )
    products, vectors, metadata = _validated_inputs(
        args.products,
        cache_path,
        args.model,
        args.expected_count,
    )
    logger.info(
        "Validated WANDS cache: products=%d model=%s dim=%d source_hash=%s",
        len(products),
        args.model,
        PGVECTOR_DIMENSION,
        metadata["data_sha256"],
    )
    if args.validate_only:
        return
    result = index_wands(products, vectors, args.model, args.batch_size)
    logger.info("WANDS pgvector index complete: %s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
