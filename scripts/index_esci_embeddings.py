#!/usr/bin/env python3
"""Build the frozen ESCI cache and index it into an isolated test database."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from marketlens.catalog import ProductCatalog
from marketlens.persistence.repositories import (
    ProductEmbeddingRepository,
    ProductRepository,
)
from marketlens.retrieval.embedding import SentenceTransformersBackend
from marketlens.retrieval.pgvector_retriever import PGVECTOR_DIMENSION
from marketlens.retrieval.service import RetrievalService, _embedding_cache_path

try:
    from scripts.index_product_embeddings import validate_index_inputs
except ModuleNotFoundError as exc:
    if not exc.name or not exc.name.startswith("scripts"):
        raise
    # Direct-file execution places scripts/ rather than the repository root on
    # sys.path. Keep both documented invocation styles usable.
    from index_product_embeddings import validate_index_inputs

logger = logging.getLogger(__name__)

DEFAULT_CATALOG = Path("data/processed/esci/catalog.json")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COUNT = 10_346


def _test_database_url() -> str:
    raw = os.environ.get("MARKETLENS_DATABASE_URL", "")
    if not raw:
        raise RuntimeError("MARKETLENS_DATABASE_URL is required")
    url = make_url(raw)
    if not url.drivername.startswith("postgresql"):
        raise RuntimeError("ESCI indexing requires PostgreSQL")
    if not url.database or "test" not in url.database.lower():
        raise RuntimeError("refusing to index a database whose name lacks 'test'")
    return raw


def main() -> None:
    """Generate the real cache, atomically import it, and verify coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    if args.expected_count < 1 or args.batch_size < 1:
        parser.error("counts and batch size must be positive")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    catalog = ProductCatalog.from_json(args.catalog)
    if len(catalog) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} products, found {len(catalog)}")
    backend = SentenceTransformersBackend(args.model)
    service = RetrievalService(
        catalog,
        data_path=args.catalog,
        embedding_backend=backend,
        semantic_backend="memory",
        embedding_model_name=args.model,
    ).initialize()
    status = service.status()
    if status["embedding_dim"] != PGVECTOR_DIMENSION:
        raise RuntimeError(f"unexpected embedding dimension: {status}")
    cache_path = _embedding_cache_path(
        args.catalog,
        args.model,
        args.expected_count,
        PGVECTOR_DIMENSION,
    )
    inputs = validate_index_inputs(
        args.catalog,
        cache_path,
        args.model,
        args.expected_count,
    )

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
    products = catalog.get_all_products()
    try:
        with factory() as session:
            product_repo = ProductRepository(session)
            embedding_repo = ProductEmbeddingRepository(session)
            for start in range(0, len(products), args.batch_size):
                end = min(start + args.batch_size, len(products))
                product_result = product_repo.upsert_many(products[start:end])
                if product_result["failed"]:
                    raise RuntimeError(f"product upsert failed: {product_result}")
                session.flush()
                embedding_result = embedding_repo.upsert_many(
                    inputs.product_ids[start:end],
                    inputs.vectors[start:end].tolist(),
                    args.model,
                    PGVECTOR_DIMENSION,
                )
                for key in ("inserted", "updated", "unchanged"):
                    totals[f"products_{key}"] += product_result[key]
                    totals[f"embeddings_{key}"] += embedding_result[key]
                logger.info("Indexed ESCI batch %d/%d", end, len(products))
            session.commit()

        product_ids = catalog.get_product_ids()
        with factory() as session:
            product_count = ProductRepository(session).count()
            index_status = ProductEmbeddingRepository(session).index_status(
                args.model,
                product_ids,
            )
        if product_count != args.expected_count:
            raise RuntimeError(f"product verification failed: {product_count}")
        if index_status["indexed_count"] != args.expected_count:
            raise RuntimeError(f"embedding verification failed: {index_status}")
        if index_status["dimensions"] != {PGVECTOR_DIMENSION}:
            raise RuntimeError(f"dimension verification failed: {index_status}")
        totals["product_count"] = product_count
        totals["indexed_count"] = index_status["indexed_count"]
        logger.info("ESCI pgvector index complete: %s", json.dumps(totals, sort_keys=True))
    finally:
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
