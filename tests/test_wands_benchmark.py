"""Focused tests for the frozen WANDS workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from marketlens.retrieval.service import (
    _cache_metadata_path,
    _compute_data_hash,
    _embedding_cache_path,
)
from scripts.benchmark_wands_backends import split_query_ids
from scripts.index_wands_embeddings import _validated_inputs


def _write_wands_products(path: Path) -> None:
    path.write_text(
        "product_id\tproduct_name\tproduct_class\tproduct_description\t"
        "product_features\tcategory_hierarchy\taverage_rating\treview_count\n"
        "P1\tOne\tClass A\tFirst\tFeature 1\tRoot A\t4.5\t10\n"
        "P2\tTwo\tClass B\tSecond\tFeature 2\tRoot B\t4.0\t5\n",
        encoding="utf-8",
    )


def test_frozen_wands_split_is_deterministic_disjoint_and_60_20_20() -> None:
    query_ids = [str(index) for index in range(480)]

    first = split_query_ids(query_ids)
    second = split_query_ids(list(reversed(query_ids)))

    assert first == second
    assert {name: len(ids) for name, ids in first.items()} == {
        "train": 288,
        "validation": 96,
        "test": 96,
    }
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(query_ids)


def test_wands_index_inputs_validate_ordered_float32_cache(tmp_path: Path) -> None:
    products_path = tmp_path / "product.csv"
    _write_wands_products(products_path)
    cache_path = _embedding_cache_path(products_path, "test-model", 2, 384)
    cache_path = tmp_path / cache_path.name
    np.save(cache_path, np.ones((2, 384), dtype=np.float32))
    metadata = {
        "model_name": "test-model",
        "dim": 384,
        "count": 2,
        "dtype": "float32",
        "data_sha256": _compute_data_hash(products_path),
    }
    _cache_metadata_path(cache_path).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    products, vectors, loaded_metadata = _validated_inputs(
        products_path,
        cache_path,
        "test-model",
        2,
    )

    assert [product.product_id for product in products] == ["P1", "P2"]
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert loaded_metadata == metadata
