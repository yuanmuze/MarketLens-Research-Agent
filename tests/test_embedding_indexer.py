"""Tests for explicit cache-to-pgvector index input validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import marketlens.retrieval.service as service_module
from marketlens.retrieval.service import (
    _cache_metadata_path,
    _compute_data_hash,
    _embedding_cache_path,
)
from scripts.index_product_embeddings import validate_index_inputs


def _write_valid_inputs(tmp_path: Path) -> tuple[Path, Path]:
    products_path = tmp_path / "products.json"
    products_path.write_text(
        json.dumps([
            {"product_id": "P1", "title": "One"},
            {"product_id": "P2", "title": "Two"},
        ]),
        encoding="utf-8",
    )
    cache_path = _embedding_cache_path(
        products_path,
        "test-model",
        2,
        384,
    )
    np.save(cache_path, np.ones((2, 384), dtype=np.float32))
    _cache_metadata_path(cache_path).write_text(
        json.dumps({
            "model_name": "test-model",
            "dim": 384,
            "count": 2,
            "dtype": "float32",
            "data_sha256": _compute_data_hash(products_path),
            "text_schema_version": "v1",
        }),
        encoding="utf-8",
    )
    return products_path, cache_path


def test_validate_index_inputs_accepts_paired_float32_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "CACHE_DIR", tmp_path / "cache")
    products_path, cache_path = _write_valid_inputs(tmp_path)

    inputs = validate_index_inputs(products_path, cache_path, "test-model", 2)

    assert inputs.product_ids == ["P1", "P2"]
    assert inputs.vectors.shape == (2, 384)
    assert inputs.vectors.dtype == np.float32


def test_validate_index_inputs_rejects_source_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "CACHE_DIR", tmp_path / "cache")
    products_path, cache_path = _write_valid_inputs(tmp_path)
    metadata_path = _cache_metadata_path(cache_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_sha256"] = "wrong"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="data_sha256"):
        validate_index_inputs(products_path, cache_path, "test-model", 2)


def test_validate_index_inputs_rejects_wrong_vector_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "CACHE_DIR", tmp_path / "cache")
    products_path, cache_path = _write_valid_inputs(tmp_path)
    np.save(cache_path, np.ones((2, 128), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        validate_index_inputs(products_path, cache_path, "test-model", 2)


def test_validate_index_inputs_rejects_duplicate_product_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "CACHE_DIR", tmp_path / "cache")
    products_path, cache_path = _write_valid_inputs(tmp_path)
    products_path.write_text(
        json.dumps([
            {"product_id": "P1", "title": "One"},
            {"product_id": "P1", "title": "Duplicate"},
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_index_inputs(products_path, cache_path, "test-model", 2)
