"""Focused tests for frozen ESCI download and query-group derivation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.benchmark_esci_backends import _backend_parity, _evaluate
from scripts.download_esci import EscFile, validate_file
from scripts.prepare_esci_subset import select_query_ids


def test_esci_query_selection_is_deterministic_grouped_and_disjoint() -> None:
    official_train = set(range(1_000))
    official_test = set(range(10_000, 10_300))

    first = select_query_ids(official_train, official_test)
    second = select_query_ids(set(reversed(sorted(official_train))), official_test)

    assert first == second
    assert {name: len(ids) for name, ids in first.items()} == {
        "train": 300,
        "validation": 100,
        "test": 100,
    }
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set(first["train"] + first["validation"]) <= official_train
    assert set(first["test"]) <= official_test


def test_validate_esci_file_checks_lfs_identity_magic_and_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny.parquet"
    pq.write_table(pa.table({"query_id": [1], "query": ["phone"]}), path)
    payload = path.read_bytes()
    expected = EscFile(
        name=path.name,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        required_columns=frozenset({"query_id", "query"}),
    )

    metadata = validate_file(path, expected)

    assert metadata["size_bytes"] == len(payload)
    assert metadata["row_count"] == 1


def test_validate_esci_file_rejects_missing_projected_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny.parquet"
    pq.write_table(pa.table({"query_id": [1]}), path)
    payload = path.read_bytes()
    expected = EscFile(
        name=path.name,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        required_columns=frozenset({"query_id", "query"}),
    )

    with pytest.raises(ValueError, match="missing columns"):
        validate_file(path, expected)


def test_esci_metrics_keep_exact_and_es_relevance_distinct() -> None:
    queries = [{"query_id": "q1", "query": "phone"}]
    qrels = {"q1": {"exact": 3, "substitute": 2, "complement": 1}}

    metrics, runs, per_query = _evaluate(
        "semantic_memory",
        queries,
        qrels,
        lambda _query: ["substitute", "exact", "unjudged"],
    )

    assert metrics["recall_at_10_es"] == 1.0
    assert metrics["mrr_at_10_relevant_es"] == 1.0
    assert metrics["exact_mrr_at_10"] == 0.5
    assert metrics["ndcg_at_10"] > 0
    assert metrics["failures"] == {}
    assert runs[0]["failure"] is None
    assert per_query[0]["relevant_count"] == 2


def test_backend_parity_reports_exact_order_and_overlap() -> None:
    runs = [
        {"strategy": "memory", "query_id": "q1", "product_ids": ["a", "b"]},
        {"strategy": "pg", "query_id": "q1", "product_ids": ["a", "b"]},
        {"strategy": "memory", "query_id": "q2", "product_ids": ["a", "b"]},
        {"strategy": "pg", "query_id": "q2", "product_ids": ["b", "a"]},
    ]

    parity = _backend_parity(runs, "memory", "pg")

    assert parity["query_count"] == 2
    assert parity["exact_ranking_matches"] == 1
    assert parity["exact_ranking_match_rate"] == 0.5
    assert parity["mean_top10_overlap"] == pytest.approx(0.2)
