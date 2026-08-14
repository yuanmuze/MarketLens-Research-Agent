#!/usr/bin/env python3
"""Derive a frozen English-US ESCI reduced subset without label leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

RAW_DIR = Path("data/raw/esci")
OUTPUT_DIR = Path("data/processed/esci")
SOURCE_MANIFEST = Path("data/manifests/esci_source.json")
SUBSET_MANIFEST = Path("data/manifests/esci_subset.json")
SEED = 20_260_814
TRAIN_QUERY_COUNT = 300
VALIDATION_QUERY_COUNT = 100
TEST_QUERY_COUNT = 100
VALID_LABELS = frozenset({"E", "S", "C", "I"})
logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_query(query_id: int, namespace: str) -> str:
    return hashlib.sha256(f"{SEED}:{namespace}:{query_id}".encode()).hexdigest()


def select_query_ids(
    official_train_ids: set[int],
    official_test_ids: set[int],
) -> dict[str, list[int]]:
    """Select fixed query groups while preserving official split semantics."""
    required_train = TRAIN_QUERY_COUNT + VALIDATION_QUERY_COUNT
    if len(official_train_ids) < required_train:
        raise ValueError("not enough official train query IDs")
    if len(official_test_ids) < TEST_QUERY_COUNT:
        raise ValueError("not enough official test query IDs")
    ranked_train = sorted(
        official_train_ids,
        key=lambda query_id: (_rank_query(query_id, "official-train"), query_id),
    )
    ranked_test = sorted(
        official_test_ids,
        key=lambda query_id: (_rank_query(query_id, "official-test"), query_id),
    )
    return {
        "train": ranked_train[:TRAIN_QUERY_COUNT],
        "validation": ranked_train[
            TRAIN_QUERY_COUNT : TRAIN_QUERY_COUNT + VALIDATION_QUERY_COUNT
        ],
        "test": ranked_test[:TEST_QUERY_COUNT],
    }


def _load_official_query_ids(examples_path: Path) -> tuple[set[int], set[int]]:
    dataset = ds.dataset(examples_path, format="parquet")
    scanner = dataset.scanner(
        columns=["query_id", "split"],
        filter=(ds.field("product_locale") == "us")
        & (ds.field("small_version") == 1),
        batch_size=65_536,
    )
    ids: dict[str, set[int]] = {"train": set(), "test": set()}
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            split = str(row["split"])
            if split not in ids:
                raise ValueError(f"unexpected official split: {split!r}")
            ids[split].add(int(row["query_id"]))
    return ids["train"], ids["test"]


def _load_selected_examples(
    examples_path: Path,
    splits: dict[str, list[int]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], set[str]]:
    split_by_id = {
        query_id: split_name
        for split_name, query_ids in splits.items()
        for query_id in query_ids
    }
    if len(split_by_id) != sum(map(len, splits.values())):
        raise ValueError("derived query splits overlap")
    selected = list(split_by_id)
    dataset = ds.dataset(examples_path, format="parquet")
    scanner = dataset.scanner(
        columns=[
            "query_id",
            "query",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "split",
        ],
        filter=(ds.field("product_locale") == "us")
        & (ds.field("small_version") == 1)
        & ds.field("query_id").isin(selected),
        batch_size=65_536,
    )
    query_texts: dict[int, str] = {}
    qrels: dict[str, dict[str, str]] = defaultdict(dict)
    product_ids: set[str] = set()
    seen_pairs: set[tuple[int, str]] = set()
    labels: set[str] = set()
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            product_id = str(row["product_id"])
            official_split = str(row["split"])
            derived_split = split_by_id[query_id]
            expected_official = "test" if derived_split == "test" else "train"
            if official_split != expected_official:
                raise ValueError(f"query {query_id} crossed official split boundary")
            query_text = str(row["query"] or "").strip()
            if not query_text:
                raise ValueError(f"query {query_id} has empty text")
            previous = query_texts.setdefault(query_id, query_text)
            if previous != query_text:
                raise ValueError(f"query {query_id} has inconsistent text")
            pair = (query_id, product_id)
            if pair in seen_pairs:
                raise ValueError(f"duplicate query-product pair: {pair}")
            seen_pairs.add(pair)
            label = str(row["esci_label"])
            if label not in VALID_LABELS:
                raise ValueError(f"unexpected ESCI label: {label!r}")
            labels.add(label)
            qrels[str(query_id)][product_id] = label
            product_ids.add(product_id)

    if set(query_texts) != set(selected):
        missing = set(selected) - set(query_texts)
        raise ValueError(f"selected queries missing judgments: {len(missing)}")
    queries = [
        {
            "query_id": str(query_id),
            "query": query_texts[query_id],
            "derived_split": split_name,
            "official_split": "test" if split_name == "test" else "train",
        }
        for split_name in ("train", "validation", "test")
        for query_id in splits[split_name]
    ]
    return queries, dict(qrels), product_ids


def _load_products(products_path: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
    dataset = ds.dataset(products_path, format="parquet")
    scanner = dataset.scanner(
        columns=[
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_locale",
        ],
        filter=ds.field("product_locale") == "us",
        batch_size=32_768,
    )
    products: dict[str, dict[str, Any]] = {}
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            product_id = str(row["product_id"])
            if product_id not in selected_ids:
                continue
            if product_id in products:
                raise ValueError(f"duplicate US product: {product_id}")
            title = str(row["product_title"] or "").strip()
            if not title:
                raise ValueError(f"selected product has empty title: {product_id}")
            description_parts = [
                str(row[field]).strip()
                for field in ("product_description", "product_bullet_point")
                if row[field] is not None and str(row[field]).strip()
            ]
            color = str(row["product_color"] or "").strip()
            products[product_id] = {
                "product_id": product_id,
                "title": title,
                "brand": str(row["product_brand"] or "").strip() or None,
                "category": "other",
                "price": None,
                "rating": None,
                "review_count": None,
                "attributes": {"color": color} if color else {},
                "description": " ".join(description_parts) or None,
                "images": [],
                "url": None,
            }
    missing = selected_ids - set(products)
    if missing:
        raise ValueError(f"product join missing {len(missing)} selected products")
    return [products[product_id] for product_id in sorted(products)]


def _write_json(path: Path, value: object) -> dict[str, object]:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _hash_file(path),
    }


def main() -> None:
    """Create ignored derived files and a tracked frozen subset manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=SUBSET_MANIFEST)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.output_dir}")
    if args.manifest.exists():
        parser.error(f"refusing to overwrite manifest: {args.manifest}")

    examples_path = args.raw_dir / "shopping_queries_dataset_examples.parquet"
    products_path = args.raw_dir / "shopping_queries_dataset_products.parquet"
    official_train, official_test = _load_official_query_ids(examples_path)
    splits = select_query_ids(official_train, official_test)
    queries, qrels, selected_product_ids = _load_selected_examples(
        examples_path,
        splits,
    )
    products = _load_products(products_path, selected_product_ids)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "catalog": _write_json(args.output_dir / "catalog.json", products),
        "queries": _write_json(args.output_dir / "queries.json", queries),
        "qrels": _write_json(args.output_dir / "qrels.json", qrels),
        "query_ids": _write_json(args.output_dir / "query_ids.json", splits),
    }
    query_split_sets = {name: set(ids) for name, ids in splits.items()}
    assert query_split_sets["train"].isdisjoint(query_split_sets["validation"])
    assert query_split_sets["train"].isdisjoint(query_split_sets["test"])
    assert query_split_sets["validation"].isdisjoint(query_split_sets["test"])

    per_split: dict[str, dict[str, int]] = {}
    for split_name, query_ids in splits.items():
        product_set = {
            product_id
            for query_id in query_ids
            for product_id in qrels[str(query_id)]
        }
        per_split[split_name] = {
            "queries": len(query_ids),
            "judgments": sum(len(qrels[str(query_id)]) for query_id in query_ids),
            "unique_products": len(product_set),
        }
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "dataset": "ESCI English-US reduced fixed subset",
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_commit": source["commit"],
        "source_file_sha256": {
            item["name"]: item["sha256"] for item in source["files"]
        },
        "selection": {
            "locale": "us",
            "small_version": 1,
            "seed": SEED,
            "algorithm": (
                "sort query IDs by SHA256(seed:official-split:query_id), then "
                "take 300 train + 100 validation from official train and 100 "
                "test from official test"
            ),
            "official_train_queries_available": len(official_train),
            "official_test_queries_available": len(official_test),
            "derived_counts": per_split,
            "query_ids_sha256": files["query_ids"]["sha256"],
            "query_splits_pairwise_disjoint": True,
        },
        "validation": {
            "product_join_missing": 0,
            "duplicate_query_product_pairs": 0,
            "empty_product_titles": 0,
            "labels": sorted(VALID_LABELS),
            "label_usage": "offline qrels only; never retrieval features or API input",
        },
        "catalog_unique_products": len(products),
        "files": files,
        "derived_files_git_policy": (
            "ignored detailed data; only this aggregate/hash manifest is committed"
        ),
        "benchmark_claim": (
            "fixed subset only; not the complete official ESCI benchmark"
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Frozen ESCI subset: queries=%d products=%d manifest=%s",
        len(queries),
        len(products),
        args.manifest,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
