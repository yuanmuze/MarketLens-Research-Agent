#!/usr/bin/env python3
"""Download and verify the two frozen official Amazon Science ESCI files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

REPOSITORY = "https://github.com/amazon-science/esci-data"
COMMIT = "7916cdf6ab75a462e77f20ab40428a10923998d5"
RAW_DIRECTORY = Path("data/raw/esci")
MANIFEST_PATH = Path("data/manifests/esci_source.json")
MAX_TOTAL_BYTES = 1_200_000_000
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class EscFile:
    """Frozen Git LFS identity and minimum schema for one ESCI file."""

    name: str
    size: int
    sha256: str
    required_columns: frozenset[str]

    @property
    def url(self) -> str:
        """Return the immutable official GitHub download URL."""
        return (
            f"{REPOSITORY}/raw/{COMMIT}/shopping_queries_dataset/{self.name}"
        )


FILES = (
    EscFile(
        name="shopping_queries_dataset_examples.parquet",
        size=51_286_808,
        sha256="4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
        required_columns=frozenset({
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "large_version",
            "split",
        }),
    ),
    EscFile(
        name="shopping_queries_dataset_products.parquet",
        size=1_108_857_465,
        sha256="25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
        required_columns=frozenset({
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_locale",
        }),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, expected: EscFile) -> dict[str, object]:
    """Validate byte identity, Parquet magic, and projected schema."""
    actual_size = path.stat().st_size
    if actual_size != expected.size:
        raise ValueError(
            f"{expected.name}: size {actual_size}, expected {expected.size}"
        )
    with path.open("rb") as handle:
        if handle.read(4) != b"PAR1":
            raise ValueError(f"{expected.name}: missing leading Parquet magic")
        handle.seek(-4, os.SEEK_END)
        if handle.read(4) != b"PAR1":
            raise ValueError(f"{expected.name}: missing trailing Parquet magic")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected.sha256:
        raise ValueError(
            f"{expected.name}: SHA-256 {actual_sha256}, expected {expected.sha256}"
        )
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    missing = expected.required_columns - columns
    if missing:
        raise ValueError(f"{expected.name}: missing columns {sorted(missing)}")
    return {
        "name": expected.name,
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "row_count": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema_columns": parquet.schema_arrow.names,
        "download_url": expected.url,
    }


def download_file(destination: Path, expected: EscFile) -> dict[str, object]:
    """Resume into `.part`, validate, then atomically publish one file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logger.info("Validating existing final file: %s", destination)
        return validate_file(destination, expected)

    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected.size:
        raise ValueError(f"{part}: partial file exceeds frozen LFS size")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    logger.info("Downloading %s from byte %d", expected.name, offset)
    with requests.get(
        expected.url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()
        if offset and response.status_code != requests.codes.partial:
            raise RuntimeError(
                f"{expected.name}: server ignored Range resume; preserving {part}"
            )
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            raise ValueError(f"{expected.name}: refusing HTML response")
        mode = "ab" if offset else "wb"
        written = offset
        with part.open(mode) as handle:
            for chunk in response.iter_content(CHUNK_BYTES):
                if not chunk:
                    continue
                written += len(chunk)
                if written > expected.size:
                    raise ValueError(
                        f"{expected.name}: download exceeded frozen LFS size"
                    )
                handle.write(chunk)
                if written % (128 * 1024 * 1024) < CHUNK_BYTES:
                    logger.info(
                        "%s: %.1f%% (%d/%d bytes)",
                        expected.name,
                        written * 100 / expected.size,
                        written,
                        expected.size,
                    )
            handle.flush()
            os.fsync(handle.fileno())

    metadata = validate_file(part, expected)
    part.replace(destination)
    logger.info("Verified and published %s", destination)
    return metadata


def main() -> None:
    """Download both frozen ESCI files and write their provenance manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIRECTORY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    total = sum(item.size for item in FILES)
    if total > MAX_TOTAL_BYTES:
        raise RuntimeError(f"frozen downloads exceed safety cap: {total}")

    completed: list[dict[str, object]] = []
    for expected in FILES:
        completed.append(download_file(args.output_dir / expected.name, expected))

    manifest = {
        "schema_version": 1,
        "dataset": "Amazon Science Shopping Queries Dataset (ESCI)",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "license": "Apache-2.0",
        "license_url": f"{REPOSITORY}/blob/{COMMIT}/LICENSE",
        "download_completed_at": datetime.now(UTC).isoformat(),
        "lfs_declared_total_bytes": total,
        "raw_files_git_policy": "ignored; raw parquet files are never committed",
        "files": completed,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote provenance manifest: %s", args.manifest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
