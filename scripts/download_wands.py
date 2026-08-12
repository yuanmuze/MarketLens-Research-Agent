#!/usr/bin/env python3
"""Download WANDS dataset from the official Wayfair GitHub repository.

WANDS (Wayfair ANnotation Dataset for Search) is a public e-commerce
search benchmark with human-labeled query-product relevance judgments.
https://github.com/wayfair/WANDS

Output:
  data/external/wands/product.csv
  data/external/wands/query.csv
  data/external/wands/label.csv
  data/external/wands/source.json   (repo commit SHA + license)

Usage:
  uv run python scripts/download_wands.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/external/wands")
WANDS_REPO = "https://github.com/wayfair/WANDS"
RAW_BASE = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"

FILES: list[tuple[str, str]] = [
    ("product.csv", f"{RAW_BASE}/product.csv"),
    ("query.csv", f"{RAW_BASE}/query.csv"),
    ("label.csv", f"{RAW_BASE}/label.csv"),
]


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to dest path."""
    import urllib.request

    logger.info("Downloading %s → %s", Path(url).name, dest)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        raise


def main() -> None:
    """Download WANDS dataset products, queries, and labels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch repo commit SHA
    logger.info("Fetching WANDS repo info from %s", WANDS_REPO)
    commit_sha = "unknown"
    try:
        import re
        import urllib.request
        page = urllib.request.urlopen(WANDS_REPO).read().decode()
        m = re.search(r'commit/([a-f0-9]{40})', page)
        if m:
            commit_sha = m.group(1)
            logger.info("Detected WANDS commit: %s", commit_sha[:12])
    except Exception as e:
        logger.warning("Could not detect commit SHA: %s", e)

    # Download data files
    for filename, url in FILES:
        dest = OUTPUT_DIR / filename
        if dest.exists():
            logger.info("%s already exists, skipping", filename)
            continue
        download_file(url, dest)

    # Verify all files exist
    missing = [fn for fn, _ in FILES if not (OUTPUT_DIR / fn).exists()]
    if missing:
        logger.error("Missing files: %s", missing)
        sys.exit(1)

    # Compute SHA256
    hashes = {}
    for fn, _ in FILES:
        hashes[fn] = sha256_file(OUTPUT_DIR / fn)
        logger.info("%s SHA256: %s", fn, hashes[fn][:16])

    # Write source manifest
    manifest = {
        "source": "Wayfair WANDS (Wayfair ANnotation Dataset for Search)",
        "repo_url": WANDS_REPO,
        "repo_commit_sha": commit_sha,
        "license": "CC BY-NC 4.0",
        "license_url": "https://github.com/wayfair/WANDS/blob/main/LICENSE",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            fn: {"sha256": hashes[fn], "size_bytes": (OUTPUT_DIR / fn).stat().st_size}
            for fn, _ in FILES
        },
    }
    manifest_path = OUTPUT_DIR / "source.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Source manifest written to %s", manifest_path)
    logger.info("WANDS download complete. Run verify_wands_data.py next.")


if __name__ == "__main__":
    main()
