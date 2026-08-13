#!/usr/bin/env python3
"""Reproducible local load test against the running Docker API.

Hits the real HTTP API (not TestClient). No real LLM calls — the agent
path uses the offline NoOp/degraded fallback when no API key is set.

Usage:
  uv run python scripts/load_test.py --base-url http://127.0.0.1:8000
  uv run python scripts/load_test.py --base-url http://127.0.0.1:8000 --requests 200 --concurrency 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx


def get_git_sha() -> str:
    """Return current git SHA (or 'unknown')."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def one_request(base_url: str, path: str, method: str = "GET") -> tuple[int, float]:
    """Perform one request; return (status_code, latency_ms)."""
    t0 = time.monotonic()
    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            if method == "GET":
                resp = client.get(path)
            else:
                resp = client.post(path, json={"message": "best wireless headphones"})
            return resp.status_code, (time.monotonic() - t0) * 1000
    except Exception:
        return 0, (time.monotonic() - t0) * 1000


def run_load(base_url: str, requests: int, concurrency: int, path: str, method: str = "GET") -> dict[str, Any]:
    """Run a load batch and return metrics."""
    # Warm-up
    for _ in range(min(5, requests)):
        one_request(base_url, path, method)

    latencies: list[float] = []
    status_counts: dict[int, int] = {}
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, base_url, path, method) for _ in range(requests)]
        for f in futures:
            status, lat = f.result()
            status_counts[status] = status_counts.get(status, 0) + 1
            if status >= 200 and status < 600:
                latencies.append(lat)

    wall_s = time.monotonic() - t0
    lat_sorted = sorted(latencies) if latencies else [0.0]

    def percentile(q: float) -> float:
        """Return the q-th percentile of latencies."""
        idx = min(len(lat_sorted) - 1, int(len(lat_sorted) * q))
        return lat_sorted[idx]

    return {
        "path": path,
        "requests": requests,
        "concurrency": concurrency,
        "success": status_counts.get(200, 0),
        "client_4xx": sum(v for k, v in status_counts.items() if 400 <= k < 500),
        "server_5xx": sum(v for k, v in status_counts.items() if 500 <= k < 600),
        "errors": status_counts.get(0, 0),
        "throughput_rps": round(requests / wall_s, 2) if wall_s > 0 else 0,
        "p50_ms": round(percentile(0.50), 2),
        "p95_ms": round(percentile(0.95), 2),
        "p99_ms": round(percentile(0.99), 2),
    }


def main() -> None:
    """Run load tests against the running API."""
    parser = argparse.ArgumentParser(description="MarketLens local load test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--out", default="reports/load_test_results.json")
    args = parser.parse_args()

    scenarios = [
        ("retrieval", "/search?q=headphones&top_k=5", "GET"),
        ("agent", "/agent/recommend", "POST"),
    ]

    results = {
        "git_sha": get_git_sha(),
        "base_url": args.base_url,
        "requests_per_scenario": args.requests,
        "scenarios": {},
    }

    for name, path, method in scenarios:
        print(f"Running {name} (concurrency={args.concurrency}, requests={args.requests})...")
        r = run_load(args.base_url, args.requests, args.concurrency, path, method)
        results["scenarios"][name] = r
        print(json.dumps(r, indent=2))

    # Also run a concurrency=1 baseline for retrieval
    print("Running retrieval baseline (concurrency=1)...")
    r = run_load(args.base_url, args.requests, 1, "/search?q=headphones&top_k=5", "GET")
    results["scenarios"]["retrieval_concurrency_1"] = r
    print(json.dumps(r, indent=2))

    import os
    os.makedirs("reports", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
