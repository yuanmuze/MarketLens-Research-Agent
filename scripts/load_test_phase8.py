#!/usr/bin/env python3
"""Run and preserve the Phase 8 local Docker HTTP load matrix."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

QUERIES = (
    "wireless headphones",
    "noise cancelling earbuds",
    "portable bluetooth speaker",
    "mechanical gaming keyboard",
    "usb c charging cable",
)


def _git_sha() -> str:
    """Return the source revision without mutating the repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _percentile(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 2)


def _request(
    client: httpx.Client,
    scenario: str,
    request_number: int,
) -> tuple[int, float, str | None]:
    """Issue and validate one real HTTP request."""
    query = QUERIES[request_number % len(QUERIES)]
    request_id = f"phase8-load-{scenario}-{uuid.uuid4().hex}"
    started = time.perf_counter()
    try:
        if scenario in {"hybrid", "quality"}:
            strategy = "hybrid" if scenario == "hybrid" else "rerank"
            response = client.get(
                "/search",
                params={
                    "q": query,
                    "strategy": strategy,
                    "top_k": 5,
                    "candidate_k": 50,
                },
                headers={"X-Request-ID": request_id},
            )
        else:
            strategy = "hybrid"
            response = client.post(
                "/agent/recommend",
                json={
                    "message": query,
                    "mode": "balanced",
                    "max_results": 5,
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
        latency_ms = (time.perf_counter() - started) * 1000
    except httpx.HTTPError as exc:
        return 0, (time.perf_counter() - started) * 1000, type(exc).__name__

    if response.status_code != 200:
        return response.status_code, latency_ms, None
    try:
        body = response.json()
        if scenario in {"hybrid", "quality"}:
            results = body["results"]
            if not results:
                return 200, latency_ms, "empty_results"
            if any(item.get("source") != strategy for item in results):
                return 200, latency_ms, "unexpected_strategy"
        else:
            if body.get("status") != "completed":
                return 200, latency_ms, "agent_not_completed"
            if body.get("degraded") is not False:
                return 200, latency_ms, "agent_degraded"
            if body.get("mode_used") != "hybrid":
                return 200, latency_ms, "unexpected_agent_mode"
            if not body.get("recommendations"):
                return 200, latency_ms, "empty_recommendations"
            if int(body.get("tool_calls", 0)) < 1:
                return 200, latency_ms, "missing_tool_call"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 200, latency_ms, "invalid_json_payload"
    return 200, latency_ms, None


def _worker(
    base_url: str,
    scenario: str,
    request_numbers: list[int],
) -> list[tuple[int, float, str | None]]:
    """Run one worker lane with a persistent HTTP connection pool."""
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        return [_request(client, scenario, number) for number in request_numbers]


def _run_batch(
    base_url: str,
    scenario: str,
    request_count: int,
    concurrency: int,
    warmup_count: int,
) -> dict[str, Any]:
    """Run an excluded warmup followed by one measured scenario."""
    _worker(base_url, scenario, list(range(-warmup_count, 0)))
    lanes = [list(range(lane, request_count, concurrency)) for lane in range(concurrency)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        nested = list(pool.map(lambda lane: _worker(base_url, scenario, lane), lanes))
    wall_seconds = time.perf_counter() - started
    samples = [sample for lane in nested for sample in lane]
    statuses = Counter(status for status, _latency, _invalid in samples)
    invalid = Counter(reason for _status, _latency, reason in samples if reason)
    latencies = [latency for status, latency, _invalid in samples if status == 200]
    successes = sum(1 for status, _latency, reason in samples if status == 200 and not reason)
    return {
        "scenario": scenario,
        "requests": request_count,
        "concurrency": concurrency,
        "warmup_requests_excluded": warmup_count,
        "success": successes,
        "status_counts": {str(key): value for key, value in sorted(statuses.items())},
        "client_4xx": sum(value for key, value in statuses.items() if 400 <= key < 500),
        "server_5xx": sum(value for key, value in statuses.items() if 500 <= key < 600),
        "transport_errors": statuses[0],
        "invalid_payloads": dict(sorted(invalid.items())),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(request_count / wall_seconds, 2),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "p99_ms": _percentile(latencies, 0.99),
    }


def _readiness(base_url: str, expected_backend: str) -> dict[str, Any]:
    """Require a real, populated semantic backend before load begins."""
    response = httpx.get(f"{base_url}/health/ready", timeout=120.0)
    response.raise_for_status()
    body = response.json()
    if body.get("semantic_backend") != expected_backend:
        raise RuntimeError(
            f"expected semantic backend {expected_backend!r}, got "
            f"{body.get('semantic_backend')!r}"
        )
    if not body.get("semantic_index_ready") or int(
        body.get("semantic_indexed_count", 0)
    ) <= 0:
        raise RuntimeError("semantic index is not populated")
    if "fake" in str(body.get("embedding_backend", "")).lower():
        raise RuntimeError("fake embedding backend is forbidden in Phase 8 load tests")
    return body


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    """Write once so an initial formal result cannot be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run_profile(args: argparse.Namespace) -> None:
    """Run either the memory or pgvector portion of the matrix."""
    if args.requests < 100:
        raise ValueError("formal Phase 8 scenarios require at least 100 requests")
    expected_backend = args.profile
    ready = _readiness(args.base_url, expected_backend)
    scenario_names = ["hybrid"] if args.profile == "memory" else [
        "hybrid",
        "quality",
        "agent",
    ]
    results = []
    for scenario in scenario_names:
        for concurrency in (1, 10):
            print(
                f"Running {args.profile}/{scenario} concurrency={concurrency} "
                f"requests={args.requests}"
            )
            result = _run_batch(
                args.base_url,
                scenario,
                args.requests,
                concurrency,
                args.warmup,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            results.append(result)
    payload = {
        "schema_version": 1,
        "scope": "local Docker development benchmark; not a production claim",
        "profile": args.profile,
        "git_sha": _git_sha(),
        "image_id": args.image_id,
        "started_from_readiness": ready,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": results,
    }
    _write_new(Path(args.output), payload)


def _merge(args: argparse.Namespace) -> None:
    """Merge the immutable profile outputs into the final report."""
    inputs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    profiles = {item.get("profile") for item in inputs}
    if profiles != {"memory", "pgvector"}:
        raise ValueError("merge requires exactly memory and pgvector profile results")
    scenarios = [scenario for item in inputs for scenario in item["scenarios"]]
    if len(scenarios) != 8 or any(item["requests"] < 100 for item in scenarios):
        raise ValueError("load matrix is incomplete")
    if any(item["success"] != item["requests"] for item in scenarios):
        raise ValueError("load matrix contains failed or invalid responses")
    payload = {
        "schema_version": 1,
        "scope": "local Docker development benchmark; not a production claim",
        "git_sha": _git_sha(),
        "profiles": sorted(inputs, key=lambda item: item["profile"]),
        "scenario_count": len(scenarios),
        "total_measured_requests": sum(item["requests"] for item in scenarios),
        "acceptance": {
            "all_http_200": True,
            "all_payloads_valid": True,
            "no_memory_fallback": True,
            "real_embeddings_only": True,
            "external_llm_calls": 0,
        },
    }
    _write_new(Path(args.output), payload)


def main() -> None:
    """Parse CLI arguments and run or merge a formal profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--profile", choices=("memory", "pgvector"), required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    run_parser.add_argument("--requests", type=int, default=100)
    run_parser.add_argument("--warmup", type=int, default=5)
    run_parser.add_argument("--image-id", required=True)
    run_parser.add_argument("--output", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", nargs=2, required=True)
    merge_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        _run_profile(args)
    else:
        _merge(args)


if __name__ == "__main__":
    main()
