from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.common.config import get_settings
from app.release_candidate.performance import PerformanceSample, build_performance_profile, freeze_regression_envelope


def _post(base_url: str, path: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    request = Request(f"{base_url.rstrip('/')}{path}", data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit deployment URL.
            return response.status, response.read()
    except (HTTPError, URLError, TimeoutError):
        return 0, b""


def _sample(base_url: str, token: str, question: str, timeout: float) -> PerformanceSample:
    return _stream_sample(base_url, token, question, timeout)


def _stream_sample(base_url: str, token: str, question: str, timeout: float) -> PerformanceSample:
    started = perf_counter()
    request = Request(
        f"{base_url.rstrip('/')}/api/v1/chat/stream",
        data=json.dumps({"message": question}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    status = 0
    raw = b""
    ttft = 0.0
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit deployment URL.
            status = response.status
            first_byte = response.read(1)
            ttft = (perf_counter() - started) * 1000
            raw = first_byte + response.read()
    except (HTTPError, URLError, TimeoutError):
        ttft = (perf_counter() - started) * 1000
    total = (perf_counter() - started) * 1000
    outcome, diagnostics = _stream_diagnostics(raw)
    timing = diagnostics.get("timing_ms") if isinstance(diagnostics, dict) and isinstance(diagnostics.get("timing_ms"), dict) else {}
    fallback = diagnostics.get("fallback") if isinstance(diagnostics, dict) and isinstance(diagnostics.get("fallback"), dict) else {}
    if status != 200:
        error = f"HTTP_{status or 0}"
    elif outcome != "evidence_gated_answer":
        error = f"OUTCOME_{outcome.upper()}"
    elif not timing:
        error = "TIMING_MISSING"
    else:
        error = None
    return PerformanceSample(
        ttft_ms=ttft,
        total_ms=total,
        retrieval_ms=float(timing.get("retrieval_ms") or 0),
        generation_provider_ms=float(timing.get("generation_provider_ms") or 0),
        embedding_provider_ms=float(timing.get("embedding_provider_ms") or 0),
        persistence_ms=float(timing.get("persistence_ms") or 0),
        outcome=outcome,
        error_code=error,
        fallback_hops=int(fallback.get("hops") or 0),
    )


def _stream_diagnostics(raw: bytes) -> tuple[str, dict[str, Any]]:
    outcome = "transport_error"
    diagnostics: dict[str, Any] = {}
    try:
        stream = raw.decode()
    except UnicodeDecodeError:
        return outcome, diagnostics
    for event in stream.split("\n\n"):
        lines = event.splitlines()
        if not lines or not lines[0].startswith("event: "):
            continue
        data_line = next((line for line in lines[1:] if line.startswith("data: ")), "")
        try:
            payload = json.loads(data_line.removeprefix("data: "))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if lines[0] == "event: outcome" and isinstance(payload.get("outcome"), str):
            outcome = payload["outcome"]
        elif lines[0] == "event: retrieval_diagnostics" and isinstance(payload.get("retrieval_diagnostics"), dict):
            diagnostics = payload["retrieval_diagnostics"]
    return outcome, diagnostics


async def _collect(base_url: str, token: str, question: str, concurrency: int, requests: int, timeout: float) -> tuple[PerformanceSample, ...]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one() -> PerformanceSample:
        async with semaphore:
            return await asyncio.to_thread(_sample, base_url, token, question, timeout)

    return tuple(await asyncio.gather(*(one() for _ in range(requests))))


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    _, raw = await asyncio.to_thread(
        _post,
        args.base_url,
        "/api/v1/auth/login",
        {"username": settings.bootstrap_admin_username, "password": settings.bootstrap_admin_password},
        {"Content-Type": "application/json"},
        args.timeout_seconds,
    )
    try:
        login = json.loads(raw.decode())
        token = login.get("data", login).get("access_token")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        token = None
    if not isinstance(token, str) or not token:
        raise RuntimeError("administrator authentication failed")
    profiles = []
    for concurrency in (1, 5):
        samples = await _collect(args.base_url, token, args.question, concurrency, args.requests, args.timeout_seconds)
        profiles.append(build_performance_profile(run_id=f"{args.run_id}-c{concurrency}", concurrency=concurrency, samples=samples))
    envelope = freeze_regression_envelope(tuple(profiles))
    return {
        "kind": "performance-run",
        "source_revision": args.source_revision,
        "generated_at": datetime.now(UTC).isoformat(),
        "conditions": {"authenticated_role": "admin", "request_path": "/api/v1/chat/stream", "ttft_definition": "first body byte received from the authenticated SSE response", "timeout_seconds": args.timeout_seconds},
        "profiles": [
            {"section": "performance", "schema_version": "1.0.0", "run_ids": [profile.run_id], "load": profile.load,
             "metrics": {"ttft_ms": profile.metrics["ttft_ms"], "total_ms": profile.metrics["total_ms"], "error_rate": profile.metrics["error_rate"], "retrieval_ms": profile.metrics["retrieval_ms"], "provider_ms": profile.metrics["generation_provider_ms"], "embedding_provider_ms": profile.metrics["embedding_provider_ms"], "persistence_ms": profile.metrics["persistence_ms"], "application_controlled_ms": profile.metrics["application_controlled_ms"]},
             "target": {"p95_seconds_target": 12, "met": profile.metrics["total_ms"]["p95"] <= 12000, "note": "Measured end-to-end P95; the target does not determine tolerance."},
             "error_counts": profile.error_counts, "regression_envelope": envelope}
            for profile in profiles
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    manifest = asyncio.run(run(args))
    target = args.output_dir / args.run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
