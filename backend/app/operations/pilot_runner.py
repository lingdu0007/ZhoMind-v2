"""Explicitly commissioned authenticated Pilot workload, using existing APIs."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx

from app.common.canonical_json import canonical_json_sha256
from app.operations.pilot_assessment import assess_reports
from app.operations.pilot_availability import ProbeMeasurement
from app.operations.pilot_collector import measure_stream
from app.operations.pilot_measurement import AdmissionPeak, MeasurementBinding, RequestMeasurement, WorkloadProfile, summarize_requests
from app.operations.pilot_triggers import TriggerObservation, evaluate_triggers


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _data(response: Any) -> dict:
    if response.status_code != 200:
        raise ValueError("authenticated measurement operation failed")
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise ValueError("invalid measurement response")
    return payload["data"]


def _assert_snapshot(client: Any, token: str, binding: MeasurementBinding) -> None:
    snapshot = _data(client.get("/api/v1/operations/measurement-snapshot", headers=_headers(token)))
    expected = binding.model_dump()
    if type(snapshot.get("background_workers")) is not int or snapshot["background_workers"] != 1:
        raise ValueError("measurement worker configuration mismatch")
    required = MeasurementBinding.model_fields.keys() - {"window_start", "window_end", "concurrency", "mode"}
    if not required <= snapshot.keys():
        raise ValueError("incomplete measurement snapshot")
    observed = MeasurementBinding.model_validate({**expected, **{key: snapshot[key] for key in required}})
    if observed != binding:
        raise ValueError("measurement snapshot drift")


def _frames(response: Any, started: float) -> list[tuple[str, str, float]]:
    result = []
    event, data = "message", []
    size = 0
    for line in response.iter_lines():
        size += len(line.encode("utf-8"))
        if size > 2 * 1024 * 1024:
            raise ValueError("measurement stream exceeded bounded response")
        if line == "":
            if data:
                result.append((event, "\n".join(data), (time.perf_counter() - started) * 1000))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].removeprefix(" ")
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    # An unframed final event is intentionally discarded, not accepted as done.
    return result


def collect_request(
    client: Any, *, binding: MeasurementBinding, member_token: str, administrator_token: str,
    question: str, burst_round: int | None = None,
) -> RequestMeasurement:
    started, observed_at = time.perf_counter(), datetime.now(UTC)
    request_id = uuid4()
    session_id = str(uuid4())
    frames = []
    status = 0
    try:
        with client.stream("POST", "/api/v1/chat/stream", headers=_headers(member_token),
                           json={"message": question, "session_id": session_id}) as response:
            status = response.status_code
            request_id = UUID(response.headers["x-request-id"])
            if status == 200:
                frames = _frames(response, started)
        terminal_ms = (time.perf_counter() - started) * 1000
    except (httpx.HTTPError, OSError, ValueError, KeyError):
        terminal_ms = (time.perf_counter() - started) * 1000
    if status != 200:
        population = {400: "invalid", 401: "unauthorized", 403: "unauthorized", 422: "invalid", 429: "throttled"}.get(status, "rejected")
        return RequestMeasurement.model_validate({
            "request_id": request_id, "binding_identity": binding.identity, "observed_at": observed_at,
            "population": population, "state": "rejected", "terminal_ms": terminal_ms,
            "normalized_failure": "authorization" if status in (401, 403) else "application",
            "burst_round": burst_round,
        })
    operation = None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/operations/requests/{request_id}", headers=_headers(administrator_token))
        if response.status_code == 200:
            operation = _data(response)
            break
        if response.status_code != 404:
            raise ValueError("measurement authority unavailable")
        time.sleep(.05)
    verified_message = None
    history = client.get(f"/api/v1/sessions/{session_id}", headers=_headers(member_token))
    if history.status_code == 200:
        # The owner-only reload performs canonical identity, evidence and QCS validation.
        payload = history.json()
        messages = payload.get("messages", payload.get("data", {}).get("messages", []))
        completed = [message for message in messages if isinstance(message, dict)
                     and isinstance(message.get("answer_execution"), dict)
                     and message["answer_execution"].get("assistant_message_id") == message.get("id")]
        if len(completed) == 1:
            verified_message = completed[0]
    return measure_stream(binding, frames, request_id=request_id, observed_at=observed_at,
                          terminal_ms=terminal_ms, operation=operation, burst_round=burst_round, verified_message=verified_message)


def run_entry_requests(
    client: Any, *, binding: MeasurementBinding, member_tokens: tuple[str, ...],
    administrator_token: str, question: str,
) -> dict:
    if len(member_tokens) != 8 or len(set(member_tokens)) != 8:
        raise ValueError("workload requires eight distinct admitted members")
    members = []
    for token in member_tokens:
        member = _data(client.get("/api/v1/auth/me", headers=_headers(token)))
        if member.get("role") != "user" or not isinstance(member.get("username"), str):
            raise ValueError("workload requires Knowledge User authority")
        members.append(member["username"])
    if len(set(members)) != 8:
        raise ValueError("workload requires eight distinct admitted members")
    _assert_snapshot(client, administrator_token, binding)
    administrator = _data(client.get("/api/v1/auth/me", headers=_headers(administrator_token)))
    owner = canonical_json_sha256({"member": administrator["username"]})
    samples: list[RequestMeasurement] = []
    reports = []
    for concurrency in (1, 2, 4):
        current = MeasurementBinding.model_validate({**binding.model_dump(), "concurrency": concurrency})
        group: list[RequestMeasurement] = []
        peaks = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for offset in range(0, 40, concurrency):
                futures = [
                    pool.submit(
                        collect_request, client, binding=current,
                        member_token=member_tokens[(offset + index) % 8], administrator_token=administrator_token,
                        question=question, burst_round=offset // 4 + 1 if concurrency == 4 else None,
                    )
                    for index in range(concurrency)
                ]
                if concurrency == 4:
                    peak_observed = False
                    while not all(future.done() for future in futures):
                        if not peak_observed:
                            admission = _data(client.get("/api/v1/operations", headers=_headers(administrator_token)))["admission"]
                            if admission["executing"] == 2 and admission["queued"] == 2:
                                peaks.append(AdmissionPeak(
                                    binding_identity=current.identity, burst_round=offset // 4 + 1,
                                    observed_at=datetime.now(UTC), executing=2, queued=2,
                                ))
                                peak_observed = True
                        time.sleep(.02)
                group.extend(future.result() for future in futures)
        _assert_snapshot(client, administrator_token, current)
        reports.append(summarize_requests(current, group, admission_peaks=peaks))
        samples.extend(group)
    profile_ok = binding.active_entries == 50 and 0 < binding.eligible_chunks <= 10000
    follow_ups = []
    objectives = [
        (report["binding_identity"], f"c{report['binding']['concurrency']}_{name}", metric["status"])
        for report in reports for name, metric in {
            **report["metrics"], "route": report["route"], "burst": report["burst"],
        }.items() if metric["status"] in {"at_risk", "unavailable", "observed_with_insufficient_sample"}
    ]
    if not profile_ok:
        objectives.append((binding.identity, "workload_profile", "unavailable"))
    for identity, objective, status in objectives:
        follow_ups.append({
            "binding_identity": identity, "objective": objective, "evidence_status": status,
            "required_action": "remediation_and_re_acceptance" if status == "at_risk" else "complete_measurement",
            "owner_sha256": owner, "content_action": "none",
            "review_due_at": (datetime.now(UTC) + timedelta(days=7)).isoformat(),
        })
    report = {
        "schema": "pilot_entry_requests/v1", "profile": WorkloadProfile().model_dump(),
        "profile_status": "observed" if profile_ok else "unavailable",
        "observed_members": len(members), "request_reports": reports,
        "samples": [sample.model_dump(mode="json") for sample in samples],
        "follow_up_decisions": follow_ups,
        "live_acceptance": False,
    }
    report["tightening_decisions"] = assess_reports([report], owner_sha256=owner, now=datetime.now(UTC))
    return report


def collect_probe(
    client: Any, *, binding: MeasurementBinding, member_token: str, administrator_token: str,
    question: str, product_check: bool,
) -> ProbeMeasurement:
    at = datetime.now(UTC)
    healthy = False
    outcome = None
    try:
        healthy = _data(client.get("/api/v1/health"))["status"] == "up"
        if product_check:
            _assert_snapshot(client, administrator_token, binding)
            sample = collect_request(client, binding=binding, member_token=member_token,
                                     administrator_token=administrator_token, question=question)
            _assert_snapshot(client, administrator_token, binding)
            if sample.state == "completed" and sample.outcome in {"evidence_gated_answer", "generation_unavailable"}:
                outcome = sample.outcome
    except (ValueError, OSError, KeyError, httpx.HTTPError):
        pass
    return ProbeMeasurement(binding_identity=binding.identity, at=at, healthy=healthy,
                            product_checked=product_check, product_outcome=outcome)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an explicitly commissioned Pilot request workload")
    parser.add_argument("--task", choices=("requests", "ten_item_bundle", "fifty_entry_rebuild", "probe"), default="requests")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--question-file", type=Path)
    parser.add_argument("--bundle-file", type=Path)
    parser.add_argument("--history-report", action="append", type=Path, default=[])
    parser.add_argument("--trigger-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commissioned", action="store_true")
    args = parser.parse_args()
    try:
        binding = MeasurementBinding.model_validate_json(args.binding.read_text())
        target = urlsplit(args.base_url)
        if target.username or target.password or target.query or target.fragment:
            raise ValueError("invalid target")
        if target.scheme != "https" and not (binding.mode == "local_deterministic" and target.hostname in {"localhost", "127.0.0.1"}):
            raise ValueError("target must use HTTPS")
        if not args.commissioned:
            raise ValueError("explicit workload commission required")
        admin = os.environ["PILOT_ADMIN_TOKEN"]
        with httpx.Client(base_url=args.base_url, timeout=65, follow_redirects=False, trust_env=False) as client:
            if args.task in {"requests", "probe"}:
                if args.question_file is None:
                    raise ValueError("a declared private question file is required")
                question = args.question_file.read_text()
                if args.task == "requests":
                    tokens = tuple(os.environ[f"PILOT_MEMBER_{index}_TOKEN"] for index in range(1, 9))
                    report = run_entry_requests(client, binding=binding, member_tokens=tokens,
                                                administrator_token=admin, question=question)
                else:
                    report = collect_probe(
                        client, binding=binding, member_token=os.environ["PILOT_MEMBER_1_TOKEN"],
                        administrator_token=admin, question=question, product_check=True,
                    ).model_dump(mode="json")
            else:
                from app.operations.pilot_build_runner import measure_bundle

                if args.bundle_file is None:
                    raise ValueError("an explicitly approved bundle file is required")
                _assert_snapshot(client, admin, binding)
                report = measure_bundle(client, binding=binding, administrator_token=admin,
                                        manifest=json.loads(args.bundle_file.read_text()), kind=args.task)
                _assert_snapshot(client, admin, binding)
            actor = _data(client.get("/api/v1/auth/me", headers=_headers(admin)))
            owner = canonical_json_sha256({"member": actor["username"]})
            if args.history_report:
                if args.task == "probe":
                    raise ValueError("individual probe is not an aggregate report")
                report["tightening_decisions"] = assess_reports(
                    [json.loads(path.read_text()) for path in args.history_report] + [report],
                    owner_sha256=owner, now=datetime.now(UTC),
                )
            if args.trigger_file:
                raw = json.loads(args.trigger_file.read_text())
                observations = [TriggerObservation.model_validate_json(json.dumps(item)) for item in raw]
                report["declared_trigger_decisions"] = evaluate_triggers(
                    binding, observations, owner_sha256=owner, now=datetime.now(UTC),
                )
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump(report, output, ensure_ascii=True, indent=2)
        print("Pilot request report written; deployment acceptance remains separate.")
        return 0
    except (ValueError, OSError, KeyError, httpx.HTTPError):
        print("Pilot measurement unavailable; verify the declared binding, authorization and target.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
