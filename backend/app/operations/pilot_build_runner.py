"""Measure explicitly imported Candidate work without publishing it."""

import time
from datetime import UTC, datetime
from typing import Any, Literal

from app.common.canonical_json import canonical_json_sha256
from app.operations.pilot_assessment import assess_reports
from app.operations.pilot_measurement import BuildItemMeasurement, BuildMeasurement, MeasurementBinding, summarize_build
from app.operations.pilot_runner import _data, _headers


def measure_bundle(
    client: Any, *, binding: MeasurementBinding, administrator_token: str,
    manifest: dict, kind: Literal["ten_item_bundle", "fifty_entry_rebuild"],
) -> dict:
    expected = 10 if kind == "ten_item_bundle" else 50
    items = manifest.get("items")
    if (not isinstance(items, list) or len(items) != expected
            or any(not isinstance(item, dict) or item.get("operation") not in {"create", "replace"} for item in items)):
        raise ValueError("measurement requires a fresh build for every item")
    observed_start = datetime.now(UTC)
    if not binding.window_start <= observed_start < binding.window_end:
        raise ValueError("build outside observation window")
    if kind == "fifty_entry_rebuild":
        revisions = []
        for item in items:
            artifact = item.get("artifact")
            if item["operation"] != "replace" or not isinstance(artifact, dict):
                raise ValueError("rebuild must use the bound corpus")
            entry, revision = artifact.get("entry_identity"), artifact.get("editorial_revision_identity")
            if not isinstance(entry, str) or not isinstance(revision, str):
                raise ValueError("rebuild must use the bound corpus")
            revisions.append((entry, revision))
        if (binding.active_entries != 50 or len(set(revisions)) != 50
                or canonical_json_sha256(sorted(revisions)) != binding.editorial_inputs_sha256):
            raise ValueError("rebuild must use the exact bound corpus")
    headers = _headers(administrator_token)
    root = "/api/v1/reviewed-release-bundles"
    started = time.perf_counter()
    imported = _data(client.post(root + "/import", headers=headers, json=manifest))
    admitted = imported.get("items", [])
    if len(admitted) != expected or any(not item.get("job_id") for item in admitted):
        raise ValueError("not every measurement item admitted a fresh build")
    if imported.get("bundle_sha256") != manifest.get("bundle_sha256"):
        raise ValueError("bundle measurement identity mismatch")
    initial = {}
    for item in admitted:
        job_id = item["job_id"]
        job = _data(client.get(f"{root}/jobs/{job_id}", headers=headers))
        if job["status"] != "queued" or job["attempt"] != 1 or job.get("dispatched_at") is not None:
            raise ValueError("measurement requires a fresh build, not historical completion")
        if job_id in initial:
            raise ValueError("duplicate build identity")
        initial[job_id] = job
    for job_id in initial:
        _data(client.post(f"{root}/jobs/{job_id}/dispatch", headers=headers))
    results = {}
    deadline = time.monotonic() + 7200
    while len(results) != expected and time.monotonic() < deadline:
        for item in admitted:
            job_id = item["job_id"]
            if job_id in results:
                continue
            job = _data(client.get(f"{root}/jobs/{job_id}", headers=headers))
            if (job.get("frozen_input_sha256") != initial[job_id].get("frozen_input_sha256")
                    or job.get("embedding_configuration") != initial[job_id].get("embedding_configuration")):
                raise ValueError("build input or configuration drift")
            if job["status"] not in {"queued", "running"}:
                results[job_id] = job
        if len(results) != expected:
            time.sleep(.25)
    completed_at = time.perf_counter()
    observed_end = datetime.now(UTC)
    observations = []
    for item in admitted:
        job_id = item["job_id"]
        job = results.get(job_id) or _data(client.get(f"{root}/jobs/{job_id}", headers=headers))
        chunks = 0
        if job["status"] == "candidate_ready":
            candidate = _data(client.get(f"{root}/candidates/{job['candidate_id']}/inspection", headers=headers))["candidate"]
            chunks = len(candidate["chunks"])
        execution_ms = None
        if job.get("started_at") and job.get("completed_at"):
            job_start = datetime.fromisoformat(job["started_at"])
            job_end = datetime.fromisoformat(job["completed_at"])
            # Persisted product timestamps are UTC; SQLite returns them without an offset.
            job_start = job_start.replace(tzinfo=UTC) if job_start.tzinfo is None else job_start
            job_end = job_end.replace(tzinfo=UTC) if job_end.tzinfo is None else job_end
            if not binding.window_start <= job_start <= job_end <= binding.window_end:
                raise ValueError("build item outside observation window")
            execution_ms = (job_end - job_start).total_seconds() * 1000
        observations.append(BuildItemMeasurement(
            item_sha256=item["bundle_item_sha256"], attempt=job["attempt"], status=job["status"],
            execution_ms=execution_ms, chunks=chunks, frozen_input_sha256=job["frozen_input_sha256"],
            embedding_configuration_sha256=canonical_json_sha256(job["embedding_configuration"]),
        ))
    measurement = BuildMeasurement(
        binding_identity=binding.identity, kind=kind, bundle_sha256=imported["bundle_sha256"],
        configuration_identity=binding.configuration, items=tuple(observations),
        elapsed_ms=(completed_at - started) * 1000,
        observed_start=observed_start, observed_end=observed_end,
    )
    report = summarize_build(binding, measurement)
    actor = _data(client.get("/api/v1/auth/me", headers=headers))
    report["tightening_decisions"] = assess_reports(
        [report], owner_sha256=canonical_json_sha256({"member": actor["username"]}), now=datetime.now(UTC),
    )
    return report
