"""Recompute decisions from explicit retained reports, never their verdict flags."""

import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from functools import partial
from itertools import pairwise

from app.common.canonical_json import canonical_json_sha256
from app.operations.pilot_measurement import (
    AdmissionPeak,
    BuildMeasurement,
    MeasurementBinding,
    RequestMeasurement,
    route_counts,
    summarize_build,
    summarize_requests,
)
from app.operations.pilot_triggers import TriggerObservation, evaluate_triggers


def _append_observation(
    observations: list[TriggerObservation], evidence_objects: dict, binding: MeasurementBinding, sources: list[dict],
    kind: str, values: tuple[float, ...], *, counts: tuple[int, ...] = (), complete: bool = True,
) -> None:
    evidence = {"sources": sources, "kind": kind, "values": values, "counts": counts, "complete": complete}
    digest = canonical_json_sha256(evidence)
    evidence_objects[digest] = evidence
    observations.append(TriggerObservation.model_validate({
        "binding_identity": binding.identity,
        "evidence_sha256": digest,
        "kind": kind, "values": values, "eligible_counts": counts, "complete_window": complete,
    }))


def assess_reports(reports: Sequence[dict], *, owner_sha256: str, now: datetime) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    seen_reports, seen_requests, seen_builds = set(), set(), set()
    for report in reports:
        if report.get("schema") == "pilot_entry_requests/v1":
            requests = [RequestMeasurement.model_validate_json(json.dumps(item)) for item in report["samples"]]
            for sample in requests:
                if sample.request_id in seen_requests or sample.observed_at > now:
                    raise ValueError("duplicate or future request evidence")
                seen_requests.add(sample.request_id)
            windows = []
            for raw in report["request_reports"]:
                binding = MeasurementBinding.model_validate_json(json.dumps(raw["binding"]))
                samples = [sample for sample in requests if sample.binding_identity == binding.identity]
                peaks = [AdmissionPeak.model_validate_json(json.dumps(item)) for item in raw.get("burst", {}).get("admission_peaks", [])]
                summary = summarize_requests(binding, samples, admission_peaks=peaks)
                evidence = {
                    "binding": binding.model_dump(mode="json"),
                    "samples": [sample.model_dump(mode="json") for sample in samples],
                    "admission_peaks": [peak.model_dump(mode="json") for peak in peaks],
                }
                windows.append({"binding": binding, "samples": samples, "summary": summary, "build": None, "evidence": evidence})
            if sum(len(window["samples"]) for window in windows) != len(requests):
                raise ValueError("request evidence has an unbound window")
        elif report.get("schema") == "pilot_build_report/v1":
            binding = MeasurementBinding.model_validate_json(json.dumps(report["binding"]))
            build = BuildMeasurement.model_validate_json(json.dumps(report["observation"]))
            if build.observed_end > now:
                raise ValueError("future build evidence")
            execution = canonical_json_sha256({
                "bundle": build.bundle_sha256,
                "items": sorted((item.item_sha256, item.attempt) for item in build.items),
                "started": build.observed_start.astimezone(UTC).isoformat(),
                "ended": build.observed_end.astimezone(UTC).isoformat(),
            })
            if execution in seen_builds:
                raise ValueError("duplicate build execution")
            seen_builds.add(execution)
            evidence = {"binding": binding.model_dump(mode="json"), "observation": build.model_dump(mode="json")}
            windows = [{"binding": binding, "samples": [], "summary": summarize_build(binding, build),
                        "build": build, "evidence": evidence}]
        else:
            raise ValueError("unsupported measurement report")
        for window in windows:
            binding = window["binding"]
            key = canonical_json_sha256(binding.model_dump(mode="json", exclude={"window_start", "window_end"}))
            identity = canonical_json_sha256({"binding": binding.identity, "summary": window["summary"]})
            if identity in seen_reports:
                raise ValueError("duplicate report evidence")
            seen_reports.add(identity)
            groups[key].append(window)

    decisions = []
    for windows in groups.values():
        windows.sort(key=lambda window: window["binding"].window_start)
        binding = windows[-1]["binding"]
        source_bindings = [window["binding"].identity for window in windows]
        observations = []
        evidence_objects: dict = {}

        add = partial(_append_observation, observations, evidence_objects, binding, [window["evidence"] for window in windows])

        builds = [window["build"] for window in windows if window["build"] is not None]
        for kind, trigger in (("ten_item_bundle", "bundle_duration"), ("fifty_entry_rebuild", "rebuild_duration")):
            durations = tuple(build.elapsed_ms for build in builds if build.kind == kind)
            if durations:
                add(trigger, durations)
        request_windows = [window for window in windows if window["build"] is None]
        samples = sorted([sample for window in request_windows for sample in window["samples"]
                          if sample.population == "admitted"], key=lambda sample: sample.observed_at)
        for previous, current in pairwise(request_windows):
            pair = [previous, current]
            ordered = previous["binding"].window_end <= current["binding"].window_start
            closed = current["binding"].window_end <= now
            for metric, trigger in (("closed_outcome_ms", "normal_latency"), ("provider_ms", "provider_latency")):
                if trigger == "normal_latency" and binding.concurrency != 2:
                    continue
                stats = [window["summary"]["metrics"][metric] for window in pair]
                _append_observation(
                    observations, evidence_objects, binding, [window["evidence"] for window in pair],
                    trigger, tuple(stat.get("p95_ms", 0.) for stat in stats),
                    counts=tuple(stat["count"] for stat in stats),
                    complete=ordered and closed and all(stat["missing_count"] == 0 for stat in stats))
        if binding.concurrency == 4:
            # Queue attribution needs an overrun that would fit without its measured wait.
            add("capacity_burst", tuple(
                sample.closed_outcome_ms for sample in samples
                if sample.queued and sample.closed_outcome_ms is not None and sample.queue_ms is not None
                and sample.queue_ms > 0 and sample.closed_outcome_ms - sample.queue_ms <= 60000
            ))
            days = set()
            for window in request_windows:
                for peak in window["summary"]["burst"].get("admission_peaks", []):
                    day = (now.date() - datetime.fromisoformat(peak["observed_at"]).date()).days
                    if 0 <= day < 7:
                        days.add(float(day))
            if days:
                add("queue_saturation_days", tuple(sorted(days)))
        generation = [sample for sample in samples if sample.outcome in {"evidence_gated_answer", "generation_unavailable"}
                      or sample.route_attempt_ms]
        if len(generation) >= 20:
            successes = min(route_counts(generation[index - 20:index])[0] for index in range(20, len(generation) + 1))
            add("route_success", (float(successes), 20.))
        unavailable = tuple((now - sample.observed_at).total_seconds() * 1000 for sample in generation
                            if route_counts([sample])[1]
                            and timedelta(0) <= now - sample.observed_at < timedelta(days=7))
        if unavailable:
            add("provider_unavailable", unavailable)
        for decision in evaluate_triggers(binding, observations, owner_sha256=owner_sha256, now=now):
            decisions.append({
                **decision, "source_binding_identities": source_bindings,
                "evidence": evidence_objects.get(decision["evidence_sha256"], binding.model_dump(mode="json")),
            })
    return decisions
