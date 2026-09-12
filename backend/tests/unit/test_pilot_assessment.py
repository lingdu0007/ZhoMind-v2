import json
from datetime import timedelta, timezone

import pytest

from tests.unit.test_pilot_measurement import binding


def build_report(*, kind="fifty_entry_rebuild", minutes=61, bound=None):
    from app.operations.pilot_measurement import BuildMeasurement, summarize_build

    bound = bound or binding()
    bound = binding(window_start=bound.window_start, window_end=bound.window_start + timedelta(hours=2))
    count = 50 if kind == "fifty_entry_rebuild" else 10
    observation = BuildMeasurement(
        binding_identity=bound.identity, configuration_identity=bound.configuration,
        bundle_sha256="a" * 64, kind=kind, elapsed_ms=float(minutes * 60000),
        observed_start=bound.window_start, observed_end=bound.window_start + timedelta(minutes=minutes),
        items=tuple({"item_sha256": f"{i:064x}", "status": "candidate_ready", "attempt": 1,
                     "execution_ms": 100., "chunks": 10} for i in range(count)),
    )
    return summarize_build(bound, observation)


def test_rebuild_miss_automatically_creates_required_indexing_decision():
    from app.operations.pilot_assessment import assess_reports

    report = build_report()
    decisions = assess_reports([report], owner_sha256="b" * 64, now=binding().window_end + timedelta(hours=3))
    assert any(item["required_action"] == "review_rebuild_index_architecture" for item in decisions)


def test_two_complete_bundle_reports_trigger_worker_review_without_manual_trigger_flags():
    from app.operations.pilot_assessment import assess_reports

    first = build_report(kind="ten_item_bundle", minutes=31)
    second = build_report(kind="ten_item_bundle", minutes=32,
                          bound=binding(window_start=binding().window_start + timedelta(days=1),
                                        window_end=binding().window_end + timedelta(days=1)))
    decisions = assess_reports([first, second], owner_sha256="b" * 64,
                               now=binding().window_end + timedelta(days=2))
    assert any(item["required_action"] == "review_worker_isolation_indexing" for item in decisions)


def test_capacity_decision_requires_queue_contribution_to_budget_overrun():
    from app.operations.pilot_assessment import assess_reports
    from app.operations.pilot_measurement import summarize_requests
    from tests.unit.test_pilot_measurement import sample

    bound = binding(concurrency=4)

    def decisions(queue_ms):
        samples = [sample(bound, queued=True, queue_ms=queue_ms, closed_outcome_ms=90000,
                          terminal_ms=90001, burst_round=1) for _ in range(2)]
        report = {"schema": "pilot_entry_requests/v1",
                  "samples": [item.model_dump(mode="json") for item in samples],
                  "request_reports": [summarize_requests(bound, samples)]}
        return assess_reports([report], owner_sha256="b" * 64, now=bound.window_end)

    assert not any(item["trigger"] == "capacity_burst" for item in decisions(1000))
    assert any(item["trigger"] == "capacity_burst" for item in decisions(40000))


def request_report(offset, duration, *, concurrency=2):
    from app.operations.pilot_measurement import summarize_requests
    from tests.unit.test_pilot_measurement import sample

    initial = binding()
    bound = binding(concurrency=concurrency,
                    window_start=initial.window_start + timedelta(hours=offset),
                    window_end=initial.window_end + timedelta(hours=offset))
    samples = [sample(bound, closed_outcome_ms=duration, terminal_ms=duration + 1,
                      provider_ms=duration, route_attempt_ms=(float(duration),)) for _ in range(20)]
    return {"schema": "pilot_entry_requests/v1",
            "samples": [item.model_dump(mode="json") for item in samples],
            "request_reports": [summarize_requests(bound, samples)]}


def test_open_window_cannot_supply_consecutive_latency_evidence():
    from app.operations.pilot_assessment import assess_reports

    decisions = assess_reports([request_report(0, 31000), request_report(1, 31000)],
                               owner_sha256="b" * 64, now=binding().window_end + timedelta(minutes=2))
    assert not any(item["trigger"] in {"normal_latency", "provider_latency"} for item in decisions)


def test_later_pass_cannot_erase_historical_consecutive_miss():
    from app.operations.pilot_assessment import assess_reports

    decisions = assess_reports([request_report(0, 31000), request_report(1, 31000), request_report(2, 30000)],
                               owner_sha256="b" * 64, now=binding().window_end + timedelta(hours=3))
    assert any(item["trigger"] == "normal_latency" for item in decisions)


def test_provider_latency_is_not_limited_to_concurrency_two():
    from app.operations.pilot_assessment import assess_reports

    decisions = assess_reports([request_report(0, 31000, concurrency=1), request_report(1, 31000, concurrency=1)],
                               owner_sha256="b" * 64, now=binding().window_end + timedelta(hours=2))
    assert any(item["trigger"] == "provider_latency" for item in decisions)
    assert not any(item["trigger"] == "normal_latency" for item in decisions)


def test_relabeling_window_does_not_create_a_second_build_execution():
    from app.operations.pilot_assessment import assess_reports
    from app.operations.pilot_measurement import BuildMeasurement, summarize_build

    first = build_report(kind="ten_item_bundle", minutes=31)
    bound = binding(window_end=binding().window_end + timedelta(hours=3))
    observation = BuildMeasurement.model_validate_json(json.dumps(first["observation"]))
    second = summarize_build(bound, observation.model_copy(update={"binding_identity": bound.identity}))
    with pytest.raises(ValueError, match="duplicate build"):
        assess_reports([first, second], owner_sha256="b" * 64, now=bound.window_end)


def test_equivalent_timezone_does_not_create_a_second_build_execution():
    from app.operations.pilot_assessment import assess_reports
    from app.operations.pilot_measurement import BuildMeasurement, MeasurementBinding, summarize_build

    first = build_report(kind="ten_item_bundle", minutes=31)
    bound = MeasurementBinding.model_validate_json(json.dumps(first["binding"]))
    original = BuildMeasurement.model_validate_json(json.dumps(first["observation"]))
    offset = timezone(timedelta(hours=8))
    second = summarize_build(bound, original.model_copy(update={
        "observed_start": original.observed_start.astimezone(offset),
        "observed_end": original.observed_end.astimezone(offset),
    }))
    with pytest.raises(ValueError, match="duplicate build"):
        assess_reports([first, second], owner_sha256="b" * 64, now=bound.window_end)


def test_decision_evidence_binds_exact_raw_samples_and_is_independently_hashable():
    from app.common.canonical_json import canonical_json_sha256
    from app.operations.pilot_assessment import assess_reports

    def decision():
        reports = [request_report(0, 31000), request_report(1, 31000)]
        decisions = assess_reports(reports, owner_sha256="b" * 64,
                                   now=binding().window_end + timedelta(hours=2))
        return next(item for item in decisions if item["trigger"] == "normal_latency")

    first, second = decision(), decision()
    assert first["evidence_sha256"] != second["evidence_sha256"]
    assert canonical_json_sha256(first["evidence"]) == first["evidence_sha256"]
