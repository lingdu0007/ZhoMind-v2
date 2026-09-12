from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.operations.pilot_measurement import distribution


def test_small_samples_retain_values_without_percentile_verdict():
    result = distribution([100, 2000, 300], threshold_ms=1000)
    assert result == {
        "count": 3, "values_ms": [100, 2000, 300], "maximum_ms": 2000,
        "status": "observed_with_insufficient_sample",
    }
    assert distribution([], threshold_ms=1000)["status"] == "unavailable"


@pytest.mark.parametrize("bad", [True, -1, float("inf"), float("nan"), "PRIVATE_TIMING"])
def test_invalid_duration_is_rejected_without_echoing_input(bad):
    with pytest.raises(ValueError, match="invalid measurement"):
        distribution([bad], threshold_ms=1000)


def binding(**changes):
    from app.operations.pilot_measurement import MeasurementBinding

    values = {
        "deployment": "deployment:" + "a" * 64, "host": "host:" + "b" * 64,
        "configuration": "configuration:" + "c" * 64,
        "provider_route": "provider_route:" + "d" * 64,
        "retrieval_profile": "retrieval_profile:" + "e" * 64,
        "corpus": "corpus:" + "f" * 64, "chunk_envelope": "configuration:" + "1" * 64,
        "editorial_inputs_sha256": "5" * 64,
        "product_revision": "product_revision:" + "2" * 40,
        "host_vcpu": 4, "host_memory_mib": 4096,
        "active_entries": 50, "eligible_chunks": 500, "concurrency": 2,
        "window_start": datetime(2026, 9, 1, tzinfo=UTC),
        "window_end": datetime(2026, 9, 1, 1, tzinfo=UTC),
        "mode": "local_deterministic",
    }
    return MeasurementBinding(**(values | changes))


def sample(bound, **changes):
    from app.operations.pilot_measurement import RequestMeasurement

    values = {
        "request_id": uuid4(), "binding_identity": bound.identity,
        "observed_at": bound.window_start + timedelta(minutes=1),
        "population": "admitted", "state": "completed", "outcome": "evidence_gated_answer",
        "acknowledgement_ms": 100, "processing_ms": 200, "answer_content_ms": 25000,
        "closed_outcome_ms": 30000, "terminal_ms": 30001, "queue_ms": 1000,
        "retrieval_ms": 3000, "provider_ms": 22000, "persistence_ms": 1000,
        "application_controlled_ms": 8001, "route_attempt_ms": (11000, 11000),
        "route_reason": "succeeded",
    }
    return RequestMeasurement(**(values | changes))


def test_population_metrics_include_all_admitted_closed_outcomes_and_all_attempts():
    from app.operations.pilot_measurement import summarize_requests

    bound = binding()
    samples = [sample(bound) for _ in range(39)]
    samples.append(sample(bound, outcome="generation_unavailable", answer_content_ms=None, route_reason="timeout",
                          observed_at=bound.window_start + timedelta(minutes=2)))
    for population in ("invalid", "unauthorized", "canceled", "throttled", "rejected"):
        samples.append(sample(bound, population=population, state="failed", outcome=None,
                              closed_outcome_ms=None, answer_content_ms=None))
    report = summarize_requests(bound, samples)
    assert report["eligible_count"] == 40
    assert report["separate_populations"] == {key: 1 for key in ("invalid", "unauthorized", "canceled", "throttled", "rejected")}
    assert report["metrics"]["closed_outcome_ms"]["p95_ms"] == 30000
    assert report["metrics"]["closed_outcome_ms"]["status"] == "passing"
    assert report["metrics"]["answer_content_ms"]["count"] == 39
    assert report["route"]["rolling_20_success_rate"] == .95
    assert report["route"]["provider_unavailable_count"] == 1
    assert report["metrics"]["provider_ms"]["p50_ms"] == 22000
    assert report["live_acceptance"] is False


def test_duplicate_drift_and_out_of_window_samples_do_not_inherit_a_pass():
    from app.operations.pilot_measurement import summarize_requests

    bound = binding()
    original = sample(bound)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_requests(bound, [original, original])
    with pytest.raises(ValueError, match="binding"):
        summarize_requests(binding(concurrency=1), [original])
    with pytest.raises(ValueError, match="window"):
        summarize_requests(bound, [sample(bound, observed_at=bound.window_end)])
    assert binding(corpus="corpus:" + "3" * 64).identity != bound.identity
    assert binding(window_end=bound.window_end + timedelta(seconds=1)).identity != bound.identity


def test_missing_critical_timing_and_failed_admitted_request_cannot_pass():
    from app.operations.pilot_measurement import summarize_requests

    bound = binding()
    requests = [sample(bound) for _ in range(20)]
    requests[0] = sample(bound, processing_ms=None)
    requests[1] = sample(bound, state="failed", outcome=None, closed_outcome_ms=None, answer_content_ms=None)
    report = summarize_requests(bound, requests)
    assert report["eligible_count"] == 20
    assert report["metrics"]["processing_ms"]["status"] == "unavailable"
    assert report["metrics"]["closed_outcome_ms"]["status"] == "unavailable"
    assert report["admitted_failures"] == 1


def test_rolling_route_report_does_not_erase_an_earlier_failed_window():
    from app.operations.pilot_measurement import summarize_requests

    bound = binding()
    requests = [
        sample(bound, observed_at=bound.window_start + timedelta(seconds=index),
               outcome="generation_unavailable" if index < 2 else "evidence_gated_answer",
               route_reason="timeout" if index < 2 else "succeeded",
               answer_content_ms=None if index < 2 else 25000)
        for index in range(40)
    ]
    report = summarize_requests(bound, requests)
    assert report["route"]["rolling_20_success_rate"] == 1
    assert report["route"]["failed_windows"] == 1
    assert report["route"]["status"] == "at_risk"
    requests[0] = sample(bound, route_attempt_ms=(), route_reason=None)
    assert summarize_requests(bound, requests)["route"]["status"] == "unavailable"


def test_entry_suite_requires_measured_profile_complete_rounds_and_failure_free_builds():
    from app.operations.pilot_measurement import BuildMeasurement, WorkloadProfile, summarize_build

    profile = WorkloadProfile()
    assert profile.model_dump() == {
        "schema_version": 1, "admitted_members": 8, "executing": 2, "queued": 2,
        "member_executing": 1, "member_queued": 1, "background_workers": 1,
        "active_entries": 50, "maximum_eligible_chunks": 10000,
        "c1_requests": 40, "c2_requests": 40, "burst_rounds": 10, "burst_size": 4,
    }
    bound = binding()
    values = {
        "binding_identity": bound.identity, "kind": "ten_item_bundle", "bundle_sha256": "a" * 64,
        "configuration_identity": bound.configuration,
        "items": tuple({
            "item_sha256": f"{index:064x}", "attempt": 1, "status": "candidate_ready",
            "execution_ms": 300000., "chunks": 10,
        } for index in range(10)),
        "elapsed_ms": 1800000.,
        "observed_start": bound.window_start, "observed_end": bound.window_end,
    }
    result = summarize_build(bound, BuildMeasurement(**values))
    assert result["status"] == "passing"
    assert result["item_execution"]["status"] == "observed_with_insufficient_sample"
    assert result["complete_items"] == 10
    values["items"] = values["items"][:-1]
    assert summarize_build(bound, BuildMeasurement(**values))["status"] == "unavailable"


def test_every_burst_request_includes_queue_and_closes_within_total_budget():
    from app.operations.pilot_measurement import AdmissionPeak, summarize_requests

    bound = binding(concurrency=4)
    samples = [sample(bound, burst_round=round_number, queued=index >= 2,
                      queue_ms=29000, provider_ms=30000, route_attempt_ms=(15000., 15000.),
                      closed_outcome_ms=60000, terminal_ms=60001, application_controlled_ms=30001)
               for round_number in range(1, 11) for index in range(4)]
    peaks = tuple(AdmissionPeak(binding_identity=bound.identity, burst_round=index,
                                observed_at=bound.window_start + timedelta(seconds=index), executing=2, queued=2)
                  for index in range(1, 11))
    assert summarize_requests(bound, samples)["burst"]["status"] == "unavailable"
    report = summarize_requests(bound, samples, admission_peaks=peaks)
    assert report["burst"]["status"] == "passing"
    assert report["burst"]["rounds_with_two_queued"] == 10
    samples[0] = sample(bound, burst_round=1, closed_outcome_ms=60001, terminal_ms=60002)
    assert summarize_requests(bound, samples, admission_peaks=peaks)["burst"]["status"] == "at_risk"
    assert summarize_requests(bound, samples[:-1], admission_peaks=peaks)["burst"]["status"] == "unavailable"
