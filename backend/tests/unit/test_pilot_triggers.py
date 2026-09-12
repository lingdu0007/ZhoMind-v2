from datetime import timedelta

import pytest

from tests.unit.test_pilot_measurement import binding


@pytest.mark.parametrize(("kind", "values", "expected"), [
    ("normal_latency", (31000., 31000.), "review_application_provider_retrieval_host"),
    ("capacity_burst", (61000., 61000.), "review_concurrency_queue_resources"),
    ("queue_saturation_days", (1., 3., 5.), "remeasure_burst_capacity"),
    ("provider_unavailable", (1., 2.), "begin_fallback_approval"),
    ("route_success", (18., 20.), "begin_fallback_approval"),
    ("provider_latency", (31000., 31000.), "review_provider_route"),
    ("provider_work_outage", (1800000.,), "begin_fallback_approval"),
    ("bundle_duration", (1800001., 1800001.), "review_worker_isolation_indexing"),
    ("rebuild_duration", (3600001.,), "review_rebuild_index_architecture"),
    ("availability", (98., 100.), "review_deployment_monitoring_availability"),
    ("reconstruction_duration", (14400001.,), "review_off_host_backup_restore"),
    ("restart_duration", (900001., 900001.), "review_startup_recovery"),
])
def test_each_accepted_trigger_creates_owned_bounded_decision(kind, values, expected):
    from app.operations.pilot_triggers import TriggerObservation, evaluate_triggers

    bound = binding(active_entries=20)
    evidence = TriggerObservation(
        binding_identity=bound.identity, evidence_sha256="a" * 64, kind=kind,
        values=values, eligible_counts=(20, 20), complete_window=True,
    )
    decisions = evaluate_triggers(bound, [evidence], owner_sha256="b" * 64, now=bound.window_end)
    assert len(decisions) == 1
    assert decisions[0]["required_action"] == expected
    assert decisions[0]["status"] == "at_risk"
    assert decisions[0]["content_action"] == "none"
    assert decisions[0]["review_due_at"] == (bound.window_end + timedelta(days=7)).isoformat()


def test_capacity_trigger_and_insufficient_windows_do_not_fabricate_percentile_failure():
    from app.operations.pilot_triggers import TriggerObservation, evaluate_triggers

    bound = binding(active_entries=40)
    small = TriggerObservation(
        binding_identity=bound.identity, evidence_sha256="a" * 64, kind="normal_latency",
        values=(90000., 90000.), eligible_counts=(19, 19), complete_window=True,
    )
    decisions = evaluate_triggers(bound, [small], owner_sha256="b" * 64, now=bound.window_end)
    assert {item["trigger"] for item in decisions} == {"capacity_headroom"}
