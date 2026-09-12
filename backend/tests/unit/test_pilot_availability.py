from datetime import timedelta

from tests.unit.test_pilot_measurement import binding


def observations():
    from app.operations.pilot_availability import CoreWindow, ProbeMeasurement

    bound = binding()
    bound = binding(window_end=bound.window_start + timedelta(days=28))
    windows, probes = [], []
    for day in range(28):
        start = bound.window_start + timedelta(days=day)
        windows.append(CoreWindow(start=start, end=start + timedelta(minutes=10)))
        for minute in range(10):
            probes.append(ProbeMeasurement(
                binding_identity=bound.identity, at=start + timedelta(minutes=minute),
                healthy=True, product_checked=True, product_outcome="generation_unavailable",
            ))
    return bound, windows, probes


def test_four_week_availability_counts_generation_unavailable_as_product_available():
    from app.operations.pilot_availability import summarize_availability

    bound, windows, probes = observations()
    report = summarize_availability(bound, windows, probes, observed_until=bound.window_end)
    assert report["expected_minutes"] == 280
    assert report["available_minutes"] == 280
    assert report["availability"] == 1
    assert report["status"] == "passing"
    assert report["live_acceptance"] is False


def test_missing_health_product_detection_and_short_observation_never_claim_four_week_pass():
    from app.operations.pilot_availability import summarize_availability

    bound, windows, probes = observations()
    assert summarize_availability(bound, windows, probes[:-1], observed_until=bound.window_end)["status"] == "unavailable"
    until = bound.window_end - timedelta(days=1)
    assert summarize_availability(bound, windows, [p for p in probes if p.at < until], observed_until=until)["status"] == "unavailable"
    absent = [probe.model_copy(update={"product_checked": False, "product_outcome": None}) for probe in probes]
    result = summarize_availability(bound, windows, absent, observed_until=bound.window_end)
    assert result["status"] == "unavailable"
    assert result["product_detection_gaps"] == 280


def test_failed_health_is_not_removed_from_availability_denominator():
    from app.operations.pilot_availability import summarize_availability

    bound, windows, probes = observations()
    probes[:3] = [probe.model_copy(update={"healthy": False}) for probe in probes[:3]]
    result = summarize_availability(bound, windows, probes, observed_until=bound.window_end)
    assert result["availability"] == 277 / 280
    assert result["status"] == "at_risk"


def test_reloaded_social_probe_cannot_certify_knowledge_path_availability():
    from app.operations.pilot_availability import ProbeMeasurement, summarize_availability

    bound, windows, probes = observations()
    reloaded = [
        ProbeMeasurement.model_validate_json(
            probe.model_copy(update={"product_outcome": "non_knowledge_base_reply"}).model_dump_json(),
        )
        for probe in probes
    ]
    report = summarize_availability(bound, windows, reloaded, observed_until=bound.window_end)
    assert report["available_minutes"] == 0
    assert report["status"] == "at_risk"


def test_six_minute_probe_cadence_cannot_hide_behind_minute_rounding():
    from app.operations.pilot_availability import summarize_availability

    bound, windows, probes = observations()
    probes = [probe.model_copy(update={
        "at": probe.at + timedelta(seconds=1),
        "product_checked": probe.at.minute % 6 == 0,
        "product_outcome": "generation_unavailable" if probe.at.minute % 6 == 0 else None,
    }) for probe in probes]
    report = summarize_availability(bound, windows, probes, observed_until=bound.window_end)
    assert report["product_detection_gaps"] > 0
    assert report["status"] == "unavailable"
