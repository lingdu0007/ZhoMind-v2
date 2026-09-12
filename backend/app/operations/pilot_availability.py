"""Evaluate a declared four-week core window without inventing missing uptime."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import AwareDatetime, model_validator

from app.common.canonical_json import canonical_json_sha256
from app.operations.pilot_measurement import ConfigurationIdentity, MeasurementBinding, Observation


class CoreWindow(Observation):
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def minute_aligned(self) -> Self:
        if self.end <= self.start or any(value.second or value.microsecond for value in (self.start, self.end)):
            raise ValueError("core window requires ordered whole minutes")
        return self


class ProbeMeasurement(Observation):
    binding_identity: ConfigurationIdentity
    at: AwareDatetime
    healthy: bool
    product_checked: bool
    product_outcome: Literal[
        "evidence_gated_answer", "insufficient_evidence_reply", "non_knowledge_base_reply", "generation_unavailable",
    ] | None = None

    @model_validator(mode="after")
    def checked_outcome(self) -> Self:
        if not self.product_checked and self.product_outcome is not None:
            raise ValueError("unchecked probe cannot have a product outcome")
        return self


def summarize_availability(
    binding: MeasurementBinding, core_windows: Sequence[CoreWindow],
    probes: Sequence[ProbeMeasurement], *, observed_until: datetime,
) -> dict:
    expected: set[datetime] = set()
    windows = sorted(core_windows, key=lambda window: window.start)
    for window in windows:
        if not binding.window_start <= window.start < window.end <= binding.window_end:
            raise ValueError("core window outside measurement binding")
        minute = window.start
        while minute < window.end:
            if minute in expected:
                raise ValueError("overlapping core window")
            expected.add(minute)
            minute += timedelta(minutes=1)
    by_minute = {}
    for probe in probes:
        if probe.binding_identity != binding.identity:
            raise ValueError("probe binding mismatch")
        minute = probe.at.replace(second=0, microsecond=0)
        if minute not in expected or probe.at > observed_until:
            raise ValueError("probe outside observed core window")
        if minute in by_minute:
            raise ValueError("duplicate minute probe")
        by_minute[minute] = probe
    available = health_missing = product_gaps = 0
    for window in windows:
        last_product: ProbeMeasurement | None = None
        minute = window.start
        while minute < window.end:
            probe = by_minute.get(minute)
            if probe is None:
                health_missing += 1
            if probe is not None and probe.product_checked:
                last_product = probe
            checked_at = probe.at if probe is not None else minute + timedelta(minutes=1)
            current_product = last_product is not None and checked_at - last_product.at < timedelta(minutes=5)
            if not current_product:
                product_gaps += 1
            if (probe is not None and probe.healthy and current_product and last_product is not None
                    and last_product.product_outcome in {"evidence_gated_answer", "generation_unavailable"}):
                available += 1
            minute += timedelta(minutes=1)
    ratio = available / len(expected) if expected else None
    covered_weeks = {int((minute - binding.window_start).total_seconds() // (7 * 86400)) for minute in expected}
    complete_window = (
        binding.window_end - binding.window_start == timedelta(days=28)
        and observed_until >= binding.window_end and covered_weeks == {0, 1, 2, 3}
    )
    status = "passing" if ratio is not None and ratio >= .99 else "at_risk"
    if not complete_window or not expected or health_missing or product_gaps:
        status = "unavailable"
    return {
        "schema": "pilot_availability_report/v1", "binding_identity": binding.identity,
        "core_schedule_sha256": canonical_json_sha256([window.model_dump(mode="json") for window in windows]),
        "expected_minutes": len(expected), "available_minutes": available,
        "health_missing_minutes": health_missing, "product_detection_gaps": product_gaps,
        "availability": ratio, "complete_four_week_window": complete_window,
        "status": status, "live_acceptance": False,
    }
