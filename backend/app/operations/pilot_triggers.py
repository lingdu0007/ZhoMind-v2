"""Evidence-linked evaluator decisions; no publication or provider activation."""

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import Field

from app.common.canonical_json import canonical_json_sha256
from app.operations.pilot_measurement import ConfigurationIdentity, Count, MeasurementBinding, Milliseconds, Observation


class TriggerObservation(Observation):
    binding_identity: ConfigurationIdentity
    evidence_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    kind: Literal[
        "normal_latency", "capacity_burst", "queue_saturation_days", "provider_unavailable",
        "route_success", "provider_latency", "provider_work_outage", "bundle_duration",
        "rebuild_duration", "availability", "reconstruction_duration", "restart_duration",
        "early_retrieval_failure",
    ]
    values: tuple[Milliseconds, ...]
    eligible_counts: tuple[Count, ...] = ()
    complete_window: bool = False


def _required_action(observation: TriggerObservation) -> str | None:
    values = observation.values
    kind = observation.kind
    if kind in {"normal_latency", "provider_latency"}:
        if (len(values) >= 2 and len(observation.eligible_counts) == len(values)
                and all(count >= 20 for count in observation.eligible_counts[-2:])
                and all(value > 30000 for value in values[-2:]) and observation.complete_window):
            return "review_application_provider_retrieval_host" if kind == "normal_latency" else "review_provider_route"
    elif kind == "capacity_burst" and sum(value > 60000 for value in values) >= 2:
        return "review_concurrency_queue_resources"
    elif kind == "queue_saturation_days":
        # Values are distinct day offsets in the evidence's declared rolling week.
        if observation.complete_window and len(set(values)) >= 3 and all(value in range(7) for value in values):
            return "remeasure_burst_capacity"
    elif kind == "provider_unavailable":
        if observation.complete_window and len(values) >= 2 and all(value < 7 * 86400000 for value in values):
            return "begin_fallback_approval"
    elif kind == "route_success" and len(values) == 2 and values[1] == 20 and 0 <= values[0] < 19:
        return "begin_fallback_approval"
    elif kind == "provider_work_outage" and any(value >= 1800000 for value in values):
        return "begin_fallback_approval"
    elif kind == "bundle_duration" and sum(value > 1800000 for value in values) >= 2:
        return "review_worker_isolation_indexing"
    elif kind == "rebuild_duration" and any(value > 3600000 for value in values):
        return "review_rebuild_index_architecture"
    elif kind == "availability" and len(values) == 2 and observation.complete_window:
        if values[1] > 0 and values[0] / values[1] < .99:
            return "review_deployment_monitoring_availability"
    elif kind == "reconstruction_duration" and any(value > 14400000 for value in values):
        return "review_off_host_backup_restore"
    elif kind == "restart_duration" and sum(value > 900000 for value in values) >= 2:
        return "review_startup_recovery"
    elif kind == "early_retrieval_failure" and values:
        return "remeasure_tune_archive_or_scale"
    return None


def evaluate_triggers(
    binding: MeasurementBinding, observations: Sequence[TriggerObservation], *, owner_sha256: str, now: datetime,
) -> list[dict]:
    if not re.fullmatch(r"[0-9a-f]{64}", owner_sha256) or now.tzinfo is None:
        raise ValueError("invalid decision owner or time")
    decisions = []

    def append(kind: str, action: str, evidence: str) -> None:
        payload = {
            "schema": "pilot_objective_decision/v1", "binding_identity": binding.identity,
            "trigger": kind, "required_action": action, "evidence_sha256": evidence,
            "owner_sha256": owner_sha256, "status": "at_risk", "content_action": "none",
            "review_due_at": (now + timedelta(days=7)).isoformat(),
            "allowed_resolutions": ["remediation", "narrower_commitment", "re_acceptance", "suspension"],
        }
        decisions.append({**payload, "identity": "configuration:" + canonical_json_sha256(payload)})

    if binding.active_entries >= 40 or binding.eligible_chunks >= 8000:
        append("capacity_headroom", "remeasure_tune_archive_or_scale", binding.identity.split(":")[1])
    seen = set()
    for observation in observations:
        if observation.binding_identity != binding.identity:
            raise ValueError("trigger observation binding mismatch")
        identity = (observation.kind, observation.evidence_sha256)
        if identity in seen:
            raise ValueError("duplicate trigger evidence")
        seen.add(identity)
        action = _required_action(observation)
        if action:
            append(observation.kind, action, observation.evidence_sha256)
    return decisions
