from __future__ import annotations

from collections.abc import Mapping

from app.reviewed_bundles.models import CandidateBuildJob


def candidate_job_event_payload(
    job: CandidateBuildJob,
    *,
    action: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "candidate_build_job_event/v1",
        "action": action,
        "stage": job.stage,
        "status": job.status,
        "progress": job.progress,
        "attempt": job.attempt,
        "editorial_source_revision": job.editorial_source_revision,
        "input_sha256": job.input_sha256,
        "failure_reason": dict(job.failure_reason) if isinstance(job.failure_reason, dict) else None,
        "allowed_next_action": job.allowed_next_action,
    }
    if extra is not None:
        payload.update(extra)
    return payload
