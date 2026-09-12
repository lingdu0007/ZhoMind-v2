"""A consuming SSE observer: retain timings and closed classes, never content."""

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from uuid import UUID

from app.common.canonical_json import canonical_json_sha256
from app.operations.events import OperationalEventService
from app.operations.pilot_measurement import MeasurementBinding, RequestMeasurement

_OUTCOMES = {"evidence_gated_answer", "insufficient_evidence_reply", "non_knowledge_base_reply", "generation_unavailable"}


def measure_stream(
    binding: MeasurementBinding, frames: Iterable[tuple[str, str, float]], *,
    request_id: UUID, observed_at: datetime, terminal_ms: float,
    operation: dict | None, burst_round: int | None = None, verified_message: dict | None = None,
) -> RequestMeasurement:
    acknowledgement = processing = content = closed = None
    execution_outcome = outcome = state = answer_id = None
    failure = None
    queued = False
    done = False
    invalid = False
    cancellation = False
    terminal_events: set[str] = set()
    previous_ms = 0.
    execution_hash = summary_hash = insufficiency_hash = None
    content_hash = hashlib.sha256()
    try:
        for event, raw, elapsed in frames:
            if elapsed < previous_ms or elapsed > terminal_ms or done:
                invalid = True
            previous_ms = elapsed
            if event == "done":
                done = raw == "[DONE]"
                invalid |= not done
                closed = elapsed if done else None
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                invalid = True
                continue
            if event == "answer_identity":
                invalid |= answer_id is not None or not isinstance(value.get("answer_id"), str) or not value.get("answer_id")
                answer_id = value.get("answer_id")
                acknowledgement = elapsed if acknowledgement is None else acknowledgement
            elif event == "stage":
                stage = value.get("stage")
                if stage in {"queued", "running"} and acknowledgement is None:
                    acknowledgement = elapsed
                queued |= stage == "queued"
                if stage in {"running", "retrieval", "generating", "persistence"} and processing is None:
                    processing = elapsed
            elif event == "answer_execution":
                execution = value.get("answer_execution")
                invalid |= event in terminal_events or not isinstance(execution, dict)
                terminal_events.add(event)
                if isinstance(execution, dict):
                    execution_hash = canonical_json_sha256(execution)
                    state = execution.get("state")
                    execution_outcome = execution.get("outcome")
                    invalid |= state not in ("completed", "stopped")
                    invalid |= execution.get("assistant_message_id") != answer_id
            elif event == "outcome":
                invalid |= event in terminal_events
                terminal_events.add(event)
                outcome = value.get("outcome")
                invalid |= not isinstance(outcome, str) or outcome not in _OUTCOMES
            elif event == "content":
                if not isinstance(value.get("content"), str):
                    invalid = True
                else:
                    content_hash.update(value["content"].encode())
                    if value["content"] and content is None:
                        content = elapsed
            elif event == "evidence_summary":
                invalid |= summary_hash is not None
                summary_hash = canonical_json_sha256(value.get("evidence_summary"))
            elif event == "insufficient_evidence_reply":
                invalid |= insufficiency_hash is not None
                insufficiency_hash = canonical_json_sha256(value.get("insufficient_evidence_reply"))
            elif event == "error":
                code = OperationalEventService.error_code(value.get("code"))
                if code == "ANSWER_EXECUTION_STOPPED" and state == "stopped":
                    cancellation = True
                    continue
                failure = {
                    "CHAT_QUEUE_TIMEOUT": "queue", "CHAT_QUEUE_FULL": "queue", "CHAT_MEMBER_LIMIT": "queue",
                    "ANSWER_EXECUTION_PERSISTENCE_FAILED": "persistence", "RETRIEVAL_FAILED": "retrieval",
                }.get(code or "", "stream")
                invalid = True
    except (ValueError, TypeError):
        invalid = True
    canceled = cancellation and state == "stopped" and outcome is None and execution_outcome is None
    completed = state == "completed" and isinstance(outcome, str) and outcome in _OUTCOMES and outcome == execution_outcome
    if completed:
        persisted = verified_message or {}
        persisted_execution = persisted.get("answer_execution")
        invalid |= not isinstance(answer_id, str) or not answer_id or persisted.get("id") != answer_id
        invalid |= not isinstance(persisted_execution, dict) or canonical_json_sha256(persisted_execution) != execution_hash
        invalid |= persisted.get("outcome") != outcome
        text = persisted.get("content")
        invalid |= (
            not isinstance(text, str)
            or hashlib.sha256((text if isinstance(text, str) else "").encode()).digest() != content_hash.digest()
        )
        for key, observed_hash in (("evidence_summary", summary_hash), ("insufficient_evidence_reply", insufficiency_hash)):
            expected = persisted.get(key)
            invalid |= (canonical_json_sha256(expected) if expected is not None else None) != observed_hash
    invalid |= not done or not (canceled or completed)
    if invalid:
        outcome = None
        closed = content = None
        state = "failed"
        failure = failure or "stream"
    if outcome == "generation_unavailable":
        content = None
    if canceled and not invalid:
        closed = content = None
    timings: dict = {}
    route: dict = {}
    if operation is not None:
        if operation.get("request_id") != str(request_id):
            raise ValueError("request measurement binding mismatch")
        dimensions = OperationalEventService.dimensions(operation.get("dimensions"))
        if dimensions.get("configuration_identity") != binding.configuration:
            raise ValueError("admission configuration binding mismatch")
        timings = dimensions.get("stage_durations_ms", {})
        route = OperationalEventService.retained_route(operation.get("generation_route")) or {}
        if route.get("route_identity") not in (None, binding.provider_route):
            raise ValueError("provider route binding mismatch")
        if invalid and operation.get("normalized_error") is not None:
            failure = OperationalEventService.failure_category(operation["normalized_error"])
            if failure == "generation_provider":
                failure = "provider"
        if not invalid and not canceled and (dimensions.get("outcome") != outcome or dimensions.get("execution_state") != state):
            raise ValueError("operational outcome contradicts product path")
    provider = timings.get("provider")
    population = "canceled" if canceled and not invalid else "admitted"
    if failure == "queue" and answer_id is None:
        population = "throttled"
        state = "throttled"
    return RequestMeasurement.model_validate(dict(
        request_id=request_id, binding_identity=binding.identity, observed_at=observed_at,
        population=population, state=state, outcome=outcome,
        acknowledgement_ms=acknowledgement, processing_ms=processing, answer_content_ms=content,
        closed_outcome_ms=closed, terminal_ms=terminal_ms,
        queue_ms=timings.get("queue"), retrieval_ms=timings.get("retrieval"),
        provider_ms=provider, persistence_ms=timings.get("persistence"),
        application_controlled_ms=terminal_ms - provider if provider is not None else None,
        route_attempt_ms=tuple(item["latency_ms"] for item in route.get("attempts", []) if "latency_ms" in item),
        route_reason=route.get("route_reason"), normalized_failure=failure,
        burst_round=burst_round, queued=queued,
    ))
