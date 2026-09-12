import json
from uuid import uuid4

import pytest

from tests.unit.test_pilot_measurement import binding


def frames():
    yield "answer_identity", '{"answer_id":"reserved"}', 50.
    yield "stage", '{"stage":"queued"}', 60.
    yield "stage", '{"stage":"running"}', 150.
    yield "answer_execution", json.dumps({"answer_execution": {
        "state": "completed", "outcome": "evidence_gated_answer", "assistant_message_id": "reserved",
    }}), 25000.
    yield "outcome", '{"outcome":"evidence_gated_answer"}', 25001.
    yield "content", '{"content":"PRIVATE_ANSWER_DO_NOT_RETAIN"}', 26000.
    yield "done", "[DONE]", 30000.


def operation(bound, request_id):
    return {
        "request_id": str(request_id), "duration_ms": 30001,
        "dimensions": {
            "configuration_identity": bound.configuration,
            "execution_state": "completed", "outcome": "evidence_gated_answer",
            "stage_durations_ms": {"queue": 1000, "retrieval": 2000, "provider": 22000, "persistence": 1000},
        },
        "generation_route": {"route_identity": bound.provider_route, "route_reason": "succeeded",
                             "attempts": [{"latency_ms": 11000}, {"latency_ms": 11000}]},
    }


def verified_message():
    return {
        "id": "reserved", "content": "PRIVATE_ANSWER_DO_NOT_RETAIN", "outcome": "evidence_gated_answer",
        "answer_execution": {"state": "completed", "outcome": "evidence_gated_answer", "assistant_message_id": "reserved"},
    }


def test_collector_distinguishes_ack_progress_content_and_closed_terminal_without_retaining_content():
    from app.operations.pilot_collector import measure_stream

    bound, request_id = binding(), uuid4()
    sample = measure_stream(
        bound, frames(), request_id=request_id, observed_at=bound.window_start,
        terminal_ms=30001, operation=operation(bound, request_id), verified_message=verified_message(),
    )
    assert sample.acknowledgement_ms == 50
    assert sample.processing_ms == 150
    assert sample.answer_content_ms == 26000
    assert sample.closed_outcome_ms == 30000
    assert sample.queue_ms == 1000
    assert sample.provider_ms == 22000
    assert sample.application_controlled_ms == 8001
    assert sample.route_attempt_ms == (11000, 11000)
    assert "PRIVATE_" not in sample.model_dump_json()


def test_interrupted_or_contradictory_stream_is_not_a_closed_outcome():
    from app.operations.pilot_collector import measure_stream

    bound, request_id = binding(), uuid4()
    for stream in (
        list(frames())[:-1],
        list(frames()) + [("error", '{"message":"PRIVATE_ERROR"}', 30001)],
        list(frames()) + [("done", "[DONE]", 30001)],
    ):
        sample = measure_stream(
            bound, stream, request_id=request_id, observed_at=bound.window_start,
            terminal_ms=30002, operation=operation(bound, request_id),
        )
        assert sample.state == "failed"
        assert sample.outcome is None
        assert sample.closed_outcome_ms is None
        assert sample.normalized_failure == "stream"
        assert "PRIVATE_" not in sample.model_dump_json()


def test_explicit_cancellation_is_separate_from_admitted_latency():
    from app.operations.pilot_collector import measure_stream

    bound, request_id = binding(), uuid4()
    stream = [
        ("answer_identity", '{"answer_id":"reserved"}', 10.),
        ("answer_execution", '{"answer_execution":{"state":"stopped","assistant_message_id":"reserved"}}', 20.),
        ("error", '{"code":"ANSWER_EXECUTION_STOPPED"}', 21.),
        ("done", "[DONE]", 22.),
    ]
    result = measure_stream(bound, stream, request_id=request_id, observed_at=bound.window_start,
                            terminal_ms=23., operation=None)
    assert result.population == "canceled"
    assert result.state == "stopped"
    assert result.closed_outcome_ms is None


def test_malformed_outcome_is_a_sanitized_stream_failure():
    from app.operations.pilot_collector import measure_stream

    bound = binding()
    stream = [("outcome", '{"outcome":{"PRIVATE":"CONTENT"}}', 1.), ("done", "[DONE]", 2.)]
    result = measure_stream(bound, stream, request_id=uuid4(), observed_at=bound.window_start,
                            terminal_ms=3., operation=None)
    assert result.state == "failed"
    assert "PRIVATE" not in result.model_dump_json()


def test_missing_admission_identity_cannot_be_a_successful_closed_stream():
    from app.operations.pilot_collector import measure_stream

    bound, request_id = binding(), uuid4()
    stream = []
    for event, data, elapsed in frames():
        if event == "answer_identity":
            continue
        if event == "answer_execution":
            value = json.loads(data)
            value["answer_execution"]["assistant_message_id"] = None
            data = json.dumps(value)
        stream.append((event, data, elapsed))
    result = measure_stream(bound, stream, request_id=request_id, observed_at=bound.window_start,
                            terminal_ms=30001., operation=operation(bound, request_id))
    assert result.state == "failed"
    assert result.closed_outcome_ms is None


@pytest.mark.parametrize("event,raw", [
    ("answer_execution", '{"answer_execution":{"assistant_message_id":"reserved"}}'),
    ("outcome", '{"outcome":null}'),
])
def test_malformed_terminal_frame_cannot_be_repaired_by_a_later_frame(event, raw):
    from app.operations.pilot_collector import measure_stream

    bound, request_id = binding(), uuid4()
    stream = list(frames())
    stream.insert(3, (event, raw, 200.))
    result = measure_stream(bound, stream, request_id=request_id, observed_at=bound.window_start,
                            terminal_ms=30001., operation=operation(bound, request_id),
                            verified_message=verified_message())
    assert result.state == "failed"
    assert result.closed_outcome_ms is None


@pytest.mark.parametrize("code,category", [
    ("RETRIEVAL_FAILED", "retrieval"), ("PROVIDER_TIMEOUT", "provider"),
    ("ANSWER_EXECUTION_PERSISTENCE_FAILED", "persistence"), ("APPLICATION_FAILED", "application"),
])
def test_bound_operational_failure_preserves_known_retrieval_attribution(code, category):
    from app.operations.pilot_collector import measure_stream

    bound, request_id = binding(), uuid4()
    retained = operation(bound, request_id) | {"normalized_error": code}
    retained["dimensions"]["stage_durations_ms"]["provider"] = 0
    retained["generation_route"] = None
    stream = [("answer_identity", '{"answer_id":"reserved"}', 10.),
              ("error", '{"code":"CHAT_STREAM_FAILED"}', 20.), ("done", "[DONE]", 21.)]
    result = measure_stream(bound, stream, request_id=request_id, observed_at=bound.window_start,
                            terminal_ms=30001., operation=retained)
    assert result.state == "failed"
    assert result.normalized_failure == category
