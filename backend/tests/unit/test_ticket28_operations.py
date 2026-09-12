from app.extensions.provider_router import normalized_generation_error
from app.operations.events import OperationalEventService


def test_route_observation_cannot_copy_content_disguised_as_a_provider_identifier():
    record = OperationalEventService.route_observation({
        "route_identity": "provider_route:" + "a" * 64,
        "route_reason": "succeeded",
        "provider_attempts": [{
            "provider": "PRIVATE_QUESTION_WITHOUT_SPACES",
            "approval_identity": "configuration:" + "b" * 64,
            "payload_sha256": "c" * 64,
            "snapshot_sha256": "d" * 64,
            "attempt": 1, "latency_ms": 12,
        }],
    })
    assert "PRIVATE_QUESTION" not in str(record)
    assert record["attempts"][0]["approval_identity"] == "configuration:" + "b" * 64


def test_event_dimensions_use_closed_states_bounded_timings_and_hash_identities():
    assert OperationalEventService.dimensions({
        "execution_state": "queued", "outcome": "PRIVATE_OUTCOME",
        "evidence_count": 2, "configuration_identity": "configuration:" + "a" * 64,
        "policy_identity": "PRIVATE_POLICY", "stage_durations_ms": {
            "queue": 14, "retrieval": "PRIVATE_TIMING", "PRIVATE_STAGE": 10,
        }, "question": "PRIVATE_QUESTION",
    }) == {
        "execution_state": "queued", "evidence_count": 2,
        "configuration_identity": "configuration:" + "a" * 64,
        "stage_durations_ms": {"queue": 14},
    }


def test_generation_application_contract_failure_is_not_a_provider_failure():
    normalized = normalized_generation_error(ValueError("PRIVATE_APPLICATION_BODY"))
    assert normalized == "application_failure"
    assert OperationalEventService.failure_category(normalized) == "application"
