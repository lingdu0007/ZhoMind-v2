from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.contracts.canonical import (
    AnswerExecutionState,
    AnswerOutcome,
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRecord,
    CanonicalRecordClass,
    CoveragePosition,
    EditorialRevisionChangeKind,
    EntryLifecycleState,
    KnowledgeAssuranceLevel,
    KnowledgeSourceTier,
    SourceAccessScope,
    StableIdentity,
    StableIdentityKind,
    compatibility_read_projection,
    serialized_round_trip,
    validate_transition,
)
from app.model.base import Base
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel


def test_all_required_identity_kinds_are_stable_and_round_trip() -> None:
    kinds = set(StableIdentityKind)
    assert kinds == {
        StableIdentityKind.EVENT,
        StableIdentityKind.MEMBER,
        StableIdentityKind.TEAM_INVITATION,
        StableIdentityKind.ADMISSION_ATTEMPT,
        StableIdentityKind.ENTRY,
        StableIdentityKind.SOURCE,
        StableIdentityKind.BUNDLE,
        StableIdentityKind.BUNDLE_ITEM,
        StableIdentityKind.BUILD_GENERATION,
        StableIdentityKind.CANDIDATE,
        StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION,
        StableIdentityKind.ANSWER_EXECUTION,
        StableIdentityKind.EVIDENCE_SET,
        StableIdentityKind.EVIDENCE_SNAPSHOT,
        StableIdentityKind.MAINTENANCE_ITEM,
        StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD,
        StableIdentityKind.COLLECTION,
        StableIdentityKind.CAPABILITY,
        StableIdentityKind.CONFIGURATION,
        StableIdentityKind.CONCURRENCY,
        StableIdentityKind.CORPUS,
        StableIdentityKind.DATA_BOUNDARY,
        StableIdentityKind.DEPLOYMENT,
        StableIdentityKind.EDITORIAL_REVISION,
        StableIdentityKind.EMBEDDING_PROFILE,
        StableIdentityKind.HOST,
        StableIdentityKind.MIGRATION,
        StableIdentityKind.OBJECTIVE,
        StableIdentityKind.PRODUCT_PATH,
        StableIdentityKind.PRODUCT_REVISION,
        StableIdentityKind.PROMPT_ENVELOPE,
        StableIdentityKind.PROVIDER_ROUTE,
        StableIdentityKind.PUBLIC_CLAIM,
        StableIdentityKind.RETRIEVAL_PROFILE,
        StableIdentityKind.USER_BOUNDARY,
    }
    identity = StableIdentity(StableIdentityKind.ENTRY, "entry-001")
    assert StableIdentity.from_stable_id(identity.stable_id) == identity
    deployment = StableIdentity.from_stable_id("deployment:pilot-target-20260905")
    assert deployment.kind is StableIdentityKind.DEPLOYMENT


def test_illegal_transitions_are_rejected() -> None:
    assert validate_transition(EntryLifecycleState, "draft", "evidence_collected") is EntryLifecycleState.EVIDENCE_COLLECTED
    assert validate_transition(AnswerExecutionState, "running", "completed") is AnswerExecutionState.COMPLETED
    with pytest.raises(ValueError, match="illegal"):
        validate_transition(EntryLifecycleState, "draft", "published")
    with pytest.raises(ValueError, match="illegal"):
        validate_transition(AnswerExecutionState, "completed", "running")


def test_editorial_authority_vocabulary_is_closed_and_canonical() -> None:
    assert {position.value for position in CoveragePosition} == {
        "rag_source_admission_and_chunking",
        "sparse_dense_hybrid_and_reranking_choices",
        "evidence_sufficiency_refusal_and_acceptance",
        "tools_and_mcp_permissions_and_failure_behavior",
        "agent_context_state_and_memory",
        "orchestration_retry_human_intervention_and_side_effects",
        "provider_failure_and_observability",
        "prompt_injection_isolation_and_security",
    }
    assert {level.value for level in KnowledgeAssuranceLevel} == {
        "source_grounded",
        "claim_linked",
        "release_assured",
    }
    assert {tier.value for tier in KnowledgeSourceTier} == {
        "primary_evidence_source",
        "reproducible_engineering_evidence",
        "secondary_discovery_source",
        "bounded_internal_case",
    }
    assert {scope.value for scope in SourceAccessScope} == {"public", "controlled_internal"}
    assert {kind.value for kind in EditorialRevisionChangeKind} == {"material", "wording_only"}


def test_record_and_event_serialization_preserve_identity_and_state() -> None:
    identity = StableIdentity(StableIdentityKind.EVIDENCE_SET, "evidence-set-001")
    record = CanonicalRecord(
        identity=identity,
        state="frozen",
        record_class=CanonicalRecordClass.IMMUTABLE,
        payload={"snapshot_ids": ["snapshot-001"]},
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
    )
    restored = serialized_round_trip(record)
    assert isinstance(restored, CanonicalRecord)
    assert restored.identity == identity
    assert restored.state == "frozen"
    assert dict(restored.payload) == dict(record.payload)

    event = CanonicalEvent(
        event_id=StableIdentity(StableIdentityKind.EVENT, "event-001"),
        aggregate=StableIdentity(StableIdentityKind.BUNDLE, "bundle-001"),
        event_type=CanonicalEventType.STATE_CHANGED,
        from_state="received",
        to_state="validated",
        payload={"item_operation": "create"},
    )
    restored_event = serialized_round_trip(event)
    assert isinstance(restored_event, CanonicalEvent)
    assert restored_event.event_id == event.event_id
    assert restored_event.to_state == "validated"


def test_legacy_projection_is_deterministic_and_fail_closed() -> None:
    document = SimpleNamespace(
        id="legacy-document-1",
        status="ready",
        published_generation=2,
        candidate_generation=3,
    )
    first = compatibility_read_projection(StableIdentityKind.ENTRY, document)
    second = compatibility_read_projection(StableIdentityKind.ENTRY, document)
    assert first == second
    assert first.identity.kind is StableIdentityKind.ENTRY
    assert first.state == EntryLifecycleState.PUBLISHED.value
    assert first.payload["answer_eligible"] is False
    assert first.payload["source_availability"] == "unknown"

    answer = compatibility_read_projection(
        StableIdentityKind.ANSWER_EXECUTION,
        {"id": "legacy-answer-1", "rag_trace": {"outcome": AnswerOutcome.INSUFFICIENT_EVIDENCE_REPLY.value}},
    )
    assert answer.state == AnswerExecutionState.COMPLETED.value
    assert answer.payload["answer_eligible"] is False
    assert answer.payload["evidence_set_id"] == "unknown"


def test_canonical_tables_are_immutable_and_events_are_append_only(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'canonical.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        row = CanonicalRecordModel(
            stable_id="entry:entry-001",
            identity_kind="entry",
            identity_value="entry-001",
            state="draft",
            record_class="authoritative",
            schema_version=1,
            payload={"answer_eligible": False},
            created_at=datetime.now(UTC),
        )
        event = CanonicalEventModel(
            aggregate_id="entry:entry-001",
            aggregate_kind="entry",
            event_type="created",
            to_state="draft",
            payload={},
            occurred_at=datetime.now(UTC),
        )
        session.add_all([row, event])
        session.commit()
        with pytest.raises(ValueError, match="immutable"):
            row.state = "published"
            session.flush()
        session.rollback()
        assert session.scalar(select(CanonicalRecordModel.stable_id)) == "entry:entry-001"
        with pytest.raises(ValueError, match="append-only"):
            event.to_state = "published"
            session.flush()
        session.rollback()
        assert session.scalar(select(CanonicalEventModel.to_state)) == "draft"
    engine.dispose()
