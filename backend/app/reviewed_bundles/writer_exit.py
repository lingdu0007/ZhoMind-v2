from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.reviewed_bundles.inputs import FrozenCandidateBuildInput
from app.reviewed_bundles.models import CandidateBuildJob


class CandidateWriterExit(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_name: Literal["candidate_writer_exit/v1"] = Field(alias="schema")
    job_id: str
    attempt: Annotated[int, Field(gt=0)]
    frozen_input_sha256: str
    document_identity: str
    generation: Annotated[int, Field(gt=0)]
    embedding_fingerprint: str | None

    @classmethod
    def for_attempt(cls, job_id: str, attempt: int, frozen: FrozenCandidateBuildInput) -> CandidateWriterExit:
        return cls.model_validate({
            "schema": "candidate_writer_exit/v1", "job_id": job_id, "attempt": attempt,
            "frozen_input_sha256": frozen.frozen_input_sha256, "document_identity": frozen.document_identity,
            "generation": frozen.requested_generation, "embedding_fingerprint": frozen.embedding_fingerprint,
        })

    def event_id(self) -> str:
        return uuid5(NAMESPACE_URL, f"candidate-writer-exit:{self.job_id}:{self.attempt}:{self.frozen_input_sha256}").hex


async def record_candidate_writer_exit(session: AsyncSession, proof: CandidateWriterExit) -> None:
    event = await session.get(CanonicalEventModel, proof.event_id())
    if event is not None:
        _validate_exit_event(event, proof)
        return
    session.add(CanonicalEventModel(
        id=proof.event_id(), aggregate_id=f"build_generation:{proof.job_id}", aggregate_kind="build_generation",
        event_type="backfill_projected", from_state="indexing", to_state="settled",
        payload=proof.model_dump(by_alias=True),
    ))


def _validate_exit_event(event: CanonicalEventModel, expected: CandidateWriterExit) -> None:
    if (
        CandidateWriterExit.model_validate(event.payload) != expected or event.id != expected.event_id()
        or event.aggregate_id != f"build_generation:{expected.job_id}" or event.aggregate_kind != "build_generation"
        or event.event_type != "backfill_projected" or event.from_state != "indexing"
        or event.to_state != "settled" or event.recorded_by is not None
    ):
        raise ValueError("Candidate writer exit binding differs")


async def require_candidate_writers_settled(
    session: AsyncSession, job: CandidateBuildJob, frozen: FrozenCandidateBuildInput,
) -> None:
    events = (await session.scalars(select(CanonicalEventModel).where(
        CanonicalEventModel.aggregate_id == f"build_generation:{job.id}",
        CanonicalEventModel.aggregate_kind == "build_generation",
    ))).all()
    indexing_attempts: set[int] = set()
    for event in events:
        if event.to_state != "indexing" or event.payload.get("action") != "indexing":
            continue
        payload = _with_verified_legacy_hash(event.payload, frozen)
        attempt = payload.get("attempt")
        if (
            payload.get("schema") != "candidate_build_job_event/v1"
            or type(attempt) is not int or not 0 < attempt <= job.attempt
            or payload.get("input_sha256") != frozen.input_sha256
            or payload.get("editorial_source_revision") != frozen.editorial_source_revision
        ):
            raise ValueError("Candidate indexing attempt cannot be verified")
        indexing_attempts.add(attempt)
    if job.stage == "indexing" and job.attempt > 0 and job.attempt not in indexing_attempts:
        raise ValueError("Candidate indexing attempt is missing")
    for attempt in indexing_attempts:
        expected = CandidateWriterExit.for_attempt(job.id, attempt, frozen)
        event = next((event for event in events if event.id == expected.event_id()), None)
        if event is None:
            await _require_completed_attempt(session, events, expected, frozen)
        else:
            _validate_exit_event(event, expected)


async def _require_completed_attempt(
    session: AsyncSession, events: Sequence[CanonicalEventModel],
    expected: CandidateWriterExit, frozen: FrozenCandidateBuildInput,
) -> None:
    # Older successful builders committed these facts only after indexing returned.
    # Mutable terminal states, failed attempts and cancellation are not exit proof.
    completed = [event for event in events if (
        event.payload.get("action") == "completed" and event.payload.get("attempt") == expected.attempt
    )]
    if len(completed) != 1:
        raise ValueError("Candidate writer termination is unproven")
    event = completed[0]
    expected_event = {
        "schema": "candidate_build_job_event/v1", "action": "completed", "stage": "indexing",
        "status": "candidate_ready", "progress": 100, "attempt": expected.attempt,
        "editorial_source_revision": frozen.editorial_source_revision, "input_sha256": frozen.input_sha256,
        "frozen_input_sha256": frozen.frozen_input_sha256, "failure_reason": None,
        "allowed_next_action": "await_candidate_inspection",
    }
    if (
        _with_verified_legacy_hash(event.payload, frozen) != expected_event or type(event.payload.get("attempt")) is not int
        or event.event_type != "state_changed" or event.from_state != "indexing"
        or event.to_state != "candidate_ready" or event.recorded_by is not None
    ):
        raise ValueError("Candidate completion cannot prove writer termination")
    candidate_value = f"{expected.job_id}-attempt-{expected.attempt}"
    candidate = await session.get(CanonicalRecordModel, f"candidate:{candidate_value}")
    candidate_payload = _with_verified_legacy_hash(candidate.payload, frozen) if candidate is not None else {}
    expected_candidate = {
        "schema": "candidate_build_candidate/v1",
        "build_generation_id": f"build_generation:{expected.job_id}", "attempt": expected.attempt,
        "bundle_id": frozen.bundle_id, "bundle_sha256": frozen.bundle_sha256,
        "bundle_item_id": frozen.bundle_item_id, "bundle_item_sha256": frozen.bundle_item_sha256,
        "entry_identity": frozen.entry_identity, "document_identity": frozen.document_identity,
        "requested_generation": frozen.requested_generation,
        "editorial_source_revision": frozen.editorial_source_revision, "input_sha256": frozen.input_sha256,
        "frozen_input_sha256": frozen.frozen_input_sha256, "chunk_strategy": frozen.chunk_strategy,
        "embedding_configuration": frozen.embedding_configuration,
        "embedding_active": frozen.embedding_configuration.get("active") is True,
        "embedding_fingerprint": frozen.embedding_fingerprint,
    }
    if (
        candidate is None or candidate.identity_kind != "candidate" or candidate.identity_value != candidate_value
        or candidate.record_class != "immutable" or candidate.state != "candidate_ready"
        or any(type(candidate_payload.get(key)) is not type(value) or candidate_payload.get(key) != value
               for key, value in expected_candidate.items())
    ):
        raise ValueError("Candidate completion binding differs")


def _with_verified_legacy_hash(payload: dict, frozen: FrozenCandidateBuildInput) -> dict:
    if "frozen_input_sha256" not in payload and frozen.is_legacy_hash_backfill:
        return {**payload, "frozen_input_sha256": frozen.frozen_input_sha256}
    if payload.get("frozen_input_sha256") != frozen.frozen_input_sha256:
        raise ValueError("Candidate frozen input hash differs")
    return payload
