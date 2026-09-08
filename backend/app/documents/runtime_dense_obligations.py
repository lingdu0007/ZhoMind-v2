from __future__ import annotations

from typing import Annotated, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.model.canonical import CanonicalEventModel


class RuntimeDenseTarget(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_name: Literal["runtime_dense_cleanup/v1"] = Field(alias="schema")
    target_identity: str
    publication_identity: str
    document_identity: str
    generation: Annotated[int, Field(gt=0)]
    embedding_fingerprint: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    state: Literal["pending", "settled", "reconciled"]
    withdrawal_event_identity: str | None


def register_runtime_dense_target(
    session: AsyncSession, *, publication_identity: str, document_identity: str,
    generation: int, embedding_fingerprint: str,
) -> RuntimeDenseTarget:
    """The caller commits this address before starting any external write."""
    event_id = uuid4().hex
    target = RuntimeDenseTarget.model_validate({
        "schema": "runtime_dense_cleanup/v1", "target_identity": f"event:{event_id}",
        "publication_identity": publication_identity, "document_identity": document_identity,
        "generation": generation, "embedding_fingerprint": embedding_fingerprint,
        "state": "pending", "withdrawal_event_identity": None,
    })
    session.add(CanonicalEventModel(
        id=event_id, aggregate_id=publication_identity, aggregate_kind="published_knowledge_version",
        event_type="backfill_projected", from_state=None, to_state="pending",
        payload=target.model_dump(by_alias=True),
    ))
    return target


async def settle_runtime_dense_target(
    session: AsyncSession, target: RuntimeDenseTarget, *, write_never_started: bool = False,
) -> None:
    """Record that the admitted writer has exited or was never started."""
    intent = await session.get(CanonicalEventModel, target.target_identity.removeprefix("event:"))
    if intent is None and write_never_started:
        return
    if (
        intent is None or intent.payload != target.model_dump(by_alias=True)
        or intent.aggregate_id != target.publication_identity
        or intent.aggregate_kind != "published_knowledge_version" or intent.event_type != "backfill_projected"
        or intent.from_state is not None or intent.to_state != "pending" or intent.recorded_by is not None
    ):
        raise AppError(
            status_code=409, code="RUNTIME_DENSE_OBLIGATION_INVALID",
            message="runtime dense intent cannot be verified",
        )
    settled = target.model_copy(update={"state": "settled"})
    event_id = uuid5(NAMESPACE_URL, f"{target.target_identity}:runtime-dense-settled").hex
    existing = await session.get(CanonicalEventModel, event_id)
    if existing is not None:
        if (
            existing.payload != settled.model_dump(by_alias=True)
            or existing.aggregate_id != target.publication_identity
            or existing.aggregate_kind != "published_knowledge_version"
            or existing.event_type != "backfill_projected"
            or existing.from_state != "pending" or existing.to_state != "settled"
            or existing.recorded_by is not None
        ):
            raise AppError(
                status_code=409, code="RUNTIME_DENSE_OBLIGATION_INVALID",
                message="runtime dense writer proof cannot be verified",
            )
        return
    session.add(CanonicalEventModel(
        id=event_id, aggregate_id=target.publication_identity, aggregate_kind="published_knowledge_version",
        event_type="backfill_projected", from_state="pending", to_state="settled",
        payload=settled.model_dump(by_alias=True),
    ))


async def pending_runtime_dense_targets(session: AsyncSession, withdrawal: dict) -> list[RuntimeDenseTarget]:
    events = (await session.scalars(select(CanonicalEventModel).where(
        CanonicalEventModel.aggregate_id == withdrawal["publication_identity"],
        CanonicalEventModel.event_type == "backfill_projected",
    ).order_by(CanonicalEventModel.occurred_at, CanonicalEventModel.id))).all()
    pending: dict[str, RuntimeDenseTarget] = {}
    for event in events:
        try:
            target = RuntimeDenseTarget.model_validate(event.payload)
            if (
                event.aggregate_kind != "published_knowledge_version"
                or target.publication_identity != withdrawal["publication_identity"]
                or target.document_identity != withdrawal["document_identity"]
                or target.generation != withdrawal["generation"] or event.to_state != target.state
            ):
                raise ValueError("runtime dense target binding differs")
            if target.state == "pending":
                if (
                    target.target_identity != f"event:{event.id}" or event.from_state is not None
                    or target.withdrawal_event_identity is not None or event.recorded_by is not None
                ):
                    raise ValueError("runtime dense intent binding differs")
                pending[target.target_identity] = target
            elif target.state == "settled":
                original = pending.get(target.target_identity)
                if (
                    original is None or original.state != "pending" or event.from_state != "pending"
                    or event.recorded_by is not None or target.withdrawal_event_identity is not None
                    or event.id != uuid5(NAMESPACE_URL, f"{target.target_identity}:runtime-dense-settled").hex
                    or original.model_copy(update={"state": "settled"}) != target
                ):
                    raise ValueError("runtime dense writer proof differs")
                pending[target.target_identity] = target
            else:
                original = pending.pop(target.target_identity, None)
                if (
                    original is None or original.state != "settled" or event.from_state != "settled"
                    or event.recorded_by != withdrawal["actor_identity"]
                    or target.withdrawal_event_identity != withdrawal["event_identity"]
                    or original.model_copy(update={
                        "state": "reconciled", "withdrawal_event_identity": withdrawal["event_identity"],
                    }) != target
                ):
                    raise ValueError("runtime dense cleanup proof differs")
        except (ValidationError, ValueError, TypeError) as exc:
            raise AppError(
                status_code=409, code="RUNTIME_DENSE_OBLIGATION_INVALID",
                message="runtime dense cleanup obligation cannot be verified",
            ) from exc
    return list(pending.values())


def complete_runtime_dense_target(session: AsyncSession, target: RuntimeDenseTarget, withdrawal: dict) -> None:
    completed = target.model_copy(update={
        "state": "reconciled", "withdrawal_event_identity": withdrawal["event_identity"],
    })
    session.add(CanonicalEventModel(
        aggregate_id=target.publication_identity, aggregate_kind="published_knowledge_version",
        event_type="backfill_projected", from_state="settled", to_state="reconciled",
        recorded_by=withdrawal["actor_identity"], payload=completed.model_dump(by_alias=True),
    ))
