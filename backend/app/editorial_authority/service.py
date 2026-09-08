from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, NoReturn

from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.contracts.candidate_claim_evidence import (
    CandidateClaimEvidenceContractError,
    build_candidate_claim_evidence_contract,
)
from app.contracts.canonical import (
    AcceptanceStatus,
    CanonicalEventType,
    CanonicalRecordClass,
    EditorialRevisionChangeKind,
    EntryLifecycleState,
    SourceAvailabilityState,
    StableIdentity,
    StableIdentityKind,
    validate_transition,
)
from app.delivery_acceptance.schemas import (
    AcceptanceCheckResult,
    AcceptanceStatusReason,
    CreateDeliveryAcceptanceRecordRequest,
    UpdateDeliveryAcceptanceStatusRequest,
)
from app.editorial_authority.schemas import (
    CreateEditorialEntryRequest,
    ReviseEditorialEntryRequest,
    editorial_export_safety_findings,
    editorial_secret_scan_findings,
    lightweight_revision_reasons,
    review_validation_reasons,
)
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.user import User
from app.rag.claim_evidence import ClaimEvidenceContractError, parse_claim_evidence_contract
from app.service.identity_audit_service import IdentityAuditService


def evaluate_answer_eligibility(
    *,
    lifecycle_state: str,
    approval_status: str,
    source_availability: list[str],
    needs_review_at: datetime | None,
    applicability_explicit: bool,
    known_contradiction: bool,
    integrity_defect: bool,
    decisive_source_loss: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Project eligibility without changing editorial authority or publication state."""

    reasons: list[str] = []
    if approval_status not in {"approved", "lightweight_accepted"}:
        reasons.append("editorial_approval_missing")
    if known_contradiction:
        reasons.append("known_contradiction")
    if integrity_defect:
        reasons.append("integrity_defect")
    if decisive_source_loss:
        reasons.append("decisive_source_loss")
    if not source_availability:
        reasons.append("source_availability_missing")
    elif any(state != "verified_usable" for state in source_availability):
        reasons.append("source_unavailable")
    if not applicability_explicit:
        reasons.append("applicability_not_explicit")

    if lifecycle_state == EntryLifecycleState.PUBLISHED.value:
        return {"answer_eligible": not reasons, "reasons": reasons}
    if lifecycle_state != EntryLifecycleState.NEEDS_REVIEW.value:
        return {"answer_eligible": False, "reasons": reasons or ["not_published"]}

    if needs_review_at is None:
        return {"answer_eligible": False, "reasons": reasons or ["needs_review_timestamp_missing"]}
    current_time = now or datetime.now(UTC)
    if needs_review_at.tzinfo is None:
        needs_review_at = needs_review_at.replace(tzinfo=UTC)
    if current_time - needs_review_at > timedelta(days=7):
        reasons.append("needs_review_grace_expired")
    if reasons:
        return {"answer_eligible": False, "reasons": reasons}
    return {"answer_eligible": True, "reasons": ["needs_review_grace_active"]}


class EditorialAuthorityService:
    """Own private editorial authority without creating runtime publication copies."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_draft(self, payload: CreateEditorialEntryRequest, author: User) -> dict[str, Any]:
        self._require_editorial_member(author)
        self._raise_for_secret_findings(payload.model_dump(mode="json"))
        identity = StableIdentity(StableIdentityKind.ENTRY, payload.entry_id)
        if await self.session.get(CanonicalRecordModel, identity.stable_id) is not None:
            raise AppError(
                status_code=409,
                code="EDITORIAL_ENTRY_ID_DUPLICATE",
                message="an editorial entry already uses this immutable entry_id",
                detail={"entry_id": payload.entry_id},
            )

        author_identity = await IdentityAuditService(self.session).ensure_member_record(
            author,
            admission_path="private_editorial_repository",
        )
        revision_identity = StableIdentity(StableIdentityKind.EDITORIAL_REVISION, f"{identity.value}.r1")
        self.session.add(
            CanonicalRecordModel(
                stable_id=identity.stable_id,
                identity_kind=identity.kind.value,
                identity_value=identity.value,
                state=EntryLifecycleState.DRAFT.value,
                record_class=CanonicalRecordClass.AUTHORITATIVE.value,
                schema_version=payload.schema_version,
                payload={
                    "schema": "editorial_entry/v1",
                    "author_identity": author_identity,
                    "initial_revision_identity": revision_identity.stable_id,
                },
            )
        )
        self.session.add(
            CanonicalRecordModel(
                stable_id=revision_identity.stable_id,
                identity_kind=revision_identity.kind.value,
                identity_value=revision_identity.value,
                state=EntryLifecycleState.DRAFT.value,
                record_class=CanonicalRecordClass.AUTHORITATIVE.value,
                schema_version=payload.schema_version,
                payload={
                    "schema": "editorial_revision/v1",
                    "entry_identity": identity.stable_id,
                    "revision_number": 1,
                    "author_identity": author_identity,
                    "material_reviser_identity": author_identity,
                    "change_kind": "initial",
                    "draft": payload.model_dump(mode="json"),
                },
            )
        )
        self.session.add(
            CanonicalEventModel(
                aggregate_id=identity.stable_id,
                aggregate_kind=identity.kind.value,
                event_type=CanonicalEventType.CREATED.value,
                from_state=None,
                to_state=EntryLifecycleState.DRAFT.value,
                payload={
                    "schema": "editorial_authority_event/v1",
                    "sequence": 1,
                    "action": "draft_created",
                    "revision_identity": revision_identity.stable_id,
                    "author_identity": author_identity,
                },
                recorded_by=author_identity,
            )
        )
        await self.session.commit()
        return await self.get_projection(identity.value)

    async def collect_evidence(self, entry_id: str, actor: User) -> dict[str, Any]:
        self._require_editorial_member(actor)
        entry, events = await self._entry_and_events(entry_id)
        entry_payload = self._payload(entry)
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            actor,
            admission_path="private_editorial_repository",
        )
        self._require_author(entry_payload, actor_identity, "collect its evidence")
        current = self._latest_event(events)
        self._require_state(current, EntryLifecycleState.DRAFT)
        draft, revision_identity = await self._current_draft(entry_payload, current)
        self._raise_for_secret_findings(draft.model_dump(mode="json"))
        self._raise_for_incomplete_draft(draft, action="evidence collection")
        await self._release_assurance_snapshot(draft)
        roles = await self._resolve_roles(draft, author_identity=actor_identity)
        await self._ensure_source_records(draft, actor_identity)
        await self._append_entry_event(
            entry,
            events,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state=EntryLifecycleState.DRAFT.value,
            to_state=validate_transition(
                EntryLifecycleState,
                EntryLifecycleState.DRAFT,
                EntryLifecycleState.EVIDENCE_COLLECTED,
            ).value,
            action="evidence_collected",
            revision_identity=revision_identity,
            actor_identity=actor_identity,
            roles=roles,
        )
        await self.session.commit()
        return await self.get_projection(entry_id)

    async def request_editorial_review(self, entry_id: str, actor: User) -> dict[str, Any]:
        self._require_editorial_member(actor)
        entry, events = await self._entry_and_events(entry_id)
        entry_payload = self._payload(entry)
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            actor,
            admission_path="private_editorial_repository",
        )
        self._require_author(entry_payload, actor_identity, "request editorial review")
        current = self._latest_event(events)
        self._require_state(current, EntryLifecycleState.EVIDENCE_COLLECTED)
        draft, revision_identity = await self._current_draft(entry_payload, current)
        self._raise_for_secret_findings(draft.model_dump(mode="json"))
        self._raise_for_incomplete_draft(draft, action="editorial review")
        await self._release_assurance_snapshot(draft)
        roles = self._roles_for_revision(events, revision_identity)
        if not roles:
            raise AppError(
                status_code=409,
                code="EDITORIAL_ROLE_ASSIGNMENT_MISSING",
                message="evidence collection must retain editorial role assignments before review",
            )
        if roles["approving_reviewer_identity"] == actor_identity:
            raise AppError(
                status_code=409,
                code="EDITORIAL_ROLE_SEPARATION_REQUIRED",
                message="the author cannot be the approving reviewer for the same revision",
            )
        self._require_maintainer_acceptance(events, revision_identity)
        await self._raise_for_usable_sources(draft, action="editorial review")
        await self._append_entry_event(
            entry,
            events,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state=EntryLifecycleState.EVIDENCE_COLLECTED.value,
            to_state=validate_transition(
                EntryLifecycleState,
                EntryLifecycleState.EVIDENCE_COLLECTED,
                EntryLifecycleState.EDITORIAL_REVIEW,
            ).value,
            action="editorial_review_requested",
            revision_identity=revision_identity,
            actor_identity=actor_identity,
            roles=roles,
        )
        await self.session.commit()
        return await self.get_projection(entry_id)

    async def approve_current_revision(self, entry_id: str, actor: User) -> dict[str, Any]:
        self._require_editorial_member(actor)
        entry, events = await self._entry_and_events(entry_id)
        entry_payload = self._payload(entry)
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            actor,
            admission_path="private_editorial_repository",
        )
        current = self._latest_event(events)
        self._require_state(current, EntryLifecycleState.EDITORIAL_REVIEW)
        draft, revision_identity = await self._current_draft(entry_payload, current)
        self._raise_for_incomplete_draft(draft, action="approval")
        release_assurance_snapshot = await self._release_assurance_snapshot(draft)
        roles = self._roles_for_revision(events, revision_identity)
        reviewer_identity = roles.get("approving_reviewer_identity")
        prohibited = {
            value
            for value in (
                roles.get("author_identity"),
                self._payload(await self._revision(revision_identity)).get("material_reviser_identity"),
            )
            if isinstance(value, str)
        }
        if actor_identity in prohibited:
            raise AppError(
                status_code=409,
                code="EDITORIAL_SELF_APPROVAL_FORBIDDEN",
                message="an author or material reviser cannot approve the same revision",
            )
        if actor_identity != reviewer_identity:
            raise AppError(
                status_code=403,
                code="EDITORIAL_REVIEWER_REQUIRED",
                message="only the assigned approving reviewer may approve this revision",
            )
        self._require_maintainer_acceptance(events, revision_identity)
        source_snapshot = await self._verified_source_snapshot(draft, action="approval")
        authority_snapshot = self._retained_approval_snapshot(
            entry_identity=entry.stable_id,
            revision_identity=revision_identity,
            events=events,
            roles=roles,
            reviewer_identity=actor_identity,
            action="revision_approved",
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=EntryLifecycleState.EDITORIAL_REVIEW.value,
            to_state=EntryLifecycleState.EDITORIAL_REVIEW.value,
            sources=source_snapshot,
            release_assurance=release_assurance_snapshot,
        )
        await self._append_entry_event(
            entry,
            events,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=EntryLifecycleState.EDITORIAL_REVIEW.value,
            to_state=EntryLifecycleState.EDITORIAL_REVIEW.value,
            action="revision_approved",
            revision_identity=revision_identity,
            actor_identity=actor_identity,
            roles=roles,
            extra={
                "authority_snapshot": authority_snapshot
            },
        )
        await self.session.commit()
        return await self.get_projection(entry_id)

    async def revise_entry(
        self,
        entry_id: str,
        payload: ReviseEditorialEntryRequest,
        actor: User,
    ) -> dict[str, Any]:
        self._require_editorial_member(actor)
        entry, events = await self._entry_and_events(entry_id)
        entry_payload = self._payload(entry)
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            actor,
            admission_path="private_editorial_repository",
        )
        current = self._latest_event(events)
        current_draft, current_revision_identity = await self._current_draft(entry_payload, current)
        current_roles = self._roles_for_revision(events, current_revision_identity)
        permitted_identities = {
            entry_payload.get("author_identity"),
            current_roles.get("approving_reviewer_identity"),
            current_roles.get("accountable_maintainer_identity"),
        }
        if actor_identity not in permitted_identities:
            raise AppError(
                status_code=403,
                code="EDITORIAL_AUTHORITY_FORBIDDEN",
                message="only the entry author, reviewer, or accountable maintainer may revise an entry",
            )
        if payload.entry.entry_id != entry.identity_value:
            raise AppError(
                status_code=409,
                code="EDITORIAL_REVISION_ENTRY_MISMATCH",
                message="a revision must retain the immutable entry_id",
            )
        self._raise_for_secret_findings(
            {
                "entry": payload.entry.model_dump(mode="json"),
                "lightweight_reason": payload.lightweight_reason,
            }
        )
        if payload.change_kind not in {kind.value for kind in EditorialRevisionChangeKind}:
            raise AppError(
                status_code=422,
                code="EDITORIAL_REVISION_KIND_INVALID",
                message="change_kind must be material or wording_only",
            )
        self._raise_for_incomplete_draft(
            payload.entry,
            action=(
                "material revision"
                if payload.change_kind == EditorialRevisionChangeKind.MATERIAL.value
                else "wording-only revision"
            ),
        )
        await self._release_assurance_snapshot(payload.entry)
        if payload.change_kind == EditorialRevisionChangeKind.WORDING_ONLY.value:
            if not isinstance(payload.lightweight_reason, str) or not payload.lightweight_reason.strip():
                raise AppError(
                    status_code=422,
                    code="EDITORIAL_LIGHTWEIGHT_REASON_REQUIRED",
                    message="wording-only revisions require an explicit lightweight reason",
                )
            if self._approval_for_revision(events, current_revision_identity) is None:
                raise AppError(
                    status_code=409,
                    code="EDITORIAL_LIGHTWEIGHT_BASE_NOT_APPROVED",
                    message="a wording-only revision requires an approved base revision",
                )
            reasons = lightweight_revision_reasons(current_draft, payload.entry)
            if reasons:
                raise AppError(
                    status_code=422,
                    code="EDITORIAL_LIGHTWEIGHT_CHANGE_INVALID",
                    message="wording-only revisions cannot alter material editorial authority data",
                    detail={"reasons": reasons},
                )
        roles = await self._resolve_roles(payload.entry, author_identity=str(entry_payload["author_identity"]))
        if (
            payload.change_kind == EditorialRevisionChangeKind.MATERIAL.value
            and roles["approving_reviewer_identity"] == actor_identity
        ):
            raise AppError(
                status_code=409,
                code="EDITORIAL_ROLE_SEPARATION_REQUIRED",
                message="a material reviser cannot be assigned as the approving reviewer",
            )
        current_revision = await self._revision(current_revision_identity)
        revision_number = int(self._payload(current_revision).get("revision_number", 0)) + 1
        revision_identity = StableIdentity(StableIdentityKind.EDITORIAL_REVISION, f"{entry.identity_value}.r{revision_number}")
        self.session.add(
            CanonicalRecordModel(
                stable_id=revision_identity.stable_id,
                identity_kind=revision_identity.kind.value,
                identity_value=revision_identity.value,
                state=current.to_state,
                record_class=CanonicalRecordClass.AUTHORITATIVE.value,
                payload={
                    "schema": "editorial_revision/v1",
                    "entry_identity": entry.stable_id,
                    "revision_number": revision_number,
                    "author_identity": entry_payload["author_identity"],
                    "material_reviser_identity": (
                        actor_identity if payload.change_kind == EditorialRevisionChangeKind.MATERIAL.value else None
                    ),
                    "wording_reviser_identity": (
                        actor_identity if payload.change_kind == EditorialRevisionChangeKind.WORDING_ONLY.value else None
                    ),
                    "change_kind": payload.change_kind,
                    "previous_revision_identity": current_revision_identity,
                    "lightweight_reason": payload.lightweight_reason,
                    "draft": payload.entry.model_dump(mode="json"),
                },
            )
        )
        await self._ensure_source_records(payload.entry, actor_identity)
        to_state = current.to_state
        if payload.change_kind == EditorialRevisionChangeKind.MATERIAL.value:
            if current.to_state in {
                EntryLifecycleState.NEEDS_REVIEW.value,
                EntryLifecycleState.PUBLISHED.value,
            }:
                to_state = validate_transition(
                    EntryLifecycleState,
                    current.to_state,
                    EntryLifecycleState.EDITORIAL_REVIEW,
                ).value
            elif current.to_state not in {
                EntryLifecycleState.DRAFT.value,
                EntryLifecycleState.EVIDENCE_COLLECTED.value,
                EntryLifecycleState.EDITORIAL_REVIEW.value,
            }:
                raise AppError(
                    status_code=409,
                    code="EDITORIAL_LIFECYCLE_STATE_INVALID",
                    message="T01 cannot revise a Candidate Build entry state",
                    detail={"current_state": current.to_state},
                )
        await self._append_entry_event(
            entry,
            events,
            event_type=CanonicalEventType.REPLACED,
            from_state=current.to_state,
            to_state=to_state,
            action=(
                "material_revised"
                if payload.change_kind == EditorialRevisionChangeKind.MATERIAL.value
                else "wording_revised"
            ),
            revision_identity=revision_identity.stable_id,
            actor_identity=actor_identity,
            roles=roles,
        )
        await self.session.commit()
        return await self.get_projection(entry_id)

    async def accept_wording_revision(self, entry_id: str, actor: User) -> dict[str, Any]:
        self._require_editorial_member(actor)
        entry, events = await self._entry_and_events(entry_id)
        entry_payload = self._payload(entry)
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            actor,
            admission_path="private_editorial_repository",
        )
        current = self._latest_event(events)
        draft, revision_identity = await self._current_draft(entry_payload, current)
        revision_payload = self._payload(await self._revision(revision_identity))
        if revision_payload.get("change_kind") != EditorialRevisionChangeKind.WORDING_ONLY.value:
            raise AppError(
                status_code=409,
                code="EDITORIAL_LIGHTWEIGHT_REVISION_REQUIRED",
                message="only a wording-only revision may use lightweight acceptance",
            )
        roles = self._roles_for_revision(events, revision_identity)
        if actor_identity in {
            roles.get("author_identity"),
            revision_payload.get("wording_reviser_identity"),
            revision_payload.get("material_reviser_identity"),
        }:
            raise AppError(
                status_code=409,
                code="EDITORIAL_SELF_APPROVAL_FORBIDDEN",
                message="a revision author cannot accept the same revision",
            )
        if actor_identity != roles.get("approving_reviewer_identity"):
            raise AppError(
                status_code=403,
                code="EDITORIAL_REVIEWER_REQUIRED",
                message="only the assigned approving reviewer may accept a wording revision",
            )
        self._require_maintainer_acceptance(events, revision_identity)
        source_snapshot = await self._verified_source_snapshot(draft, action="lightweight acceptance")
        release_assurance_snapshot = await self._release_assurance_snapshot(draft)
        previous_revision_identity = revision_payload.get("previous_revision_identity")
        if not isinstance(previous_revision_identity, str) or self._approval_for_revision(events, previous_revision_identity) is None:
            raise AppError(
                status_code=409,
                code="EDITORIAL_LIGHTWEIGHT_BASE_NOT_APPROVED",
                message="a wording-only revision requires an approved base revision",
            )
        authority_snapshot = self._retained_approval_snapshot(
            entry_identity=entry.stable_id,
            revision_identity=revision_identity,
            events=events,
            roles=roles,
            reviewer_identity=actor_identity,
            action="wording_revision_accepted",
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=current.to_state,
            to_state=current.to_state,
            sources=source_snapshot,
            release_assurance=release_assurance_snapshot,
        )
        await self._append_entry_event(
            entry,
            events,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=current.to_state,
            to_state=current.to_state,
            action="wording_revision_accepted",
            revision_identity=revision_identity,
            actor_identity=actor_identity,
            roles=roles,
            extra={
                "authority_snapshot": authority_snapshot
            },
        )
        await self.session.commit()
        return await self.get_projection(entry_id)

    async def accept_maintainer_responsibility(self, entry_id: str, actor: User) -> dict[str, Any]:
        """Retain the accountable maintainer's acceptance as authority evidence."""

        self._require_editorial_member(actor)
        entry, events = await self._entry_and_events(entry_id)
        entry_payload = self._payload(entry)
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            actor,
            admission_path="private_editorial_repository",
        )
        current = self._latest_event(events)
        _draft, revision_identity = await self._current_draft(entry_payload, current)
        revision_payload = self._payload(await self._revision(revision_identity))
        is_published_wording_revision = (
            current.to_state == EntryLifecycleState.PUBLISHED.value
            and revision_payload.get("change_kind") == EditorialRevisionChangeKind.WORDING_ONLY.value
        )
        if current.to_state not in {
            EntryLifecycleState.EVIDENCE_COLLECTED.value,
            EntryLifecycleState.EDITORIAL_REVIEW.value,
            EntryLifecycleState.NEEDS_REVIEW.value,
        } and not is_published_wording_revision:
            raise AppError(
                status_code=409,
                code="EDITORIAL_LIFECYCLE_STATE_INVALID",
                message="maintainer responsibility may be accepted only after evidence collection or for a published wording revision",
                detail={"current_state": current.to_state},
            )
        roles = self._roles_for_revision(events, revision_identity)
        if actor_identity != roles.get("accountable_maintainer_identity"):
            raise AppError(
                status_code=403,
                code="EDITORIAL_MAINTAINER_REQUIRED",
                message="only the accountable maintainer may accept responsibility",
            )
        if self._maintainer_acceptance_for_revision(events, revision_identity) is not None:
            raise AppError(
                status_code=409,
                code="EDITORIAL_MAINTAINER_ALREADY_ACCEPTED",
                message="the accountable maintainer has already accepted this revision",
            )
        await self._append_entry_event(
            entry,
            events,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=current.to_state,
            to_state=current.to_state,
            action="maintainer_responsibility_accepted",
            revision_identity=revision_identity,
            actor_identity=actor_identity,
            roles=roles,
        )
        await self.session.commit()
        return await self.get_projection(entry_id)

    async def record_source_availability(
        self,
        entry_id: str,
        source_id: str,
        availability: str,
        actor: User,
    ) -> dict[str, Any]:
        """Append a maintainer-owned source fact without rewriting the source record."""

        self._require_editorial_member(actor)
        try:
            target_availability = SourceAvailabilityState(availability)
        except ValueError as exc:
            raise AppError(
                status_code=422,
                code="EDITORIAL_SOURCE_AVAILABILITY_INVALID",
                message="availability must use the canonical source availability vocabulary",
                detail={"availability": availability},
            ) from exc

        entry, events = await self._entry_and_events(entry_id, for_update=True)
        entry_payload = self._payload(entry)
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            actor,
            admission_path="private_editorial_repository",
        )
        current = self._latest_event(events)
        draft, revision_identity = await self._current_draft(entry_payload, current)
        roles = self._roles_for_revision(events, revision_identity)
        if actor_identity != roles.get("accountable_maintainer_identity"):
            raise AppError(
                status_code=403,
                code="EDITORIAL_MAINTAINER_REQUIRED",
                message="only the accountable maintainer may record source availability",
            )
        maintainer_acceptance = self._require_maintainer_acceptance(events, revision_identity)
        source = next(
            (
                item
                for item in draft.sources or []
                if isinstance(item, dict) and item.get("source_id") == source_id
            ),
            None,
        )
        if source is None:
            raise AppError(
                status_code=404,
                code="EDITORIAL_SOURCE_NOT_FOUND",
                message="source is not part of the current editorial revision",
                detail={"source_id": source_id},
            )
        source_identity = StableIdentity(StableIdentityKind.SOURCE, source_id)
        locked_records = await self._lock_canonical_records({source_identity.stable_id})
        source_record = locked_records.get(source_identity.stable_id)
        if source_record is None:
            raise AppError(
                status_code=409,
                code="EDITORIAL_SOURCE_STORE_UNAUDITABLE",
                message="source availability requires a retained authority record",
                detail={"source_id": source_id},
            )
        previous_availability = await self._source_availability(source_record)
        self.session.add(
            CanonicalEventModel(
                aggregate_id=source_record.stable_id,
                aggregate_kind=StableIdentityKind.SOURCE.value,
                event_type=CanonicalEventType.STATUS_CHANGED.value,
                from_state=previous_availability,
                to_state=target_availability.value,
                payload={
                    "schema": "editorial_source_event/v1",
                    "action": "source_availability_recorded",
                    "entry_identity": entry.stable_id,
                    "editorial_revision_identity": revision_identity,
                    "maintainer_acceptance_event_id": maintainer_acceptance["event_id"],
                },
                recorded_by=actor_identity,
            )
        )
        if target_availability is SourceAvailabilityState.UNAVAILABLE_FOR_NEW_EVIDENCE:
            next_state = current.to_state
            event_type = CanonicalEventType.STATUS_CHANGED
            if current.to_state == EntryLifecycleState.PUBLISHED.value:
                next_state = validate_transition(
                    EntryLifecycleState,
                    EntryLifecycleState.PUBLISHED,
                    EntryLifecycleState.NEEDS_REVIEW,
                ).value
                event_type = CanonicalEventType.STATE_CHANGED
            await self._append_entry_event(
                entry,
                events,
                event_type=event_type,
                from_state=current.to_state,
                to_state=next_state,
                action="source_decisive_loss_observed",
                revision_identity=revision_identity,
                actor_identity=actor_identity,
                roles=roles,
                extra={
                    "source_identity": source_record.stable_id,
                    "decisive_source_loss": True,
                    "needs_review_at": datetime.now(UTC).isoformat(),
                },
            )
        await self.session.commit()
        return await self.get_projection(entry_id)

    async def export_approved_revision(self, entry_id: str, administrator: User) -> dict[str, Any]:
        if not administrator.is_active or administrator.role != "admin":
            raise AppError(
                status_code=403,
                code="EDITORIAL_EXPORT_FORBIDDEN",
                message="only a System Administrator may receive an editorial export",
            )
        entry, events = await self._entry_and_events(entry_id)
        current = self._latest_event(events)
        draft, revision_identity = await self._current_draft(self._payload(entry), current)
        if self._approval_for_revision(events, revision_identity) is None:
            raise AppError(
                status_code=409,
                code="EDITORIAL_EXPORT_APPROVAL_REQUIRED",
                message="only an approved editorial revision may be exported",
            )
        self._require_maintainer_acceptance(events, revision_identity)
        await self._verified_source_snapshot(draft, action="export")
        await self._release_assurance_snapshot(draft)
        export = await self.reconstruct_export(entry_id, revision_identity)
        administrator_identity = await IdentityAuditService(self.session).ensure_member_record(
            administrator,
            admission_path="editorial_export",
        )
        revision = await self._revision(revision_identity)
        self.session.add(
            CanonicalEventModel(
                aggregate_id=revision.stable_id,
                aggregate_kind=StableIdentityKind.EDITORIAL_REVISION.value,
                event_type=CanonicalEventType.STATUS_CHANGED.value,
                from_state="approved",
                to_state="approved",
                payload={
                    "schema": "editorial_export_audit/v1",
                    "action": "exported",
                    "entry_identity": entry.stable_id,
                    "editorial_revision_identity": revision.stable_id,
                    "artifact_sha256": export["artifact_sha256"],
                },
                recorded_by=administrator_identity,
            )
        )
        await self.session.commit()
        return export

    async def get_private_projection(self, entry_id: str, actor: User) -> dict[str, Any]:
        self._require_editorial_member(actor)
        entry, events = await self._entry_and_events(entry_id)
        entry_payload = self._payload(entry)
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            actor,
            admission_path="private_editorial_repository",
        )
        current = self._latest_event(events)
        draft, revision_identity = await self._current_draft(entry_payload, current)
        roles = self._roles_for_revision(events, revision_identity)
        permitted_identities = {
            entry_payload.get("author_identity"),
            roles.get("approving_reviewer_identity"),
            roles.get("accountable_maintainer_identity"),
        }
        if actor_identity not in permitted_identities:
            raise AppError(
                status_code=403,
                code="EDITORIAL_AUTHORITY_FORBIDDEN",
                message="the private editorial repository is not accessible to this member",
            )
        projection = await self.get_projection(entry_id)
        projection["entry"] = draft.model_dump(mode="json")
        return projection

    async def reconstruct_export(self, entry_id: str, revision_identity: str) -> dict[str, Any]:
        entry, events = await self._entry_and_events(entry_id)
        revision = await self._revision(revision_identity)
        revision_payload = self._payload(revision)
        if revision_payload.get("entry_identity") != entry.stable_id:
            raise AppError(
                status_code=409,
                code="EDITORIAL_EXPORT_REVISION_MISMATCH",
                message="editorial revision does not belong to the requested entry",
            )
        draft_data = revision_payload.get("draft")
        if not isinstance(draft_data, dict):
            raise RuntimeError("editorial revision draft is invalid")
        draft = CreateEditorialEntryRequest.model_validate(draft_data)
        legacy_claim_linked_contract = (
            draft.assurance_level == "claim_linked"
            and "claim_evidence_contract" not in draft_data
        )
        self._raise_for_incomplete_draft(
            draft,
            action="export",
            allow_legacy_claim_linked_contract=legacy_claim_linked_contract,
        )
        approval = self._approval_for_revision(events, revision.stable_id)
        if approval is None:
            raise AppError(
                status_code=409,
                code="EDITORIAL_EXPORT_APPROVAL_REQUIRED",
                message="only an approved editorial revision may be reconstructed for export",
            )
        authority_snapshot = self._approval_authority_snapshot(
            approval,
            draft=draft,
            entry_identity=entry.stable_id,
            revision_identity=revision.stable_id,
            events=events,
        )
        claim_evidence_snapshot = (
            {}
            if legacy_claim_linked_contract
            else self._claim_evidence_contract_snapshot(
                draft,
                entry_identity=entry.stable_id,
                revision_identity=revision.stable_id,
            )
        )
        entry_payload = draft.model_dump(mode="json")
        # New optional fields must not rewrite the byte-exact shape of older exports.
        if entry_payload.get("claim_evidence_contract") is None:
            entry_payload.pop("claim_evidence_contract", None)
        artifact = {
            "schema": "editorial_export/v1",
            "entry_identity": entry.stable_id,
            "entry_id": entry.identity_value,
            "editorial_revision_identity": revision.stable_id,
            "revision_number": revision_payload.get("revision_number"),
            "revision_sha256": self._sha256(revision_payload),
            "roles": authority_snapshot["roles"],
            "approval": authority_snapshot["approval"],
            "entry": entry_payload,
            "sources": authority_snapshot["sources"],
            "release_assurance_snapshot": authority_snapshot["release_assurance"],
            "editorial_audit": authority_snapshot["editorial_audit"],
            **claim_evidence_snapshot,
        }
        export_findings = editorial_export_safety_findings(artifact)
        if export_findings:
            raise AppError(
                status_code=422,
                code="EDITORIAL_EXPORT_UNSAFE",
                message="editorial export contains credentials or an automatic-publication instruction",
                detail={"findings": export_findings},
            )
        return {
            "entry_identity": entry.stable_id,
            "editorial_revision_identity": revision.stable_id,
            "artifact_sha256": self._sha256(artifact),
            "artifact": artifact,
        }

    async def verify_approved_export(self, artifact: object, artifact_sha256: str) -> dict[str, Any]:
        """Read and verify an approved export without creating an export audit event."""

        if not isinstance(artifact, dict):
            raise AppError(
                status_code=422,
                code="EDITORIAL_EXPORT_NOT_APPROVED",
                message="reviewed bundle item must contain an editorial export object",
            )
        findings = editorial_export_safety_findings(artifact)
        if findings:
            raise AppError(
                status_code=422,
                code="EDITORIAL_EXPORT_UNSAFE",
                message="editorial export contains credentials or an automatic-publication instruction",
                detail={"findings": findings},
            )

        entry_id = artifact.get("entry_id")
        entry_identity = artifact.get("entry_identity")
        revision_identity = artifact.get("editorial_revision_identity")
        if (
            not isinstance(entry_id, str)
            or not isinstance(entry_identity, str)
            or not isinstance(revision_identity, str)
            or entry_identity != StableIdentity(StableIdentityKind.ENTRY, entry_id).stable_id
        ):
            raise AppError(
                status_code=422,
                code="EDITORIAL_EXPORT_NOT_APPROVED",
                message="editorial export identity is incomplete or inconsistent",
            )

        reconstructed = await self.reconstruct_export(entry_id, revision_identity)
        if reconstructed["artifact_sha256"] != artifact_sha256 or reconstructed["artifact"] != artifact:
            raise AppError(
                status_code=409,
                code="EDITORIAL_EXPORT_NOT_APPROVED",
                message="bundle item does not match the retained approved editorial export",
                detail={"entry_id": entry_id, "editorial_revision_identity": revision_identity},
            )

        entry_payload = artifact.get("entry")
        try:
            draft = CreateEditorialEntryRequest.model_validate(entry_payload)
        except ValidationError as exc:
            raise AppError(
                status_code=422,
                code="EDITORIAL_EXPORT_NOT_APPROVED",
                message="retained editorial export entry cannot be validated",
            ) from exc
        await self._verified_source_snapshot(draft, action="bundle intake")
        await self._release_assurance_snapshot(draft)
        return reconstructed["artifact"]

    @asynccontextmanager
    async def verify_approved_export_for_candidate_finalization(
        self,
        artifact: object,
        artifact_sha256: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Hold current authority facts stable through Candidate persistence."""

        await self._acquire_candidate_finalization_fence()
        await self._lock_canonical_records(self._candidate_finalization_authority_record_ids(artifact))
        yield await self.verify_approved_export(artifact, artifact_sha256)

    async def record_candidate_publication(
        self,
        artifact: dict[str, Any],
        *,
        candidate_identity: str,
        published_knowledge_version_identity: str,
        actor_identity: str,
    ) -> None:
        """Append the authority lifecycle facts that make a verified Candidate retrievable."""

        entry_id = artifact.get("entry_id")
        entry_identity = artifact.get("entry_identity")
        revision_identity = artifact.get("editorial_revision_identity")
        if (
            not isinstance(entry_id, str)
            or not entry_id
            or not isinstance(entry_identity, str)
            or not entry_identity
            or not isinstance(revision_identity, str)
            or not revision_identity
        ):
            raise AppError(
                status_code=409,
                code="EDITORIAL_PUBLICATION_AUTHORITY_INVALID",
                message="Candidate publication cannot identify its verified editorial authority",
            )
        entry, events = await self._entry_and_events(entry_id, for_update=True)
        if entry.stable_id != entry_identity:
            raise AppError(
                status_code=409,
                code="EDITORIAL_PUBLICATION_AUTHORITY_INVALID",
                message="Candidate publication entry identity does not match editorial authority",
            )
        current = self._latest_event(events)
        _draft, current_revision_identity = await self._current_draft(self._payload(entry), current)
        if current_revision_identity != revision_identity:
            raise AppError(
                status_code=409,
                code="EDITORIAL_PUBLICATION_AUTHORITY_INVALID",
                message="Candidate publication revision is no longer current",
            )
        roles = self._roles_for_revision(events, revision_identity)
        if current.to_state == EntryLifecycleState.EDITORIAL_REVIEW.value:
            await self._append_entry_event(
                entry,
                events,
                event_type=CanonicalEventType.STATE_CHANGED,
                from_state=EntryLifecycleState.EDITORIAL_REVIEW.value,
                to_state=validate_transition(
                    EntryLifecycleState,
                    EntryLifecycleState.EDITORIAL_REVIEW,
                    EntryLifecycleState.CANDIDATE_BUILD,
                ).value,
                action="candidate_build_finalized",
                revision_identity=revision_identity,
                actor_identity=actor_identity,
                roles=roles,
                extra={"candidate_identity": candidate_identity},
            )
            await self.session.flush()
            entry, events = await self._entry_and_events(entry_id, for_update=True)
            current = self._latest_event(events)
        if current.to_state == EntryLifecycleState.CANDIDATE_BUILD.value:
            await self._append_entry_event(
                entry,
                events,
                event_type=CanonicalEventType.PUBLISHED,
                from_state=EntryLifecycleState.CANDIDATE_BUILD.value,
                to_state=validate_transition(
                    EntryLifecycleState,
                    EntryLifecycleState.CANDIDATE_BUILD,
                    EntryLifecycleState.PUBLISHED,
                ).value,
                action="candidate_published",
                revision_identity=revision_identity,
                actor_identity=actor_identity,
                roles=roles,
                extra={
                    "candidate_identity": candidate_identity,
                    "published_knowledge_version_identity": published_knowledge_version_identity,
                },
            )
            return
        if current.to_state != EntryLifecycleState.PUBLISHED.value:
            raise AppError(
                status_code=409,
                code="EDITORIAL_PUBLICATION_AUTHORITY_INVALID",
                message="Candidate publication cannot advance the current editorial lifecycle state",
                detail={"current_state": current.to_state},
            )

    async def get_projection(self, entry_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        entry, events = await self._entry_and_events(entry_id)
        entry_payload = self._payload(entry)
        current = self._latest_event(events)
        draft, revision_identity = await self._current_draft(entry_payload, current)
        roles = self._roles_for_revision(events, revision_identity)
        approval = self._approval_for_revision(events, revision_identity)
        revision_payload = self._payload(await self._revision(revision_identity))
        maintainer_acceptance = self._maintainer_acceptance_for_revision(events, revision_identity)
        sources = await self._source_projections(draft)
        eligibility = evaluate_answer_eligibility(
            lifecycle_state=current.to_state,
            approval_status=(approval or {}).get("status", "pending"),
            source_availability=[source["availability"] for source in sources],
            needs_review_at=self._needs_review_at(events, revision_identity, current),
            applicability_explicit=bool(draft.applicability_conditions),
            known_contradiction=self._revision_flag(events, revision_identity, "known_contradiction"),
            integrity_defect=self._revision_flag(events, revision_identity, "integrity_defect"),
            decisive_source_loss=self._revision_flag(events, revision_identity, "decisive_source_loss")
            or self._published_revision_has_decisive_source_loss(events, draft),
            now=now,
        )
        return {
            "entry_id": entry.identity_value,
            "entry_identity": entry.stable_id,
            "revision_identity": revision_identity,
            "lifecycle_state": current.to_state,
            "author_identity": entry_payload.get("author_identity"),
            "approving_reviewer_identity": roles.get("approving_reviewer_identity"),
            "accountable_maintainer_identity": roles.get("accountable_maintainer_identity"),
            "maintainer_acceptance": maintainer_acceptance or {"status": "pending"},
            "approval": approval
            or {
                "status": (
                    "pending_lightweight_acceptance"
                    if revision_payload.get("change_kind") == EditorialRevisionChangeKind.WORDING_ONLY.value
                    else "pending"
                )
            },
            "review_validation_reasons": review_validation_reasons(draft),
            "sources": sources,
            "eligibility": eligibility,
            "answer_eligible": eligibility["answer_eligible"],
        }

    async def get_retrieval_authority(self, entry_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        """Resolve the current authority facts required before Pilot ranking."""

        projection = await self.get_projection(entry_id, now=now)
        authority: dict[str, Any] = {
            "entry_id": projection["entry_id"],
            "entry_identity": projection["entry_identity"],
            "editorial_revision_identity": projection["revision_identity"],
            "lifecycle_state": projection["lifecycle_state"],
            "answer_eligible": projection["answer_eligible"],
            "eligibility_reasons": list(projection["eligibility"]["reasons"]),
        }
        if projection["answer_eligible"] is not True:
            return authority

        entry, events = await self._entry_and_events(entry_id)
        current = self._latest_event(events)
        draft, revision_identity = await self._current_draft(self._payload(entry), current)
        if revision_identity != projection["revision_identity"]:
            raise RuntimeError("retrieval authority revision changed during resolution")

        sources = await self._verified_source_snapshot(draft, action="retrieval")
        release_assurance_snapshot = await self._release_assurance_snapshot(draft)
        source_by_id: dict[str, dict[str, str]] = {}
        source_definitions: list[dict[str, str]] = []
        for source_snapshot in sources:
            source_identity = source_snapshot.get("source_identity")
            source_definition = source_snapshot.get("source")
            if (
                not isinstance(source_identity, str)
                or not isinstance(source_definition, dict)
                or not isinstance(source_definition.get("source_id"), str)
                or source_definition.get("access_scope") not in {"public", "controlled_internal"}
            ):
                raise RuntimeError("retrieval authority source snapshot is invalid")
            access_scope = str(source_definition["access_scope"])
            source_projection = {
                "source_identity": source_identity,
                "title": str(source_definition.get("title") or ""),
                "authority": str(source_definition.get("authority") or ""),
                "version": str(source_definition.get("version_or_date") or ""),
                "access_scope": access_scope,
            }
            if not all(source_projection[key] for key in ("title", "authority", "version")):
                raise RuntimeError("retrieval authority source snapshot is invalid")
            if access_scope == "public":
                public_url = source_definition.get("public_url")
                if not isinstance(public_url, str) or not public_url:
                    raise RuntimeError("retrieval authority public source locator is invalid")
                source_projection["public_url"] = public_url
            else:
                controlled_locator = source_definition.get("controlled_locator")
                if not isinstance(controlled_locator, str) or not controlled_locator:
                    raise RuntimeError("retrieval authority controlled source locator is invalid")
                source_projection["controlled_locator"] = controlled_locator
            source_by_id[source_definition["source_id"]] = {
                "source_identity": source_identity,
                "availability": str(source_snapshot["availability"]),
                "access_scope": access_scope,
            }
            source_definitions.append(source_projection)

        body = draft.body if isinstance(draft.body, dict) else {}
        decision_query = body.get("decision_query")
        if not isinstance(decision_query, str) or not decision_query.strip():
            raise RuntimeError("retrieval authority decision query is missing")
        relationships = draft.section_source_relationships
        if not isinstance(relationships, list):
            raise RuntimeError("retrieval authority section-source relationships are missing")
        by_section: dict[str, list[dict[str, str]]] = {}
        for relationship in relationships:
            if not isinstance(relationship, dict):
                raise RuntimeError("retrieval authority section-source relationship is invalid")
            section_id = relationship.get("section_id")
            source_ids = relationship.get("source_ids")
            if (
                not isinstance(section_id, str)
                or not section_id
                or section_id in by_section
                or not isinstance(source_ids, list)
                or not source_ids
            ):
                raise RuntimeError("retrieval authority section-source relationship is invalid")
            section_sources: list[dict[str, str]] = []
            seen_source_ids: set[str] = set()
            for source_id in source_ids:
                if not isinstance(source_id, str) or source_id in seen_source_ids or source_id not in source_by_id:
                    raise RuntimeError("retrieval authority section-source relationship is invalid")
                seen_source_ids.add(source_id)
                section_sources.append(dict(source_by_id[source_id]))
            by_section[section_id] = sorted(section_sources, key=lambda item: item["source_identity"])
        if set(by_section) != set(body):
            raise RuntimeError("retrieval authority section-source relationships do not cover the retained body")

        authority.update(
            {
                "section_source_relationships": by_section,
                "assurance_level": draft.assurance_level,
                "applicability_conditions": list(draft.applicability_conditions or []),
                "non_applicability_conditions": list(draft.non_applicability_conditions or []),
                "freshness_triggers": list(draft.freshness_triggers or []),
                "release_assurance_snapshot": release_assurance_snapshot,
                "decision_query": decision_query.strip(),
                "entry_title": draft.title,
                "coverage_position": draft.coverage_position,
                "review_date": draft.review_date,
                "applicable_versions": list(draft.applicable_versions or []),
                "source_definitions": source_definitions,
            }
        )
        return authority

    async def get_retrieval_authority_for_revision(
        self,
        entry_id: str,
        revision_identity: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Revalidate current source facts for one immutable published revision."""

        entry, events = await self._entry_and_events(entry_id)
        revision = await self._revision(revision_identity)
        revision_payload = self._payload(revision)
        draft_data = revision_payload.get("draft")
        if (
            revision_payload.get("entry_identity") != entry.stable_id
            or not isinstance(draft_data, dict)
        ):
            raise RuntimeError("published retrieval authority revision is invalid")
        try:
            draft = CreateEditorialEntryRequest.model_validate(draft_data)
        except ValueError as exc:
            raise RuntimeError("published retrieval authority draft is invalid") from exc
        if draft.entry_id != entry_id:
            raise RuntimeError("published retrieval authority entry identity is invalid")

        sources = await self._verified_source_snapshot(draft, action="published-version retrieval")
        approval = self._approval_for_revision(events, revision_identity)
        revision_events = [
            event
            for event in events
            if self._payload(event).get("revision_identity") == revision_identity
        ]
        if not revision_events:
            raise RuntimeError("published retrieval authority revision has no lifecycle event")
        historical_current = self._latest_event(revision_events)
        eligibility = evaluate_answer_eligibility(
            lifecycle_state=EntryLifecycleState.PUBLISHED.value,
            approval_status=(approval or {}).get("status", "pending"),
            source_availability=[source["availability"] for source in sources],
            needs_review_at=self._needs_review_at(events, revision_identity, historical_current),
            applicability_explicit=bool(draft.applicability_conditions),
            known_contradiction=self._revision_flag(events, revision_identity, "known_contradiction"),
            integrity_defect=self._revision_flag(events, revision_identity, "integrity_defect"),
            decisive_source_loss=(
                self._revision_flag(events, revision_identity, "decisive_source_loss")
                or self._published_revision_has_decisive_source_loss(events, draft)
            ),
            now=now,
        )
        authority: dict[str, Any] = {
            "entry_id": entry_id,
            "entry_identity": entry.stable_id,
            "editorial_revision_identity": revision_identity,
            "lifecycle_state": EntryLifecycleState.PUBLISHED.value,
            "answer_eligible": eligibility["answer_eligible"],
            "eligibility_reasons": list(eligibility["reasons"]),
        }
        if eligibility["answer_eligible"] is not True:
            return authority

        release_assurance_snapshot = await self._release_assurance_snapshot(draft)
        source_by_id: dict[str, dict[str, str]] = {}
        source_definitions: list[dict[str, str]] = []
        for source_snapshot in sources:
            source_identity = source_snapshot.get("source_identity")
            source_definition = source_snapshot.get("source")
            if (
                not isinstance(source_identity, str)
                or not isinstance(source_definition, dict)
                or not isinstance(source_definition.get("source_id"), str)
                or source_definition.get("access_scope") not in {"public", "controlled_internal"}
            ):
                raise RuntimeError("published retrieval authority source snapshot is invalid")
            access_scope = str(source_definition["access_scope"])
            source_projection = {
                "source_identity": source_identity,
                "title": str(source_definition.get("title") or ""),
                "authority": str(source_definition.get("authority") or ""),
                "version": str(source_definition.get("version_or_date") or ""),
                "access_scope": access_scope,
            }
            if not all(source_projection[key] for key in ("title", "authority", "version")):
                raise RuntimeError("published retrieval authority source snapshot is invalid")
            if access_scope == "public":
                public_url = source_definition.get("public_url")
                if not isinstance(public_url, str) or not public_url:
                    raise RuntimeError("published retrieval authority public source locator is invalid")
                source_projection["public_url"] = public_url
            else:
                controlled_locator = source_definition.get("controlled_locator")
                if not isinstance(controlled_locator, str) or not controlled_locator:
                    raise RuntimeError("published retrieval authority controlled source locator is invalid")
                source_projection["controlled_locator"] = controlled_locator
            source_by_id[source_definition["source_id"]] = {
                "source_identity": source_identity,
                "availability": str(source_snapshot["availability"]),
                "access_scope": access_scope,
            }
            source_definitions.append(source_projection)

        body = draft.body if isinstance(draft.body, dict) else {}
        decision_query = body.get("decision_query")
        if not isinstance(decision_query, str) or not decision_query.strip():
            raise RuntimeError("published retrieval authority decision query is missing")
        relationships = draft.section_source_relationships
        if not isinstance(relationships, list):
            raise RuntimeError("published retrieval authority section-source relationships are missing")
        by_section: dict[str, list[dict[str, str]]] = {}
        for relationship in relationships:
            if not isinstance(relationship, dict):
                raise RuntimeError("published retrieval authority section-source relationship is invalid")
            section_id = relationship.get("section_id")
            source_ids = relationship.get("source_ids")
            if (
                not isinstance(section_id, str)
                or not section_id
                or section_id in by_section
                or not isinstance(source_ids, list)
                or not source_ids
            ):
                raise RuntimeError("published retrieval authority section-source relationship is invalid")
            section_sources: list[dict[str, str]] = []
            seen_source_ids: set[str] = set()
            for source_id in source_ids:
                if not isinstance(source_id, str) or source_id in seen_source_ids or source_id not in source_by_id:
                    raise RuntimeError("published retrieval authority section-source relationship is invalid")
                seen_source_ids.add(source_id)
                section_sources.append(dict(source_by_id[source_id]))
            by_section[section_id] = sorted(section_sources, key=lambda item: item["source_identity"])
        if set(by_section) != set(body):
            raise RuntimeError("published retrieval authority section-source relationships do not cover the retained body")
        authority.update(
            {
                "section_source_relationships": by_section,
                "assurance_level": draft.assurance_level,
                "applicability_conditions": list(draft.applicability_conditions or []),
                "non_applicability_conditions": list(draft.non_applicability_conditions or []),
                "freshness_triggers": list(draft.freshness_triggers or []),
                "release_assurance_snapshot": release_assurance_snapshot,
                "decision_query": decision_query.strip(),
                "entry_title": draft.title,
                "coverage_position": draft.coverage_position,
                "review_date": draft.review_date,
                "applicable_versions": list(draft.applicable_versions or []),
                "source_definitions": source_definitions,
                "source_revalidated_at": (now.isoformat() if now is not None else None),
            }
        )
        return authority

    async def _entry_and_events(
        self,
        entry_id: str,
        *,
        for_update: bool = False,
    ) -> tuple[CanonicalRecordModel, list[CanonicalEventModel]]:
        identity = StableIdentity(StableIdentityKind.ENTRY, entry_id)
        if for_update:
            entry = await self.session.scalar(
                select(CanonicalRecordModel)
                .where(CanonicalRecordModel.stable_id == identity.stable_id)
                .with_for_update()
            )
        else:
            entry = await self.session.get(CanonicalRecordModel, identity.stable_id)
        if entry is None:
            raise AppError(status_code=404, code="EDITORIAL_ENTRY_NOT_FOUND", message="editorial entry was not found")
        result = await self.session.execute(
            select(CanonicalEventModel)
            .where(
                CanonicalEventModel.aggregate_id == entry.stable_id,
                CanonicalEventModel.aggregate_kind == StableIdentityKind.ENTRY.value,
            )
            .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
        )
        events = list(result.scalars())
        if not events:
            raise RuntimeError("editorial entry has no lifecycle event")
        return entry, events

    @staticmethod
    def _candidate_finalization_authority_record_ids(artifact: object) -> set[str]:
        if not isinstance(artifact, dict):
            return set()
        identities: set[str] = set()

        def add_identity(value: object, expected_kind: StableIdentityKind) -> None:
            if not isinstance(value, str):
                return
            try:
                identity = StableIdentity.from_stable_id(value)
            except ValueError:
                return
            if identity.kind is expected_kind:
                identities.add(identity.stable_id)

        add_identity(artifact.get("entry_identity"), StableIdentityKind.ENTRY)
        entry = artifact.get("entry")
        if not isinstance(entry, dict):
            return identities
        sources = entry.get("sources")
        if isinstance(sources, list):
            for source in sources:
                source_id = source.get("source_id") if isinstance(source, dict) else None
                if isinstance(source_id, str):
                    try:
                        identities.add(StableIdentity(StableIdentityKind.SOURCE, source_id).stable_id)
                    except ValueError:
                        continue
        source_snapshots = artifact.get("sources")
        if isinstance(source_snapshots, list):
            for source_snapshot in source_snapshots:
                if isinstance(source_snapshot, dict):
                    add_identity(source_snapshot.get("source_identity"), StableIdentityKind.SOURCE)
        release_assurance = entry.get("release_assurance")
        if isinstance(release_assurance, dict):
            for field in (
                "contract_identity",
                "calibration_identity",
                "frozen_acceptance_identity",
                "named_gate",
            ):
                raw_identity = release_assurance.get(field)
                if not isinstance(raw_identity, str):
                    continue
                try:
                    identities.add(StableIdentity.from_stable_id(raw_identity).stable_id)
                except ValueError:
                    continue
        return identities

    async def _lock_canonical_records(self, record_identities: set[str]) -> dict[str, CanonicalRecordModel]:
        if not record_identities:
            return {}
        result = await self.session.execute(
            select(CanonicalRecordModel)
            .where(CanonicalRecordModel.stable_id.in_(record_identities))
            .order_by(CanonicalRecordModel.stable_id.asc())
            .with_for_update()
        )
        return {record.stable_id: record for record in result.scalars()}

    async def _acquire_candidate_finalization_fence(self) -> None:
        bind = self.session.bind
        if bind is None or bind.dialect.name != "sqlite":
            return
        try:
            await self.session.execute(text("BEGIN IMMEDIATE"))
        except OperationalError as exc:
            await self.session.rollback()
            raise AppError(
                status_code=409,
                code="EDITORIAL_AUTHORITY_FENCE_UNAVAILABLE",
                message="Candidate finalization could not acquire the editorial authority fence",
            ) from exc

    async def _current_draft(
        self,
        entry_payload: dict[str, Any],
        event: CanonicalEventModel,
    ) -> tuple[CreateEditorialEntryRequest, str]:
        event_payload = self._payload(event)
        revision_identity = event_payload.get("revision_identity") or entry_payload.get("initial_revision_identity")
        if not isinstance(revision_identity, str):
            raise RuntimeError("editorial entry has no revision identity")
        revision = await self._revision(revision_identity)
        draft_data = self._payload(revision).get("draft")
        if not isinstance(draft_data, dict):
            raise RuntimeError("editorial revision draft is invalid")
        return CreateEditorialEntryRequest.model_validate(draft_data), revision_identity

    async def _revision(self, revision_identity: str) -> CanonicalRecordModel:
        revision = await self.session.get(CanonicalRecordModel, revision_identity)
        if revision is None or revision.identity_kind != StableIdentityKind.EDITORIAL_REVISION.value:
            raise RuntimeError("editorial revision is missing")
        return revision

    async def _ensure_source_records(self, draft: CreateEditorialEntryRequest, actor_identity: str) -> None:
        if not isinstance(draft.sources, list):
            return
        for source in draft.sources:
            if not isinstance(source, dict):
                continue
            source_id = source.get("source_id")
            if not isinstance(source_id, str):
                continue
            identity = StableIdentity(StableIdentityKind.SOURCE, source_id)
            existing = await self.session.get(CanonicalRecordModel, identity.stable_id)
            if existing is not None:
                if self._payload(existing).get("source") != self._source_definition(source):
                    raise AppError(
                        status_code=409,
                        code="EDITORIAL_SOURCE_ID_CONFLICT",
                        message="a stable source identity cannot be reused for a different source definition",
                        detail={"source_id": source_id},
                    )
                continue
            availability = SourceAvailabilityState.CHANGED_OR_UNREACHABLE_AWAITING_REVIEW.value
            self.session.add(
                CanonicalRecordModel(
                    stable_id=identity.stable_id,
                    identity_kind=identity.kind.value,
                    identity_value=identity.value,
                    state=availability,
                    record_class=CanonicalRecordClass.AUTHORITATIVE.value,
                    payload={
                        "schema": "editorial_source/v1",
                        "source": self._source_definition(source),
                    },
                )
            )
            self.session.add(
                CanonicalEventModel(
                    aggregate_id=identity.stable_id,
                    aggregate_kind=identity.kind.value,
                    event_type=CanonicalEventType.CREATED.value,
                    from_state=None,
                    to_state=availability,
                    payload={
                        "schema": "editorial_source_event/v1",
                        "action": "source_evidence_proposed",
                        "author_declared_availability": source.get("availability"),
                    },
                    recorded_by=actor_identity,
                )
            )

    async def _resolve_roles(self, draft: CreateEditorialEntryRequest, *, author_identity: str) -> dict[str, str]:
        reviewer = await self._editorial_principal(draft.approving_reviewer_username, "approving reviewer")
        maintainer = await self._editorial_principal(draft.accountable_maintainer_username, "accountable maintainer")
        return {
            "author_identity": author_identity,
            "approving_reviewer_identity": await IdentityAuditService(self.session).ensure_member_record(
                reviewer,
                admission_path="private_editorial_repository",
            ),
            "accountable_maintainer_identity": await IdentityAuditService(self.session).ensure_member_record(
                maintainer,
                admission_path="private_editorial_repository",
            ),
        }

    async def _editorial_principal(self, username: str | None, role_name: str) -> User:
        if not isinstance(username, str) or not username.strip():
            raise AppError(
                status_code=422,
                code="EDITORIAL_ENTRY_INVALID",
                message="editorial entry is incomplete for role assignment",
                detail={"reasons": [{"field": role_name.replace(" ", "_") + "_username", "code": "required"}]},
            )
        member = (await self.session.execute(select(User).where(User.username == username.strip()))).scalar_one_or_none()
        if member is None or not member.is_active or member.role == "admin":
            raise AppError(
                status_code=422,
                code="EDITORIAL_EDITORIAL_PRINCIPAL_INVALID",
                message="assigned editorial principal must be an active non-administrator member",
                detail={"username": username},
            )
        return member

    async def _append_entry_event(
        self,
        entry: CanonicalRecordModel,
        events: list[CanonicalEventModel],
        *,
        event_type: CanonicalEventType,
        from_state: str | None,
        to_state: str,
        action: str,
        revision_identity: str,
        actor_identity: str,
        roles: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.session.add(
            CanonicalEventModel(
                aggregate_id=entry.stable_id,
                aggregate_kind=StableIdentityKind.ENTRY.value,
                event_type=event_type.value,
                from_state=from_state,
                to_state=to_state,
                payload={
                    "schema": "editorial_authority_event/v1",
                    "sequence": self._event_sequence(events) + 1,
                    "action": action,
                    "revision_identity": revision_identity,
                    **({"roles": roles} if roles is not None else {}),
                    **(extra or {}),
                },
                recorded_by=actor_identity,
            )
        )

    async def _verified_source_snapshot(
        self,
        draft: CreateEditorialEntryRequest,
        *,
        action: str,
    ) -> list[dict[str, Any]]:
        """Resolve current source authority for a new approval or export action."""

        reasons: list[dict[str, str]] = []
        snapshots: list[dict[str, Any]] = []
        sources = draft.sources if isinstance(draft.sources, list) else []
        if not sources:
            reasons.append(
                {
                    "field": "sources",
                    "code": "source_authority_missing",
                    "message": "an editorial action requires at least one retained authoritative source record",
                }
            )
        for index, source in enumerate(sources):
            prefix = f"sources[{index}]"
            if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
                reasons.append(
                    {
                        "field": f"{prefix}.source_id",
                        "code": "source_authority_missing",
                        "message": "source has no stable source identity",
                    }
                )
                continue
            source_id = source["source_id"]
            identity = StableIdentity(StableIdentityKind.SOURCE, source_id)
            record = await self.session.get(CanonicalRecordModel, identity.stable_id)
            source_definition = self._payload(record).get("source") if record is not None else None
            if (
                record is None
                or record.identity_kind != StableIdentityKind.SOURCE.value
                or record.record_class != CanonicalRecordClass.AUTHORITATIVE.value
                or self._payload(record).get("schema") != "editorial_source/v1"
                or not isinstance(source_definition, dict)
                or source_definition != self._source_definition(source)
            ):
                reasons.append(
                    {
                        "field": f"{prefix}.source_id",
                        "code": "source_authority_missing",
                        "message": "source has no retained authoritative source record with its immutable definition",
                    }
                )
                continue
            availability_event = await self._qualified_source_availability_event(record)
            availability = availability_event.to_state if availability_event is not None else "unknown"
            if availability_event is None or availability != SourceAvailabilityState.VERIFIED_USABLE.value:
                reasons.append(
                    {
                        "field": f"{prefix}.availability",
                        "code": "source_unavailable",
                        "message": "the accountable maintainer has not recorded qualified verified_usable availability",
                    }
                )
                continue
            snapshots.append(
                {
                    "source_identity": record.stable_id,
                    "availability": availability,
                    "source": source_definition,
                    "source_definition_sha256": self._sha256(source_definition),
                    "availability_event": self._event_snapshot(availability_event),
                }
            )
        if reasons:
            raise AppError(
                status_code=409,
                code="EDITORIAL_SOURCE_UNAVAILABLE",
                message=f"editorial sources are not usable for {action}",
                detail={"reasons": reasons},
            )
        return sorted(snapshots, key=lambda item: item["source_identity"])

    async def _source_projections(self, draft: CreateEditorialEntryRequest) -> list[dict[str, str]]:
        projected: list[dict[str, str]] = []
        for source in draft.sources or []:
            if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
                continue
            identity = StableIdentity(StableIdentityKind.SOURCE, source["source_id"])
            record = await self.session.get(CanonicalRecordModel, identity.stable_id)
            availability = await self._source_availability(record) if record is not None else "unknown"
            projected.append(
                {
                    "source_id": source["source_id"],
                    "source_identity": identity.stable_id,
                    "availability": availability,
                    "access_scope": str(source.get("access_scope") or "unknown"),
                }
            )
        return sorted(projected, key=lambda item: item["source_identity"])

    async def _qualified_source_availability_event(self, source: CanonicalRecordModel) -> CanonicalEventModel | None:
        if (
            source.identity_kind != StableIdentityKind.SOURCE.value
            or source.record_class != CanonicalRecordClass.AUTHORITATIVE.value
            or self._payload(source).get("schema") != "editorial_source/v1"
        ):
            return None
        result = await self.session.execute(
            select(CanonicalEventModel)
            .where(
                CanonicalEventModel.aggregate_id == source.stable_id,
                CanonicalEventModel.aggregate_kind == StableIdentityKind.SOURCE.value,
            )
            .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
        )
        for event in reversed(list(result.scalars())):
            payload = self._payload(event)
            if (
                payload.get("schema") != "editorial_source_event/v1"
                or payload.get("action") != "source_availability_recorded"
            ):
                continue
            if (
                event.event_type != CanonicalEventType.STATUS_CHANGED.value
                or event.to_state not in {state.value for state in SourceAvailabilityState}
                or not await self._source_event_has_accepted_maintainer(event)
            ):
                return None
            return event
        return None

    async def _source_event_has_accepted_maintainer(self, event: CanonicalEventModel) -> bool:
        payload = self._payload(event)
        entry_identity = payload.get("entry_identity")
        revision_identity = payload.get("editorial_revision_identity")
        acceptance_event_id = payload.get("maintainer_acceptance_event_id")
        if (
            not isinstance(entry_identity, str)
            or not entry_identity
            or not isinstance(revision_identity, str)
            or not revision_identity
            or not isinstance(acceptance_event_id, str)
            or not acceptance_event_id
            or not isinstance(event.recorded_by, str)
            or not event.recorded_by
        ):
            return False
        try:
            entry = StableIdentity.from_stable_id(entry_identity)
            revision = StableIdentity.from_stable_id(revision_identity)
        except ValueError:
            return False
        if entry.kind is not StableIdentityKind.ENTRY or revision.kind is not StableIdentityKind.EDITORIAL_REVISION:
            return False
        entry_record = await self.session.get(CanonicalRecordModel, entry.stable_id)
        revision_record = await self.session.get(CanonicalRecordModel, revision.stable_id)
        if (
            entry_record is None
            or entry_record.identity_kind != StableIdentityKind.ENTRY.value
            or revision_record is None
            or revision_record.identity_kind != StableIdentityKind.EDITORIAL_REVISION.value
            or self._payload(revision_record).get("entry_identity") != entry.stable_id
        ):
            return False
        draft_data = self._payload(revision_record).get("draft")
        sources = draft_data.get("sources") if isinstance(draft_data, dict) else None
        if not isinstance(sources, list):
            return False
        source_belongs_to_revision = False
        for source in sources:
            source_id = source.get("source_id") if isinstance(source, dict) else None
            if not isinstance(source_id, str):
                continue
            try:
                source_identity = StableIdentity(StableIdentityKind.SOURCE, source_id)
            except ValueError:
                continue
            if source_identity.stable_id == event.aggregate_id:
                source_belongs_to_revision = True
                break
        if not source_belongs_to_revision:
            return False
        result = await self.session.execute(
            select(CanonicalEventModel)
            .where(
                CanonicalEventModel.aggregate_id == entry.stable_id,
                CanonicalEventModel.aggregate_kind == StableIdentityKind.ENTRY.value,
            )
            .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
        )
        accepted_maintainer = self._maintainer_acceptance_for_revision(
            list(result.scalars()),
            revision.stable_id,
        )
        return (
            accepted_maintainer is not None
            and accepted_maintainer["event_id"] == acceptance_event_id
            and accepted_maintainer["maintainer_identity"] == event.recorded_by
        )

    async def _source_availability(self, source: CanonicalRecordModel) -> str:
        event = await self._qualified_source_availability_event(source)
        return event.to_state if event is not None else "unknown"

    async def _raise_for_usable_sources(self, draft: CreateEditorialEntryRequest, *, action: str) -> None:
        await self._verified_source_snapshot(draft, action=action)

    @staticmethod
    def _claim_evidence_contract_snapshot(
        draft: CreateEditorialEntryRequest,
        *,
        entry_identity: str,
        revision_identity: str,
    ) -> dict[str, str]:
        if draft.assurance_level != "claim_linked":
            return {}
        if draft.claim_evidence_contract is None:
            try:
                contract = build_candidate_claim_evidence_contract(
                    entry_identity=entry_identity,
                    editorial_revision_identity=revision_identity,
                    claims=draft.claims,
                )
            except CandidateClaimEvidenceContractError as exc:
                raise AppError(
                    status_code=422,
                    code="EDITORIAL_ENTRY_INVALID",
                    message="Claim-Linked editorial export has no valid reviewed Claim-Evidence Links",
                ) from exc
            return {
                "claim_evidence_contract": contract.canonical_json,
                "claim_evidence_contract_sha256": contract.sha256,
            }
        try:
            contract = parse_claim_evidence_contract(draft.claim_evidence_contract)
        except ClaimEvidenceContractError as exc:
            raise AppError(
                status_code=422,
                code="EDITORIAL_ENTRY_INVALID",
                message="Claim-Linked editorial export has no valid frozen Claim-Evidence contract",
            ) from exc
        return {
            "claim_evidence_contract": contract.canonical_json,
            "claim_evidence_contract_sha256": contract.sha256,
        }

    async def _release_assurance_snapshot(self, draft: CreateEditorialEntryRequest) -> dict[str, Any] | None:
        if draft.assurance_level != "release_assured":
            return None
        assurance = draft.release_assurance
        if not isinstance(assurance, dict):
            return None
        references = (
            (
                "contract_identity",
                {StableIdentityKind.PRODUCT_PATH, StableIdentityKind.PRODUCT_REVISION},
                {CanonicalRecordClass.AUTHORITATIVE.value, CanonicalRecordClass.IMMUTABLE.value},
            ),
            (
                "calibration_identity",
                {StableIdentityKind.CONFIGURATION, StableIdentityKind.CAPABILITY},
                {CanonicalRecordClass.AUTHORITATIVE.value, CanonicalRecordClass.IMMUTABLE.value},
            ),
            (
                "frozen_acceptance_identity",
                {StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD},
                {CanonicalRecordClass.IMMUTABLE.value},
            ),
            (
                "named_gate",
                {StableIdentityKind.CAPABILITY},
                {CanonicalRecordClass.AUTHORITATIVE.value, CanonicalRecordClass.IMMUTABLE.value},
            ),
        )
        reasons: list[dict[str, str]] = []
        resolved: dict[str, CanonicalRecordModel] = {}
        for field, allowed_kinds, allowed_record_classes in references:
            raw_identity = assurance.get(field)
            if not isinstance(raw_identity, str):
                continue
            try:
                identity = StableIdentity.from_stable_id(raw_identity)
            except ValueError:
                continue
            record = await self.session.get(CanonicalRecordModel, identity.stable_id)
            if (
                record is None
                or record.identity_kind not in {kind.value for kind in allowed_kinds}
                or record.record_class not in allowed_record_classes
                or (
                    field == "frozen_acceptance_identity"
                    and self._payload(record).get("schema") != "delivery_acceptance_record/v1"
                )
            ):
                reasons.append(
                    {
                        "field": f"release_assurance.{field}",
                        "code": "unverified_identity",
                        "message": "Release-Assured references must resolve to retained canonical authority records",
                    }
                )
                continue
            resolved[field] = record

        frozen_acceptance = resolved.get("frozen_acceptance_identity")
        active_acceptance_event: CanonicalEventModel | None = None
        if frozen_acceptance is not None:
            entry_identity = StableIdentity(StableIdentityKind.ENTRY, draft.entry_id).stable_id
            acceptance_payload = self._validated_delivery_acceptance_payload(frozen_acceptance)
            if acceptance_payload is None:
                reasons.append(
                    {
                        "field": "release_assurance.frozen_acceptance_identity",
                        "code": "acceptance_record_invalid",
                        "message": "frozen delivery acceptance must retain a complete canonical acceptance record",
                    }
                )
            elif not self._delivery_acceptance_covers_release_assurance(
                acceptance_payload.model_dump(mode="json"),
                entry_identity=entry_identity,
                contract_identity=assurance.get("contract_identity"),
                calibration_identity=assurance.get("calibration_identity"),
                named_gate=assurance.get("named_gate"),
            ):
                reasons.append(
                    {
                        "field": "release_assurance.frozen_acceptance_identity",
                        "code": "acceptance_scope_mismatch",
                        "message": (
                            "frozen delivery acceptance must cover this entry and every named "
                            "Release-Assured authority record"
                        ),
                    }
                )
            else:
                active_acceptance_event = await self._active_delivery_acceptance_status_event(
                    frozen_acceptance,
                    acceptance_payload,
                )
            if active_acceptance_event is None and not reasons:
                reasons.append(
                    {
                        "field": "release_assurance.frozen_acceptance_identity",
                        "code": "acceptance_not_active",
                        "message": "Release-Assured delivery acceptance must have an active retained status event",
                    }
                )
        if reasons:
            raise AppError(
                status_code=422,
                code="EDITORIAL_RELEASE_ASSURANCE_UNVERIFIED",
                message="Release-Assured references are not retained and active canonical authority facts",
                detail={"reasons": reasons},
            )
        if active_acceptance_event is None:
            raise RuntimeError("Release-Assured snapshot is missing its active delivery acceptance event")
        return {
            "schema": "editorial_release_assurance_snapshot/v1",
            "entry_identity": StableIdentity(StableIdentityKind.ENTRY, draft.entry_id).stable_id,
            "records": [
                {
                    "field": field,
                    "identity": resolved[field].stable_id,
                    "record_class": resolved[field].record_class,
                    "payload_sha256": self._sha256(self._payload(resolved[field])),
                }
                for field, _, _ in references
            ],
            "frozen_acceptance_status": self._event_snapshot(active_acceptance_event),
        }

    async def _active_delivery_acceptance_status_event(
        self,
        record: CanonicalRecordModel,
        acceptance_payload: CreateDeliveryAcceptanceRecordRequest | None = None,
    ) -> CanonicalEventModel | None:
        validated_payload = acceptance_payload or self._validated_delivery_acceptance_payload(record)
        if validated_payload is None:
            return None
        expected_verified_checks = {
            check.check_id: list(check.evidence_links)
            for check in validated_payload.checks
            if check.result in {AcceptanceCheckResult.PASSED, AcceptanceCheckResult.CARRIED_FORWARD}
        }
        result = await self.session.execute(
            select(CanonicalEventModel)
            .where(
                CanonicalEventModel.aggregate_id == record.stable_id,
                CanonicalEventModel.aggregate_kind == StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value,
            )
            .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
        )
        for event in reversed(list(result.scalars())):
            payload = self._payload(event)
            if payload.get("schema") != "delivery_acceptance_status/v1":
                continue
            if event.event_type != CanonicalEventType.STATUS_CHANGED.value:
                return None
            if (
                event.from_state != AcceptanceStatus.AT_RISK.value
                or event.to_state != AcceptanceStatus.ACTIVE.value
                or payload.get("reason_code") != AcceptanceStatusReason.CHECKS_VERIFIED.value
                or not self._is_member_identity(event.recorded_by)
            ):
                return None
            status_payload = {
                "status": event.to_state,
                **{
                    field: payload[field]
                    for field in UpdateDeliveryAcceptanceStatusRequest.model_fields
                    if field in payload
                },
            }
            try:
                verified_status = UpdateDeliveryAcceptanceStatusRequest.model_validate(status_payload)
            except ValidationError:
                return None
            actual_verified_checks = {
                check.check_id: list(check.evidence_links) for check in verified_status.verified_checks
            }
            if actual_verified_checks != expected_verified_checks:
                return None
            return event
        return None

    @staticmethod
    def _validated_delivery_acceptance_payload(
        record: CanonicalRecordModel,
    ) -> CreateDeliveryAcceptanceRecordRequest | None:
        if (
            record.identity_kind != StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value
            or record.record_class != CanonicalRecordClass.IMMUTABLE.value
        ):
            return None
        payload = EditorialAuthorityService._payload(record)
        if payload.get("schema") != "delivery_acceptance_record/v1":
            return None
        candidate = {
            field: payload[field]
            for field in CreateDeliveryAcceptanceRecordRequest.model_fields
            if field in payload
        }
        try:
            return CreateDeliveryAcceptanceRecordRequest.model_validate(candidate)
        except ValidationError:
            return None

    @staticmethod
    def _delivery_acceptance_covers_release_assurance(
        payload: dict[str, Any],
        *,
        entry_identity: str,
        contract_identity: object,
        calibration_identity: object,
        named_gate: object,
    ) -> bool:
        content_identities = payload.get("content_identities")
        product_identities = payload.get("product_identities")
        affected_scope = payload.get("affected_scope")
        entry_scope = affected_scope.get("entry_identities") if isinstance(affected_scope, dict) else None
        required_product_identities = (contract_identity, calibration_identity, named_gate)
        return (
            isinstance(content_identities, list)
            and entry_identity in content_identities
            and isinstance(entry_scope, list)
            and entry_identity in entry_scope
            and isinstance(product_identities, list)
            and all(isinstance(identity, str) for identity in required_product_identities)
            and all(isinstance(identity, str) for identity in product_identities)
            and set(required_product_identities).issubset(set(product_identities))
        )

    @staticmethod
    def _latest_event(events: list[CanonicalEventModel]) -> CanonicalEventModel:
        return max(events, key=lambda event: (EditorialAuthorityService._event_sequence([event]), event.occurred_at, event.id))

    @staticmethod
    def _event_sequence(events: list[CanonicalEventModel]) -> int:
        sequences = []
        for event in events:
            sequence = EditorialAuthorityService._payload(event).get("sequence")
            if isinstance(sequence, int):
                sequences.append(sequence)
        return max(sequences, default=0)

    @staticmethod
    def _roles_for_revision(events: list[CanonicalEventModel], revision_identity: str) -> dict[str, str]:
        for event in reversed(sorted(events, key=lambda item: (EditorialAuthorityService._event_sequence([item]), item.id))):
            payload = EditorialAuthorityService._payload(event)
            if (
                payload.get("schema") != "editorial_authority_event/v1"
                or payload.get("revision_identity") != revision_identity
            ):
                continue
            roles = EditorialAuthorityService._editorial_roles(payload.get("roles"))
            if roles is not None and EditorialAuthorityService._is_role_assignment_event(event, payload, roles):
                return roles
        return {}

    @staticmethod
    def _approval_for_revision(events: list[CanonicalEventModel], revision_identity: str) -> dict[str, Any] | None:
        for event in reversed(sorted(events, key=lambda item: (EditorialAuthorityService._event_sequence([item]), item.id))):
            payload = EditorialAuthorityService._payload(event)
            if (
                payload.get("schema") != "editorial_authority_event/v1"
                or payload.get("revision_identity") != revision_identity
                or payload.get("action") not in {"revision_approved", "wording_revision_accepted"}
            ):
                continue
            roles = EditorialAuthorityService._roles_for_revision(events, revision_identity)
            if (
                not roles
                or payload.get("roles") != roles
                or event.event_type != CanonicalEventType.STATUS_CHANGED.value
                or event.recorded_by != roles["approving_reviewer_identity"]
                or roles["author_identity"] == roles["approving_reviewer_identity"]
            ):
                continue
            if payload.get("action") == "revision_approved" and (
                event.from_state != EntryLifecycleState.EDITORIAL_REVIEW.value
                or event.to_state != EntryLifecycleState.EDITORIAL_REVIEW.value
            ):
                continue
            if payload.get("action") == "wording_revision_accepted" and (
                event.from_state != event.to_state
                or event.to_state
                not in {
                    EntryLifecycleState.DRAFT.value,
                    EntryLifecycleState.EVIDENCE_COLLECTED.value,
                    EntryLifecycleState.EDITORIAL_REVIEW.value,
                    EntryLifecycleState.NEEDS_REVIEW.value,
                    EntryLifecycleState.PUBLISHED.value,
                }
            ):
                continue
            return {
                "status": "approved" if payload.get("action") == "revision_approved" else "lightweight_accepted",
                "reviewer_identity": event.recorded_by,
                "roles": roles,
                "authority_snapshot": payload.get("authority_snapshot"),
                "event_id": event.id,
                "sequence": EditorialAuthorityService._event_sequence([event]),
            }
        return None

    @staticmethod
    def _retained_approval_snapshot(
        *,
        entry_identity: str,
        revision_identity: str,
        events: list[CanonicalEventModel],
        roles: dict[str, str],
        reviewer_identity: str,
        action: str,
        event_type: CanonicalEventType,
        from_state: str,
        to_state: str,
        sources: list[dict[str, Any]],
        release_assurance: dict[str, Any] | None,
    ) -> dict[str, Any]:
        status = "approved" if action == "revision_approved" else "lightweight_accepted"
        return {
            "schema": "editorial_approval_snapshot/v1",
            "entry_identity": entry_identity,
            "editorial_revision_identity": revision_identity,
            "roles": roles,
            "approval": {"status": status, "reviewer_identity": reviewer_identity},
            "sources": sources,
            "release_assurance": release_assurance,
            "editorial_audit": EditorialAuthorityService._export_audit(events, revision_identity)
            + [
                {
                    "sequence": EditorialAuthorityService._event_sequence(events) + 1,
                    "action": action,
                    "event_type": event_type.value,
                    "from_state": from_state,
                    "to_state": to_state,
                    "recorded_by": reviewer_identity,
                }
            ],
        }

    @staticmethod
    def _approval_authority_snapshot(
        approval: dict[str, Any],
        *,
        draft: CreateEditorialEntryRequest,
        entry_identity: str,
        revision_identity: str,
        events: list[CanonicalEventModel],
    ) -> dict[str, Any]:
        snapshot = approval.get("authority_snapshot")
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema") != "editorial_approval_snapshot/v1"
            or snapshot.get("entry_identity") != entry_identity
            or snapshot.get("editorial_revision_identity") != revision_identity
        ):
            EditorialAuthorityService._raise_export_snapshot_missing(
                "approved editorial revisions require a retained authority snapshot bound to their entry and revision",
            )
        expected_roles = approval.get("roles")
        snapshot_roles = EditorialAuthorityService._editorial_roles(snapshot.get("roles"))
        if (
            not isinstance(expected_roles, dict)
            or snapshot_roles is None
            or snapshot_roles != expected_roles
            or snapshot.get("approval")
            != {
                "status": approval.get("status"),
                "reviewer_identity": approval.get("reviewer_identity"),
            }
        ):
            EditorialAuthorityService._raise_export_snapshot_missing(
                "approved editorial revisions require retained reviewer and role authority proof",
            )

        expected_source_definitions: dict[str, dict[str, Any]] = {}
        for source in draft.sources or []:
            if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
                EditorialAuthorityService._raise_export_snapshot_missing(
                    "approved editorial revisions require stable source definitions for reconstruction",
                )
            identity = StableIdentity(StableIdentityKind.SOURCE, source["source_id"])
            expected_source_definitions[identity.stable_id] = EditorialAuthorityService._source_definition(source)
        source_snapshots = snapshot.get("sources")
        if not isinstance(source_snapshots, list) or not source_snapshots:
            EditorialAuthorityService._raise_export_snapshot_missing(
                "approved editorial revisions require retained verified source snapshots for reconstruction",
            )
        sources: list[dict[str, Any]] = []
        source_identities: set[str] = set()
        for source_snapshot in source_snapshots:
            if not isinstance(source_snapshot, dict):
                EditorialAuthorityService._raise_export_snapshot_missing(
                    "approved editorial source snapshots must remain structured authority data",
                )
            source_identity = source_snapshot.get("source_identity")
            source = source_snapshot.get("source")
            availability_event = source_snapshot.get("availability_event")
            if not isinstance(source_identity, str) or not isinstance(source, dict):
                EditorialAuthorityService._raise_export_snapshot_missing(
                    "approved editorial source snapshots must retain a stable source identity and definition",
                )
            if (
                source_identity in source_identities
                or source_identity not in expected_source_definitions
                or source_snapshot.get("availability") != SourceAvailabilityState.VERIFIED_USABLE.value
                or source != expected_source_definitions.get(source_identity)
                or source_snapshot.get("source_definition_sha256") != EditorialAuthorityService._sha256(source)
                or not EditorialAuthorityService._is_event_snapshot(
                    availability_event,
                    to_state=SourceAvailabilityState.VERIFIED_USABLE.value,
                )
            ):
                EditorialAuthorityService._raise_export_snapshot_missing(
                    "approved editorial source snapshots cannot prove verified immutable source authority",
                )
            source_identities.add(source_identity)
            sources.append(
                {
                    "source_identity": source_identity,
                    "availability": source_snapshot["availability"],
                    "source": source,
                    "source_definition_sha256": source_snapshot["source_definition_sha256"],
                    "availability_event": availability_event,
                }
            )
        if source_identities != set(expected_source_definitions):
            EditorialAuthorityService._raise_export_snapshot_missing(
                "approved editorial source snapshots must exactly match the approved revision sources",
            )
        release_assurance = snapshot.get("release_assurance")
        if draft.assurance_level == "release_assured":
            if not EditorialAuthorityService._release_assurance_snapshot_matches(
                release_assurance,
                draft=draft,
                entry_identity=entry_identity,
            ):
                EditorialAuthorityService._raise_export_snapshot_missing(
                    "Release-Assured reconstruction requires a retained frozen assurance snapshot bound to its revision",
                )
        elif release_assurance is not None:
            EditorialAuthorityService._raise_export_snapshot_missing(
                "non-Release-Assured reconstruction cannot retain unrelated assurance authority",
            )

        approval_sequence = approval.get("sequence")
        if not isinstance(approval_sequence, int) or approval_sequence < 1:
            EditorialAuthorityService._raise_export_snapshot_missing(
                "approved editorial revisions require a retained approval event sequence",
            )
        expected_audit = EditorialAuthorityService._export_audit(
            [
                event
                for event in events
                if EditorialAuthorityService._event_sequence([event]) <= approval_sequence
            ],
            revision_identity,
        )
        editorial_audit = snapshot.get("editorial_audit")
        if not isinstance(editorial_audit, list) or editorial_audit != expected_audit:
            EditorialAuthorityService._raise_export_snapshot_missing(
                "approved editorial revisions require an immutable approval-time editorial audit",
            )
        return {
            "roles": snapshot_roles,
            "approval": snapshot["approval"],
            "sources": sorted(sources, key=lambda item: item["source_identity"]),
            "release_assurance": release_assurance,
            "editorial_audit": editorial_audit,
        }

    @staticmethod
    def _editorial_roles(value: object) -> dict[str, str] | None:
        required_keys = {
            "author_identity",
            "approving_reviewer_identity",
            "accountable_maintainer_identity",
        }
        if not isinstance(value, dict) or set(value) != required_keys:
            return None
        roles = {str(key): str(role) for key, role in value.items() if isinstance(role, str)}
        if (
            len(roles) != len(required_keys)
            or not all(EditorialAuthorityService._is_member_identity(role) for role in roles.values())
        ):
            return None
        return roles

    @staticmethod
    def _is_role_assignment_event(
        event: CanonicalEventModel,
        payload: dict[str, Any],
        roles: dict[str, str],
    ) -> bool:
        action = payload.get("action")
        if action == "evidence_collected":
            return (
                event.event_type == CanonicalEventType.STATE_CHANGED.value
                and event.from_state == EntryLifecycleState.DRAFT.value
                and event.to_state == EntryLifecycleState.EVIDENCE_COLLECTED.value
                and event.recorded_by == roles["author_identity"]
            )
        if action in {"material_revised", "wording_revised"}:
            return (
                event.event_type == CanonicalEventType.REPLACED.value
                and event.from_state
                in {
                    EntryLifecycleState.DRAFT.value,
                    EntryLifecycleState.EVIDENCE_COLLECTED.value,
                    EntryLifecycleState.EDITORIAL_REVIEW.value,
                    EntryLifecycleState.NEEDS_REVIEW.value,
                    EntryLifecycleState.PUBLISHED.value,
                }
                and event.to_state
                in {
                    EntryLifecycleState.DRAFT.value,
                    EntryLifecycleState.EVIDENCE_COLLECTED.value,
                    EntryLifecycleState.EDITORIAL_REVIEW.value,
                    EntryLifecycleState.NEEDS_REVIEW.value,
                    EntryLifecycleState.PUBLISHED.value,
                }
                and EditorialAuthorityService._is_member_identity(event.recorded_by)
            )
        return False

    @staticmethod
    def _release_assurance_snapshot_matches(
        value: object,
        *,
        draft: CreateEditorialEntryRequest,
        entry_identity: str,
    ) -> bool:
        if not isinstance(value, dict) or value.get("schema") != "editorial_release_assurance_snapshot/v1":
            return False
        if value.get("entry_identity") != entry_identity:
            return False
        assurance = draft.release_assurance
        if not isinstance(assurance, dict):
            return False
        expected_identities = {
            "contract_identity": assurance.get("contract_identity"),
            "calibration_identity": assurance.get("calibration_identity"),
            "frozen_acceptance_identity": assurance.get("frozen_acceptance_identity"),
            "named_gate": assurance.get("named_gate"),
        }
        allowed_record_classes = {
            "contract_identity": {
                CanonicalRecordClass.AUTHORITATIVE.value,
                CanonicalRecordClass.IMMUTABLE.value,
            },
            "calibration_identity": {
                CanonicalRecordClass.AUTHORITATIVE.value,
                CanonicalRecordClass.IMMUTABLE.value,
            },
            "frozen_acceptance_identity": {CanonicalRecordClass.IMMUTABLE.value},
            "named_gate": {
                CanonicalRecordClass.AUTHORITATIVE.value,
                CanonicalRecordClass.IMMUTABLE.value,
            },
        }
        records = value.get("records")
        if not isinstance(records, list) or len(records) != len(expected_identities):
            return False
        seen_fields: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                return False
            field = record.get("field")
            if (
                not isinstance(field, str)
                or field in seen_fields
                or field not in expected_identities
                or record.get("identity") != expected_identities[field]
                or record.get("record_class") not in allowed_record_classes[field]
                or not EditorialAuthorityService._is_sha256(record.get("payload_sha256"))
            ):
                return False
            seen_fields.add(field)
        return seen_fields == set(expected_identities) and EditorialAuthorityService._is_event_snapshot(
            value.get("frozen_acceptance_status"),
            to_state=AcceptanceStatus.ACTIVE.value,
        )

    @staticmethod
    def _is_event_snapshot(value: object, *, to_state: str) -> bool:
        return (
            isinstance(value, dict)
            and EditorialAuthorityService._is_canonical_event_id(value.get("event_id"))
            and EditorialAuthorityService._is_sha256(value.get("event_sha256"))
            and value.get("to_state") == to_state
        )

    @staticmethod
    def _is_canonical_event_id(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 32
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _is_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _is_member_identity(value: object) -> bool:
        if not isinstance(value, str):
            return False
        try:
            return StableIdentity.from_stable_id(value).kind is StableIdentityKind.MEMBER
        except ValueError:
            return False

    @staticmethod
    def _raise_export_snapshot_missing(message: str) -> NoReturn:
        raise AppError(
            status_code=409,
            code="EDITORIAL_EXPORT_SNAPSHOT_MISSING",
            message=message,
        )

    @staticmethod
    def _maintainer_acceptance_for_revision(
        events: list[CanonicalEventModel],
        revision_identity: str,
    ) -> dict[str, str] | None:
        for event in reversed(sorted(events, key=lambda item: (EditorialAuthorityService._event_sequence([item]), item.id))):
            payload = EditorialAuthorityService._payload(event)
            if (
                payload.get("schema") != "editorial_authority_event/v1"
                or payload.get("revision_identity") != revision_identity
                or payload.get("action") != "maintainer_responsibility_accepted"
            ):
                continue
            roles = EditorialAuthorityService._roles_for_revision(events, revision_identity)
            if (
                not roles
                or payload.get("roles") != roles
                or event.event_type != CanonicalEventType.STATUS_CHANGED.value
                or event.from_state != event.to_state
                or event.to_state
                not in {
                    EntryLifecycleState.EVIDENCE_COLLECTED.value,
                    EntryLifecycleState.EDITORIAL_REVIEW.value,
                    EntryLifecycleState.NEEDS_REVIEW.value,
                    EntryLifecycleState.PUBLISHED.value,
                }
                or not isinstance(event.recorded_by, str)
                or roles.get("accountable_maintainer_identity") != event.recorded_by
            ):
                continue
            return {
                "status": "accepted",
                "maintainer_identity": event.recorded_by,
                "event_id": event.id,
            }
        return None

    @staticmethod
    def _require_maintainer_acceptance(events: list[CanonicalEventModel], revision_identity: str) -> dict[str, str]:
        acceptance = EditorialAuthorityService._maintainer_acceptance_for_revision(events, revision_identity)
        if acceptance is None:
            raise AppError(
                status_code=409,
                code="EDITORIAL_MAINTAINER_ACCEPTANCE_REQUIRED",
                message="the accountable maintainer must accept responsibility before review, approval, or export",
            )
        return acceptance

    @staticmethod
    def _revision_flag(events: list[CanonicalEventModel], revision_identity: str, flag: str) -> bool:
        return any(
            EditorialAuthorityService._payload(event).get("revision_identity") == revision_identity
            and EditorialAuthorityService._payload(event).get(flag) is True
            for event in events
        )

    @staticmethod
    def _published_revision_has_decisive_source_loss(
        events: list[CanonicalEventModel],
        draft: CreateEditorialEntryRequest,
    ) -> bool:
        source_identities = {
            StableIdentity(StableIdentityKind.SOURCE, source["source_id"]).stable_id
            for source in draft.sources or []
            if isinstance(source, dict) and isinstance(source.get("source_id"), str)
        }
        return bool(source_identities) and any(
            EditorialAuthorityService._payload(event).get("decisive_source_loss") is True
            and EditorialAuthorityService._payload(event).get("source_identity") in source_identities
            for event in events
        )

    @staticmethod
    def _needs_review_at(
        events: list[CanonicalEventModel],
        revision_identity: str,
        current: CanonicalEventModel,
    ) -> datetime | None:
        for event in reversed(sorted(events, key=lambda item: (EditorialAuthorityService._event_sequence([item]), item.id))):
            payload = EditorialAuthorityService._payload(event)
            if payload.get("revision_identity") != revision_identity:
                continue
            candidate = payload.get("needs_review_at")
            if not isinstance(candidate, str):
                continue
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                continue
        if current.to_state == EntryLifecycleState.NEEDS_REVIEW.value:
            return current.occurred_at
        return None

    @staticmethod
    def _export_audit(events: list[CanonicalEventModel], revision_identity: str) -> list[dict[str, Any]]:
        exported: list[dict[str, Any]] = []
        for event in sorted(events, key=lambda item: (EditorialAuthorityService._event_sequence([item]), item.id)):
            payload = EditorialAuthorityService._payload(event)
            if payload.get("revision_identity") != revision_identity:
                continue
            exported.append(
                {
                    "sequence": payload.get("sequence"),
                    "action": payload.get("action"),
                    "event_type": event.event_type,
                    "from_state": event.from_state,
                    "to_state": event.to_state,
                    "recorded_by": event.recorded_by,
                }
            )
        return exported

    @staticmethod
    def _require_state(event: CanonicalEventModel, expected: EntryLifecycleState) -> None:
        if event.to_state != expected.value:
            raise AppError(
                status_code=409,
                code="EDITORIAL_LIFECYCLE_STATE_INVALID",
                message=f"editorial entry must be in {expected.value} state",
                detail={"current_state": event.to_state, "expected_state": expected.value},
            )

    @staticmethod
    def _require_author(payload: dict[str, Any], actor_identity: str, action: str) -> None:
        if payload.get("author_identity") != actor_identity:
            raise AppError(
                status_code=403,
                code="EDITORIAL_AUTHORITY_FORBIDDEN",
                message=f"only the entry author may {action}",
            )

    @staticmethod
    def _raise_for_incomplete_draft(
        payload: CreateEditorialEntryRequest,
        *,
        action: str,
        allow_legacy_claim_linked_contract: bool = False,
    ) -> None:
        reasons = review_validation_reasons(
            payload,
            allow_legacy_claim_linked_contract=allow_legacy_claim_linked_contract,
        )
        if reasons:
            raise AppError(
                status_code=422,
                code="EDITORIAL_ENTRY_INVALID",
                message=f"editorial entry is incomplete for {action}",
                detail={"reasons": reasons},
            )

    @staticmethod
    def _raise_for_secret_findings(value: object) -> None:
        findings = editorial_secret_scan_findings(value)
        if findings:
            raise AppError(
                status_code=422,
                code="EDITORIAL_SECRET_REJECTED",
                message="editorial authority data cannot retain credential-shaped values",
                detail={"findings": findings},
            )

    @staticmethod
    def _payload(record: CanonicalRecordModel | CanonicalEventModel) -> dict[str, Any]:
        return record.payload if isinstance(record.payload, dict) else {}

    @staticmethod
    def _sha256(value: object) -> str:
        return canonical_json_sha256(value)

    @staticmethod
    def _event_snapshot(event: CanonicalEventModel) -> dict[str, str]:
        return {
            "event_id": event.id,
            "event_sha256": EditorialAuthorityService._sha256(
                {
                    "aggregate_id": event.aggregate_id,
                    "aggregate_kind": event.aggregate_kind,
                    "event_type": event.event_type,
                    "from_state": event.from_state,
                    "to_state": event.to_state,
                    "payload": EditorialAuthorityService._payload(event),
                    "occurred_at": event.occurred_at.isoformat(),
                    "recorded_by": event.recorded_by,
                }
            ),
            "to_state": event.to_state,
        }

    @staticmethod
    def _source_definition(source: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in source.items() if key != "availability"}

    @staticmethod
    def _require_editorial_member(actor: User) -> None:
        if not actor.is_active or actor.role == "admin":
            raise AppError(
                status_code=403,
                code="EDITORIAL_AUTHORITY_FORBIDDEN",
                message="the private editorial repository is limited to active editorial members",
            )
