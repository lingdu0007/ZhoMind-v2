from __future__ import annotations

from datetime import UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.contracts.canonical import (
    AcceptanceStatus,
    CanonicalEventType,
    CanonicalRecordClass,
    DeliveryAcceptanceStage,
    StableIdentity,
    StableIdentityKind,
    validate_transition,
)
from app.delivery_acceptance.schemas import (
    AcceptanceBlockingScope,
    AcceptanceChangeClassification,
    AcceptanceCheckInput,
    AcceptanceCheckResult,
    AcceptanceFailureKind,
    AcceptanceStatusFailureInput,
    AcceptanceStatusReason,
    AffectedScopeInput,
    CreateDeliveryAcceptanceRecordRequest,
    UpdateDeliveryAcceptanceStatusRequest,
)
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.user import User
from app.service.identity_audit_service import IdentityAuditService


class DeliveryAcceptanceService:
    """Create immutable acceptance evidence and project its append-only status."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        payload: CreateDeliveryAcceptanceRecordRequest,
        administrator: User,
    ) -> dict[str, Any]:
        related_records = await self._lock_records(await self._create_related_record_ids(payload))
        self._validate_candidate_publication_binding(payload, related_records)
        await self._validate_predecessor(payload, related_records)
        await self._validate_pilot_baseline(payload, related_records)
        await self._validate_carried_forward_checks(payload, related_records)
        self._validate_replacement_reference(payload, related_records)
        evaluator_identity = await IdentityAuditService(self.session).ensure_member_record(
            administrator,
            admission_path="delivery_evaluator",
        )
        identity = StableIdentity.new(StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD)
        record = CanonicalRecordModel(
            stable_id=identity.stable_id,
            identity_kind=identity.kind.value,
            identity_value=identity.value,
            state="created",
            record_class=CanonicalRecordClass.IMMUTABLE.value,
            payload={
                "schema": "delivery_acceptance_record/v1",
                **payload.model_dump(mode="json"),
                "change_owner_identity": evaluator_identity,
                "evaluator_identity": evaluator_identity,
            },
        )
        self.session.add(record)
        await self.session.flush()
        initial_status, initial_reason = self._initial_status(payload)
        await self._append_status(
            record_identity=record.stable_id,
            event_type=CanonicalEventType.CREATED,
            from_status=None,
            to_status=initial_status,
            reason_code=initial_reason,
            actor_identity=evaluator_identity,
        )
        await self.session.commit()
        return await self.get_projection(record.stable_id)

    async def get_projection(self, record_identity: str) -> dict[str, Any]:
        record = await self._get_record(record_identity)
        events = await self._status_events(record.stable_id)
        if not events:
            raise RuntimeError("delivery acceptance record has no status event")
        payload = self._record_payload(record)
        checks = self._validated_persisted_checks(payload)
        blockers = self._current_blockers(events, checks)
        return {
            "record_id": record.stable_id,
            "stage": payload["stage"],
            "change_classification": payload["change_classification"],
            "predecessor_record_identity": payload["predecessor_record_identity"],
            "baseline_record_identity": payload["baseline_record_identity"],
            "is_pilot_entry_baseline": payload["is_pilot_entry_baseline"],
            "replaces_record_identity": payload["replaces_record_identity"],
            "affected_scope": payload["affected_scope"],
            "accepted_scope": self._current_accepted_scope(payload["affected_scope"], events, blockers),
            "content_identities": payload["content_identities"],
            "product_identities": payload["product_identities"],
            "conditions": payload["conditions"],
            "assumptions": payload["assumptions"],
            "checks": [check.model_dump(mode="json") for check in checks],
            "candidate_publication_binding": payload.get("candidate_publication_binding"),
            "change_owner_identity": payload["change_owner_identity"],
            "evaluator_identity": payload["evaluator_identity"],
            "approver_identities": self._approver_identities(events),
            "known_limits": payload["known_limits"],
            "risks": payload["risks"],
            "evidence_links": payload["evidence_links"],
            "reacceptance_triggers": payload["reacceptance_triggers"],
            "blockers": blockers,
            "current_status": events[-1].to_state,
            "status_history": [self._status_projection(event) for event in events],
            "created_at": self._utc_timestamp(record.created_at),
        }

    async def update_status(
        self,
        record_identity: str,
        payload: UpdateDeliveryAcceptanceStatusRequest,
        administrator: User,
    ) -> dict[str, Any]:
        snapshot = await self._get_record(record_identity)
        snapshot_payload = self._record_payload(snapshot)
        related_ids = {record_identity}
        if payload.status is AcceptanceStatus.ACTIVE:
            related_ids.update(await self._ancestor_record_ids(snapshot_payload.get("predecessor_record_identity")))
            baseline_record_identity = snapshot_payload.get("baseline_record_identity")
            if isinstance(baseline_record_identity, str):
                related_ids.add(baseline_record_identity)
        if payload.superseding_record_identity is not None:
            related_ids.add(payload.superseding_record_identity)
        records = await self._lock_records(related_ids)
        record = self._required_record(records, record_identity)
        events = await self._status_events(record.stable_id)
        if not events:
            raise RuntimeError("delivery acceptance record has no status event")
        current_status = AcceptanceStatus(events[-1].to_state)
        try:
            validate_transition(AcceptanceStatus, current_status, payload.status)
        except ValueError as exc:
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_STATUS_TRANSITION_INVALID",
                message="requested acceptance status transition is not allowed",
            ) from exc
        stored_payload = self._record_payload(record)
        stored_checks = self._validated_persisted_checks(stored_payload)
        if payload.status is AcceptanceStatus.ACTIVE:
            if current_status is AcceptanceStatus.SUSPENDED:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_REACCEPTANCE_REQUIRED",
                    message="a suspended record requires a new immutable acceptance record",
                )
            if self._blockers(stored_checks):
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_BLOCKERS_REMAIN",
                    message="a record with failed or required checks cannot become active",
                )
            self._validate_verified_checks(payload, stored_checks)
            await self._validate_active_ancestry(stored_payload, records)
            await self._validate_active_pilot_baseline(stored_payload, records)
        if payload.status_failure is not None:
            self._validate_status_failure(payload.status_failure, stored_payload, stored_checks)
        if payload.reacceptance_trigger is not None:
            if payload.reacceptance_trigger not in stored_payload.get("reacceptance_triggers", []):
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_REACCEPTANCE_TRIGGER_UNKNOWN",
                    message="reacceptance trigger is not declared by the immutable record",
                )
            selected_check_ids = {check.check_id for check in stored_checks}
            if not set(payload.invalidated_check_ids).issubset(selected_check_ids):
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_REACCEPTANCE_CHECK_UNKNOWN",
                    message="reacceptance invalidation must name selected checks",
                )
        if payload.status is AcceptanceStatus.SUPERSEDED:
            assert payload.superseding_record_identity is not None
            successor = self._required_record(records, payload.superseding_record_identity)
            if successor.stable_id == record.stable_id:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_SELF_SUPERSESSION",
                    message="a delivery acceptance record cannot supersede itself",
                )
            successor_events = await self._status_events(successor.stable_id)
            if not successor_events or successor_events[-1].to_state != AcceptanceStatus.ACTIVE.value:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_SUCCESSOR_NOT_ACTIVE",
                    message="a superseding record must already be active",
                )
            successor_payload = self._record_payload(successor)
            if successor_payload.get("stage") != stored_payload.get("stage") or (
                successor_payload.get("change_classification") != stored_payload.get("change_classification")
            ):
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_SUCCESSOR_STAGE_INVALID",
                    message="a replacement must reach the same acceptance stage and classification",
                )
            if successor_payload.get("replaces_record_identity") != record.stable_id:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_SUCCESSOR_LINEAGE_INVALID",
                    message="a superseding record must explicitly replace this record",
                )
            if successor.created_at <= record.created_at:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_SUCCESSOR_LINEAGE_INVALID",
                    message="a superseding record must be created after the record it replaces",
                )
            self._ensure_compatible_replacement_scope(
                source_payload=stored_payload,
                target_payload=successor_payload,
                error_code="ACCEPTANCE_SUCCESSOR_SCOPE_MISMATCH",
            )
        actor_identity = await IdentityAuditService(self.session).ensure_member_record(
            administrator,
            admission_path="delivery_evaluator",
        )
        await self._append_status(
            record_identity=record.stable_id,
            event_type=(
                CanonicalEventType.SUPERSEDED
                if payload.status is AcceptanceStatus.SUPERSEDED
                else CanonicalEventType.STATUS_CHANGED
            ),
            from_status=current_status,
            to_status=payload.status,
            reason_code=payload.reason_code,
            actor_identity=actor_identity,
            superseding_record_identity=payload.superseding_record_identity,
            verified_checks=[check.model_dump(mode="json") for check in payload.verified_checks],
            status_failure=payload.status_failure.model_dump(mode="json") if payload.status_failure is not None else None,
            reacceptance_trigger=payload.reacceptance_trigger,
            invalidated_check_ids=payload.invalidated_check_ids,
        )
        await self.session.commit()
        return await self.get_projection(record.stable_id)

    async def _create_related_record_ids(self, payload: CreateDeliveryAcceptanceRecordRequest) -> set[str]:
        related_ids = await self._ancestor_record_ids(payload.predecessor_record_identity)
        if payload.baseline_record_identity is not None:
            related_ids.add(payload.baseline_record_identity)
        if payload.replaces_record_identity is not None:
            related_ids.add(payload.replaces_record_identity)
        related_ids.update(
            check.carried_forward_from
            for check in payload.checks
            if check.carried_forward_from is not None
        )
        if payload.candidate_publication_binding is not None:
            related_ids.update(payload.candidate_publication_binding.persisted_record_identities)
        related_ids.update(await self._referenced_published_knowledge_version_ids(payload))
        return related_ids

    def _validate_candidate_publication_binding(
        self,
        payload: CreateDeliveryAcceptanceRecordRequest,
        records: dict[str, CanonicalRecordModel],
    ) -> None:
        binding = payload.candidate_publication_binding
        if binding is None:
            if any(
                record.identity_kind == StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION.value
                and isinstance(record.payload, dict)
                and record.payload.get("schema") == "published_knowledge_version/v1"
                for record in records.values()
            ):
                raise self._candidate_publication_binding_error()
            return

        candidate = self._bound_record(
            records,
            binding.candidate_identity,
            identity_kind=StableIdentityKind.CANDIDATE,
            schema="candidate_build_candidate/v1",
        )
        inspection = self._bound_record(
            records,
            binding.inspection_record_identity,
            identity_kind=StableIdentityKind.EVENT,
            schema="candidate_inspection/v1",
        )
        acceptance = self._bound_record(
            records,
            binding.acceptance_record_identity,
            identity_kind=StableIdentityKind.EVENT,
            schema="candidate_acceptance/v1",
        )
        published = self._bound_record(
            records,
            binding.published_knowledge_version_identity,
            identity_kind=StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION,
            schema="published_knowledge_version/v1",
        )
        configuration = self._bound_record(
            records,
            binding.configuration_identity,
            identity_kind=StableIdentityKind.CONFIGURATION,
            schema="candidate_publication_configuration/v1",
        )

        candidate_payload = self._record_payload(candidate)
        inspection_payload = self._record_payload(inspection)
        acceptance_payload = self._record_payload(acceptance)
        published_payload = self._record_payload(published)
        configuration_payload = self._record_payload(configuration)
        expected = {
            "candidate_id": binding.candidate_identity,
            "entry_identity": binding.entry_identity,
            "bundle_sha256": binding.bundle_sha256,
            "frozen_input_sha256": binding.frozen_input_sha256,
            "configuration_identity": binding.configuration_identity,
        }
        publication_records = (inspection_payload, acceptance_payload, published_payload)
        if (
            candidate_payload.get("entry_identity") != binding.entry_identity
            or candidate_payload.get("bundle_sha256") != binding.bundle_sha256
            or candidate_payload.get("frozen_input_sha256") != binding.frozen_input_sha256
            or candidate_payload.get("embedding_configuration") != configuration_payload.get("configuration")
            or inspection_payload.get("inspection_record_identity") != inspection.stable_id
            or acceptance_payload.get("acceptance_record_identity") != acceptance.stable_id
            or acceptance_payload.get("inspection_record_identity") != inspection.stable_id
            or published_payload.get("inspection_record_identity") != inspection.stable_id
            or published_payload.get("acceptance_record_identity") != acceptance.stable_id
            or any(
                record_payload.get(key) != value
                for record_payload in publication_records
                for key, value in expected.items()
            )
            or published_payload.get("candidate_id") != binding.candidate_identity
        ):
            raise self._candidate_publication_binding_error()

    async def _referenced_published_knowledge_version_ids(
        self,
        payload: CreateDeliveryAcceptanceRecordRequest,
    ) -> set[str]:
        declared_identities = {
            *payload.affected_scope.entry_identities,
            *payload.affected_scope.collection_identities,
            *payload.affected_scope.product_path_identities,
            *payload.affected_scope.configuration_identities,
            *payload.affected_scope.protected_capability_identities,
            *payload.affected_scope.public_claim_identities,
            *payload.content_identities,
            *payload.product_identities,
        }
        if payload.affected_scope.deployment_identity is not None:
            declared_identities.add(payload.affected_scope.deployment_identity)
        published_version_identities = {
            identity for identity in declared_identities if identity.startswith("published_knowledge_version:")
        }
        if not published_version_identities:
            return set()
        result = await self.session.execute(
            select(CanonicalRecordModel.stable_id).where(
                CanonicalRecordModel.stable_id.in_(published_version_identities),
                CanonicalRecordModel.identity_kind == StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION.value,
            )
        )
        return set(result.scalars())

    @staticmethod
    def _bound_record(
        records: dict[str, CanonicalRecordModel],
        record_identity: str,
        *,
        identity_kind: StableIdentityKind,
        schema: str,
    ) -> CanonicalRecordModel:
        record = records.get(record_identity)
        if (
            record is None
            or record.identity_kind != identity_kind.value
            or not isinstance(record.payload, dict)
            or record.payload.get("schema") != schema
        ):
            raise DeliveryAcceptanceService._candidate_publication_binding_error()
        return record

    @staticmethod
    def _candidate_publication_binding_error() -> AppError:
        return AppError(
            status_code=409,
            code="ACCEPTANCE_CANDIDATE_PUBLICATION_BINDING_INVALID",
            message="delivery acceptance cannot prove its exact Candidate publication binding",
        )

    async def _ancestor_record_ids(self, initial_identity: object) -> set[str]:
        if initial_identity is None:
            return set()
        if not isinstance(initial_identity, str):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_RECORD_INVALID",
                message="delivery acceptance record has an invalid predecessor chain",
            )
        identities: set[str] = set()
        current_identity: str | None = initial_identity
        while current_identity is not None:
            if current_identity in identities:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_RECORD_INVALID",
                    message="delivery acceptance record has a cyclic predecessor chain",
                )
            identities.add(current_identity)
            record = await self._get_record(current_identity)
            current_identity = self._record_payload(record).get("predecessor_record_identity")
            if current_identity is not None and not isinstance(current_identity, str):
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_RECORD_INVALID",
                    message="delivery acceptance record has an invalid predecessor chain",
                )
        return identities

    async def _validate_predecessor(
        self,
        payload: CreateDeliveryAcceptanceRecordRequest,
        records: dict[str, CanonicalRecordModel],
    ) -> None:
        if payload.stage is DeliveryAcceptanceStage.LOCAL_DEVELOPMENT:
            return
        assert payload.predecessor_record_identity is not None
        predecessor = self._required_record(records, payload.predecessor_record_identity)
        predecessor_payload = self._record_payload(predecessor)
        try:
            predecessor_stage = DeliveryAcceptanceStage(predecessor_payload["stage"])
            validate_transition(DeliveryAcceptanceStage, predecessor_stage, payload.stage)
        except (KeyError, ValueError) as exc:
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_PREDECESSOR_STAGE_INVALID",
                message="predecessor record cannot promote to the requested stage",
            ) from exc
        if (
            payload.change_classification is AcceptanceChangeClassification.PROTECTED_PRODUCT_PATH
            and payload.stage is DeliveryAcceptanceStage.DAILY_USE_RELEASE
            and predecessor_stage is not DeliveryAcceptanceStage.LIMITED_TEAM_PILOT
        ):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_PROTECTED_PATH_INVALID",
                message="protected changes require Limited-Team Pilot before Daily-Use restoration",
            )
        predecessor_events = await self._status_events(predecessor.stable_id)
        if not predecessor_events or predecessor_events[-1].to_state != AcceptanceStatus.ACTIVE.value:
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_PREDECESSOR_NOT_ACTIVE",
                message="later stage acceptance requires an active predecessor record",
            )
        self._ensure_same_release_scope(
            source_payload=predecessor_payload,
            target_payload={
                "stage": payload.stage.value,
                "baseline_record_identity": payload.baseline_record_identity,
                "is_pilot_entry_baseline": payload.is_pilot_entry_baseline,
                "affected_scope": payload.affected_scope.model_dump(mode="json"),
                "content_identities": payload.content_identities,
                "product_identities": payload.product_identities,
            },
            error_code="ACCEPTANCE_PREDECESSOR_SCOPE_MISMATCH",
        )

    async def _validate_pilot_baseline(
        self,
        payload: CreateDeliveryAcceptanceRecordRequest,
        records: dict[str, CanonicalRecordModel],
    ) -> None:
        if payload.baseline_record_identity is None:
            return
        await self._validate_active_pilot_baseline(
            {
                "baseline_record_identity": payload.baseline_record_identity,
                "affected_scope": payload.affected_scope.model_dump(mode="json"),
                "product_identities": payload.product_identities,
            },
            records,
        )

    async def _validate_active_pilot_baseline(
        self,
        payload: dict[str, Any],
        records: dict[str, CanonicalRecordModel],
    ) -> None:
        baseline_record_identity = payload.get("baseline_record_identity")
        if baseline_record_identity is None:
            return
        if not isinstance(baseline_record_identity, str):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_RECORD_INVALID",
                message="delivery acceptance record has an invalid Pilot baseline reference",
            )
        baseline = self._required_record(records, baseline_record_identity)
        baseline_payload = self._record_payload(baseline)
        if (
            baseline_payload.get("stage") != DeliveryAcceptanceStage.LIMITED_TEAM_PILOT.value
            or baseline_payload.get("is_pilot_entry_baseline") is not True
        ):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_BASELINE_KIND_INVALID",
                message="a Pilot baseline reference must name a declared Pilot Entry Baseline",
            )
        baseline_events = await self._status_events(baseline.stable_id)
        if not baseline_events or baseline_events[-1].to_state != AcceptanceStatus.ACTIVE.value:
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_BASELINE_NOT_ACTIVE",
                message="a Pilot baseline reference must remain active",
            )
        affected_scope = payload.get("affected_scope")
        baseline_scope = baseline_payload.get("affected_scope")
        if not isinstance(affected_scope, dict) or not isinstance(baseline_scope, dict):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_RECORD_INVALID",
                message="delivery acceptance record cannot prove its Pilot baseline scope",
            )
        if affected_scope.get("deployment_identity") != baseline_scope.get("deployment_identity"):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_BASELINE_SCOPE_MISMATCH",
                message="a Pilot baseline must bind the same exact target deployment",
            )
        if payload.get("product_identities") != baseline_payload.get("product_identities"):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_BASELINE_SCOPE_MISMATCH",
                message="a Pilot baseline must bind the same exact product identities",
            )

    async def _validate_carried_forward_checks(
        self,
        payload: CreateDeliveryAcceptanceRecordRequest,
        records: dict[str, CanonicalRecordModel],
    ) -> None:
        ancestors = await self._ancestor_record_ids(payload.predecessor_record_identity)
        for check in payload.checks:
            if check.result is not AcceptanceCheckResult.CARRIED_FORWARD:
                continue
            assert check.carried_forward_from is not None
            if check.carried_forward_from not in ancestors:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_CARRY_FORWARD_INVALID",
                    message="carried-forward evidence must come from the predecessor lineage",
                )
            source = self._required_record(records, check.carried_forward_from)
            source_events = await self._status_events(source.stable_id)
            if not source_events or source_events[-1].to_state != AcceptanceStatus.ACTIVE.value:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_CARRY_FORWARD_INVALID",
                    message="carried-forward evidence requires an active source record",
                )
            source_check = next(
                (
                    candidate
                    for candidate in self._validated_persisted_checks(self._record_payload(source))
                    if candidate.check_id == check.check_id
                    and candidate.identity_dependencies == check.identity_dependencies
                    and candidate.applicability_conditions == check.applicability_conditions
                    and candidate.assumptions == check.assumptions
                    and candidate.evidence_links == check.evidence_links
                    and candidate.result in {AcceptanceCheckResult.PASSED, AcceptanceCheckResult.CARRIED_FORWARD}
                ),
                None,
            )
            if source_check is None:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_CARRY_FORWARD_INVALID",
                    message="carried-forward evidence no longer matches its check dependencies",
                )

    def _validate_replacement_reference(
        self,
        payload: CreateDeliveryAcceptanceRecordRequest,
        records: dict[str, CanonicalRecordModel],
    ) -> None:
        if payload.replaces_record_identity is None:
            return
        replaced = self._required_record(records, payload.replaces_record_identity)
        self._ensure_compatible_replacement_scope(
            source_payload=self._record_payload(replaced),
            target_payload={"affected_scope": payload.affected_scope.model_dump(mode="json")},
            error_code="ACCEPTANCE_REPLACEMENT_SCOPE_MISMATCH",
        )

    async def _get_record(self, record_identity: str) -> CanonicalRecordModel:
        record = await self.session.get(CanonicalRecordModel, record_identity)
        if record is None or record.identity_kind != StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="delivery acceptance record not found")
        return record

    async def _lock_records(self, record_identities: set[str]) -> dict[str, CanonicalRecordModel]:
        if not record_identities:
            return {}
        result = await self.session.execute(
            select(CanonicalRecordModel)
            .where(CanonicalRecordModel.stable_id.in_(record_identities))
            .order_by(CanonicalRecordModel.stable_id.asc())
            .with_for_update()
        )
        records = {record.stable_id: record for record in result.scalars()}
        for record_identity in record_identities:
            if record_identity not in records:
                raise AppError(
                    status_code=404,
                    code="RESOURCE_NOT_FOUND",
                    message="canonical record not found",
                    detail={"record_identity": record_identity},
                )
        return records

    async def _status_events(self, record_identity: str) -> list[CanonicalEventModel]:
        result = await self.session.execute(
            select(CanonicalEventModel)
            .where(
                CanonicalEventModel.aggregate_id == record_identity,
                CanonicalEventModel.aggregate_kind == StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value,
            )
            .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
        )
        return list(result.scalars())

    async def _append_status(
        self,
        *,
        record_identity: str,
        event_type: CanonicalEventType,
        from_status: AcceptanceStatus | None,
        to_status: AcceptanceStatus,
        reason_code: AcceptanceStatusReason,
        actor_identity: str,
        superseding_record_identity: str | None = None,
        verified_checks: list[dict[str, Any]] | None = None,
        status_failure: dict[str, Any] | None = None,
        reacceptance_trigger: str | None = None,
        invalidated_check_ids: list[str] | None = None,
    ) -> None:
        self.session.add(
            CanonicalEventModel(
                aggregate_id=record_identity,
                aggregate_kind=StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value,
                event_type=event_type.value,
                from_state=from_status.value if from_status is not None else None,
                to_state=to_status.value,
                payload={
                    "schema": "delivery_acceptance_status/v1",
                    "reason_code": reason_code.value,
                    "superseding_record_identity": superseding_record_identity,
                    "verified_checks": verified_checks or [],
                    "status_failure": status_failure,
                    "reacceptance_trigger": reacceptance_trigger,
                    "invalidated_check_ids": invalidated_check_ids or [],
                },
                recorded_by=actor_identity,
            )
        )
        await self.session.flush()

    @staticmethod
    def _record_payload(record: CanonicalRecordModel) -> dict[str, Any]:
        if not isinstance(record.payload, dict):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_RECORD_INVALID",
                message="delivery acceptance record cannot prove its immutable payload",
            )
        return record.payload

    @staticmethod
    def _required_record(
        records: dict[str, CanonicalRecordModel],
        record_identity: str,
    ) -> CanonicalRecordModel:
        record = records.get(record_identity)
        if record is None or record.identity_kind != StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="delivery acceptance record not found")
        return record

    @classmethod
    def _ensure_same_release_scope(
        cls,
        *,
        source_payload: dict[str, Any],
        target_payload: dict[str, Any],
        error_code: str,
    ) -> None:
        source_scope = source_payload.get("affected_scope")
        target_scope = target_payload.get("affected_scope")
        if not isinstance(source_scope, dict) or not isinstance(target_scope, dict):
            raise AppError(
                status_code=409,
                code=error_code,
                message="stage acceptance records must prove the same affected scope",
            )
        public_claim_expansion = {
            source_payload.get("stage"),
            target_payload.get("stage"),
        } == {
            DeliveryAcceptanceStage.PUBLIC_EVIDENCE_RELEASE.value,
            DeliveryAcceptanceStage.DAILY_USE_RELEASE.value,
        }
        if public_claim_expansion:
            source_scope = cls._without_public_claim_scope(source_scope)
            target_scope = cls._without_public_claim_scope(target_scope)
        if (
            source_scope != target_scope
            or source_payload.get("content_identities") != target_payload.get("content_identities")
            or not cls._same_product_scope(source_payload, target_payload)
        ):
            raise AppError(
                status_code=409,
                code=error_code,
                message="stage acceptance records must bind the same affected scope and exact identities",
            )

    @classmethod
    def _same_product_scope(cls, source_payload: dict[str, Any], target_payload: dict[str, Any]) -> bool:
        source_product_identities = source_payload.get("product_identities")
        target_product_identities = target_payload.get("product_identities")
        if source_product_identities == target_product_identities:
            return True
        return cls._is_pilot_scope_extension(source_payload, target_payload) or cls._is_pilot_scope_extension(
            target_payload,
            source_payload,
        )

    @staticmethod
    def _is_pilot_scope_extension(base_payload: dict[str, Any], extended_payload: dict[str, Any]) -> bool:
        if not (
            extended_payload.get("is_pilot_entry_baseline") is True
            or isinstance(extended_payload.get("baseline_record_identity"), str)
        ):
            return False
        base_identities = base_payload.get("product_identities")
        extended_identities = extended_payload.get("product_identities")
        if not isinstance(base_identities, list) or not isinstance(extended_identities, list):
            return False
        if not all(isinstance(identity, str) for identity in [*base_identities, *extended_identities]):
            return False
        added_identities = set(extended_identities) - set(base_identities)
        allowed_kinds = {"user_boundary", "data_boundary", "host", "corpus", "concurrency"}
        return bool(added_identities) and set(base_identities).issubset(extended_identities) and all(
            identity.partition(":")[0] in allowed_kinds for identity in added_identities
        )

    @staticmethod
    def _without_public_claim_scope(affected_scope: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in affected_scope.items()
            if key != "public_claim_identities"
        }

    @staticmethod
    def _ensure_compatible_replacement_scope(
        *,
        source_payload: dict[str, Any],
        target_payload: dict[str, Any],
        error_code: str,
    ) -> None:
        source_scope = source_payload.get("affected_scope")
        target_scope = target_payload.get("affected_scope")
        if not isinstance(source_scope, dict) or not isinstance(target_scope, dict):
            raise AppError(
                status_code=409,
                code=error_code,
                message="replacement records must prove compatible affected scope",
            )
        logical_fields = (
            "entry_identities",
            "collection_identities",
            "product_path_identities",
            "protected_capability_identities",
            "public_claim_identities",
            "deployment_identity",
        )
        if any(source_scope.get(field) != target_scope.get(field) for field in logical_fields):
            raise AppError(
                status_code=409,
                code=error_code,
                message="replacement records must bind the same logical affected scope",
            )

    @staticmethod
    def _validate_verified_checks(
        payload: UpdateDeliveryAcceptanceStatusRequest,
        checks: list[AcceptanceCheckInput],
    ) -> None:
        expected = {
            check.check_id
            for check in checks
            if check.result in {AcceptanceCheckResult.PASSED, AcceptanceCheckResult.CARRIED_FORWARD}
        }
        actual = [check.check_id for check in payload.verified_checks]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_VERIFICATION_INCOMPLETE",
                message="active status must bind verification evidence for every passing check",
            )

    @staticmethod
    def _validate_status_failure(
        failure: AcceptanceStatusFailureInput,
        stored_payload: dict[str, Any],
        checks: list[AcceptanceCheckInput],
    ) -> None:
        if failure.check_id not in {check.check_id for check in checks}:
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_FAILURE_CHECK_UNKNOWN",
                message="status failure must refer to a selected check",
            )
        try:
            affected_scope = AffectedScopeInput.model_validate(stored_payload["affected_scope"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_RECORD_INVALID",
                message="delivery acceptance record cannot prove its affected scope",
            ) from exc
        if not affected_scope.contains(failure.blocking_scope.scope, failure.blocking_scope.identity):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_FAILURE_SCOPE_INVALID",
                message="status failure scope must be part of the immutable affected scope",
            )

    async def _validate_active_ancestry(
        self,
        payload: dict[str, Any],
        records: dict[str, CanonicalRecordModel],
    ) -> None:
        current_payload = payload
        seen: set[str] = set()
        while True:
            predecessor_identity = current_payload.get("predecessor_record_identity")
            if predecessor_identity is None:
                return
            if not isinstance(predecessor_identity, str) or predecessor_identity in seen:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_RECORD_INVALID",
                    message="delivery acceptance record has an invalid predecessor chain",
                )
            seen.add(predecessor_identity)
            predecessor = self._required_record(records, predecessor_identity)
            predecessor_payload = self._record_payload(predecessor)
            predecessor_events = await self._status_events(predecessor.stable_id)
            if not predecessor_events or predecessor_events[-1].to_state != AcceptanceStatus.ACTIVE.value:
                raise AppError(
                    status_code=409,
                    code="ACCEPTANCE_PREDECESSOR_NOT_ACTIVE",
                    message="activation requires every predecessor record to remain active",
                )
            self._ensure_same_release_scope(
                source_payload=current_payload,
                target_payload=predecessor_payload,
                error_code="ACCEPTANCE_PREDECESSOR_SCOPE_MISMATCH",
            )
            current_payload = predecessor_payload

    @staticmethod
    def _initial_status(
        payload: CreateDeliveryAcceptanceRecordRequest,
    ) -> tuple[AcceptanceStatus, AcceptanceStatusReason]:
        failed_kinds = {
            check.failure_kind
            for check in payload.checks
            if check.result is AcceptanceCheckResult.FAILED and check.failure_kind is not None
        }
        deployment_failures = failed_kinds - {
            AcceptanceFailureKind.ENTRY_SPECIFIC,
            AcceptanceFailureKind.COLLECTION_RETRIEVAL,
            AcceptanceFailureKind.COLLECTION_CHUNKING,
            AcceptanceFailureKind.COLLECTION_IMPORT,
            AcceptanceFailureKind.COLLECTION_BUNDLE,
            AcceptanceFailureKind.COLLECTION_CROSS_ENTRY,
            AcceptanceFailureKind.PERFORMANCE,
        }
        if deployment_failures:
            return AcceptanceStatus.SUSPENDED, AcceptanceStatusReason.RECORD_CREATED_KNOWN_FAILURE
        if failed_kinds:
            if AcceptanceFailureKind.PERFORMANCE in failed_kinds:
                return AcceptanceStatus.AT_RISK, AcceptanceStatusReason.PERFORMANCE_OBJECTIVE_MISSED
            return AcceptanceStatus.AT_RISK, AcceptanceStatusReason.RECORD_CREATED_KNOWN_FAILURE
        return AcceptanceStatus.AT_RISK, AcceptanceStatusReason.RECORD_CREATED_PENDING_VERIFICATION

    @staticmethod
    def _validated_persisted_checks(payload: dict[str, Any]) -> list[AcceptanceCheckInput]:
        raw_checks = payload.get("checks")
        if not isinstance(raw_checks, list):
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_RECORD_INVALID",
                message="delivery acceptance record cannot prove its check state",
            )
        try:
            return [AcceptanceCheckInput.model_validate(check) for check in raw_checks]
        except (TypeError, ValueError) as exc:
            raise AppError(
                status_code=409,
                code="ACCEPTANCE_RECORD_INVALID",
                message="delivery acceptance record contains invalid checks",
            ) from exc

    @staticmethod
    def _blockers(checks: list[AcceptanceCheckInput]) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        for check in checks:
            if check.result not in {AcceptanceCheckResult.FAILED, AcceptanceCheckResult.REQUIRED}:
                continue
            blocker: dict[str, Any] = {
                "check_id": check.check_id,
                "result": check.result.value,
            }
            if check.reason is not None:
                blocker["reason"] = check.reason
            if check.failure_kind is not None:
                blocker["failure_kind"] = check.failure_kind.value
            if check.blocking_scope is not None:
                blocker["blocking_scope"] = check.blocking_scope.model_dump(mode="json")
            if check.performance_objective_identity is not None:
                blocker["performance_objective_identity"] = check.performance_objective_identity
            blockers.append(blocker)
        return blockers

    @classmethod
    def _current_blockers(
        cls,
        events: list[CanonicalEventModel],
        checks: list[AcceptanceCheckInput],
    ) -> list[dict[str, Any]]:
        blockers = cls._blockers(checks)
        current_event = events[-1]
        payload = current_event.payload if isinstance(current_event.payload, dict) else {}
        if current_event.to_state in {AcceptanceStatus.AT_RISK.value, AcceptanceStatus.SUSPENDED.value}:
            status_failure = payload.get("status_failure")
            if isinstance(status_failure, dict):
                try:
                    failure = AcceptanceStatusFailureInput.model_validate(status_failure)
                except ValueError as exc:
                    raise AppError(
                        status_code=409,
                        code="ACCEPTANCE_RECORD_INVALID",
                        message="acceptance status event contains an invalid failure",
                    ) from exc
                blockers.append(
                    {
                        "check_id": failure.check_id,
                        "result": "failed",
                        "reason": failure.reason,
                        "failure_kind": failure.failure_kind.value,
                        "blocking_scope": failure.blocking_scope.model_dump(mode="json"),
                        **(
                            {"performance_objective_identity": failure.performance_objective_identity}
                            if failure.performance_objective_identity is not None
                            else {}
                        ),
                    }
                )
            elif payload.get("reason_code") == AcceptanceStatusReason.REACCEPTANCE_DUE.value:
                blockers.append(
                    {
                        "check_id": "check:reacceptance-trigger",
                        "result": "required",
                        "reacceptance_trigger": payload.get("reacceptance_trigger"),
                    }
                )
        return blockers

    @classmethod
    def _current_accepted_scope(
        cls,
        affected_scope: dict[str, Any],
        events: list[CanonicalEventModel],
        blockers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not any(event.to_state == AcceptanceStatus.ACTIVE.value for event in events):
            return cls._empty_scope(affected_scope)
        if events[-1].to_state == AcceptanceStatus.SUPERSEDED.value:
            return cls._empty_scope(affected_scope)
        return cls._accepted_scope(affected_scope, blockers)

    @staticmethod
    def _empty_scope(affected_scope: dict[str, Any]) -> dict[str, Any]:
        empty_scope = {
            key: list(value) if isinstance(value, list) else value
            for key, value in affected_scope.items()
        }
        for field in (
            "entry_identities",
            "collection_identities",
            "product_path_identities",
            "configuration_identities",
            "protected_capability_identities",
            "public_claim_identities",
        ):
            empty_scope[field] = []
        empty_scope["deployment_identity"] = None
        return empty_scope

    @staticmethod
    def _accepted_scope(affected_scope: dict[str, Any], blockers: list[dict[str, Any]]) -> dict[str, Any]:
        accepted_scope = {
            key: list(value) if isinstance(value, list) else value
            for key, value in affected_scope.items()
        }
        blocked_entries = {
            str(blocker["blocking_scope"]["identity"])
            for blocker in blockers
            if isinstance(blocker.get("blocking_scope"), dict)
            and blocker["blocking_scope"].get("scope") == AcceptanceBlockingScope.ENTRY_VERSION.value
            and isinstance(blocker["blocking_scope"].get("identity"), str)
        }
        if blocked_entries and isinstance(accepted_scope.get("entry_identities"), list):
            accepted_scope["entry_identities"] = [
                entry for entry in accepted_scope["entry_identities"] if entry not in blocked_entries
            ]
        blocked_collections = {
            str(blocker["blocking_scope"]["identity"])
            for blocker in blockers
            if isinstance(blocker.get("blocking_scope"), dict)
            and blocker["blocking_scope"].get("scope") == AcceptanceBlockingScope.COLLECTION.value
            and isinstance(blocker["blocking_scope"].get("identity"), str)
        }
        if blocked_collections and isinstance(accepted_scope.get("collection_identities"), list):
            accepted_scope["collection_identities"] = [
                collection for collection in accepted_scope["collection_identities"] if collection not in blocked_collections
            ]
        blocked_public_claims = {
            str(blocker["blocking_scope"]["identity"])
            for blocker in blockers
            if isinstance(blocker.get("blocking_scope"), dict)
            and blocker["blocking_scope"].get("scope") == AcceptanceBlockingScope.PUBLIC_CLAIM.value
            and isinstance(blocker["blocking_scope"].get("identity"), str)
        }
        if blocked_public_claims and isinstance(accepted_scope.get("public_claim_identities"), list):
            accepted_scope["public_claim_identities"] = [
                identity
                for identity in accepted_scope["public_claim_identities"]
                if identity not in blocked_public_claims
            ]
        deployment_block = any(
            isinstance(blocker.get("blocking_scope"), dict)
            and blocker["blocking_scope"].get("scope") == AcceptanceBlockingScope.DEPLOYMENT.value
            and blocker.get("failure_kind") != AcceptanceFailureKind.PERFORMANCE.value
            for blocker in blockers
        )
        if deployment_block:
            for field in (
                "entry_identities",
                "collection_identities",
                "product_path_identities",
                "configuration_identities",
                "protected_capability_identities",
                "public_claim_identities",
            ):
                accepted_scope[field] = []
            accepted_scope["deployment_identity"] = None
        return accepted_scope

    @staticmethod
    def _approver_identities(events: list[CanonicalEventModel]) -> list[str]:
        return [
            event.recorded_by
            for event in events
            if event.to_state == AcceptanceStatus.ACTIVE.value
            and isinstance(event.payload, dict)
            and event.payload.get("reason_code") == AcceptanceStatusReason.CHECKS_VERIFIED.value
            and event.recorded_by is not None
        ]

    @classmethod
    def _status_projection(cls, event: CanonicalEventModel) -> dict[str, Any]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        projection: dict[str, Any] = {
            "event_id": event.id,
            "occurred_at": cls._utc_timestamp(event.occurred_at).isoformat(),
            "from_status": event.from_state,
            "status": event.to_state,
            "reason_code": str(payload.get("reason_code", "unknown")),
            "event_type": event.event_type,
        }
        superseding_record_identity = payload.get("superseding_record_identity")
        if isinstance(superseding_record_identity, str):
            projection["superseding_record_identity"] = superseding_record_identity
        if event.recorded_by is not None:
            projection["recorded_by"] = event.recorded_by
        status_failure = payload.get("status_failure")
        if isinstance(status_failure, dict):
            projection["status_failure"] = status_failure
        reacceptance_trigger = payload.get("reacceptance_trigger")
        if isinstance(reacceptance_trigger, str):
            projection["reacceptance_trigger"] = reacceptance_trigger
        invalidated_check_ids = payload.get("invalidated_check_ids")
        if isinstance(invalidated_check_ids, list) and invalidated_check_ids:
            projection["invalidated_check_ids"] = invalidated_check_ids
        verified_checks = payload.get("verified_checks")
        if isinstance(verified_checks, list) and verified_checks:
            projection["verified_checks"] = verified_checks
        return projection

    @staticmethod
    def _utc_timestamp(value):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
