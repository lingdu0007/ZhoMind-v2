from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app.common.exceptions import AppError
from app.contracts.canonical import (
    AcceptanceStatus,
    CanonicalEventType,
    CanonicalRecordClass,
    StableIdentityKind,
)
from app.delivery_acceptance.schemas import CreateDeliveryAcceptanceRecordRequest
from app.editorial_authority.schemas import (
    CreateEditorialEntryRequest,
    ReviseEditorialEntryRequest,
    editorial_export_safety_findings,
    editorial_secret_scan_findings,
    lightweight_revision_reasons,
    review_validation_reasons,
)
from app.editorial_authority.service import EditorialAuthorityService, evaluate_answer_eligibility
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.document import Document
from app.model.user import User
from app.reviewed_bundles.verifier import CanonicalEditorialExportVerifier


async def test_author_can_save_private_draft_but_incomplete_entry_cannot_collect_evidence(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    db_session.add(author)
    await db_session.commit()

    authority = EditorialAuthorityService(db_session)
    created = await authority.create_draft(
        CreateEditorialEntryRequest(entry_id="rag-source-admission-001", title="Choose a source admission boundary"),
        author,
    )

    assert created["entry_id"] == "rag-source-admission-001"
    assert created["lifecycle_state"] == "draft"
    assert created["author_identity"].startswith("member:")
    assert created["answer_eligible"] is False

    with pytest.raises(AppError) as exc_info:
        await authority.collect_evidence(created["entry_id"], author)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "EDITORIAL_ENTRY_INVALID"
    reasons = exc_info.value.detail["reasons"]
    assert {reason["field"] for reason in reasons} >= {
        "coverage_position",
        "assurance_level",
        "approving_reviewer_username",
        "accountable_maintainer_username",
        "sources",
        "acceptance_material",
        "body",
    }


def _review_ready_entry(
    *,
    entry_id: str = "rag-source-admission-001",
    assurance_level: str = "source_grounded",
) -> CreateEditorialEntryRequest:
    return CreateEditorialEntryRequest(
        entry_id=entry_id,
        title="Choose a source admission boundary",
        coverage_position="rag_source_admission_and_chunking",
        assurance_level=assurance_level,
        approving_reviewer_username="reviewer",
        accountable_maintainer_username="maintainer",
        review_date="2026-09-05",
        applicable_versions=["rag-runtime-v1"],
        applicability_conditions=[
            {
                "condition_id": "condition-production",
                "field": "deployment",
                "operator": "equals",
                "value": "production",
            }
        ],
        non_applicability_conditions=[
            {
                "condition_id": "condition-untrusted",
                "field": "data_scope",
                "operator": "equals",
                "value": "untrusted-public-corpus",
            }
        ],
        freshness_triggers=[
            {
                "trigger_id": "freshness-source-release",
                "trigger_type": "source_release",
                "review_within_days": 7,
            }
        ],
        sources=[
            {
                "source_id": "source-rag-admission-001",
                "source_tier": "primary_evidence_source",
                "title": "RAG source admission design record",
                "authority": "ZhoMind architecture group",
                "version_or_date": "2026-09-05",
                "availability": "verified_usable",
                "access_scope": "public",
                "public_url": "https://example.com/rag/source-admission",
                "independent_public_verifiability": True,
            }
        ],
        chunk_strategy={
            "strategy_id": "section-aware-900-120",
            "max_characters": 900,
            "overlap_characters": 120,
            "preserve_section_boundaries": True,
        },
        acceptance_material={
            "supported_queries": [
                {
                    "query_id": "supported-source-admission",
                    "query": "How should a RAG source be admitted?",
                    "expected_outcome": "supported",
                }
            ],
            "boundary_queries": [
                {
                    "query_id": "boundary-source-admission",
                    "query": "Should an unknown public URL be treated as verified?",
                    "expected_outcome": "insufficient_evidence",
                }
            ],
        },
        body={
            "decision_query": "Which source admission conditions are required before indexing?",
            "recommendation_or_reviewed_branches": "Require source identity, tier, authority, access scope, and availability.",
            "applicability": "Applies to team-shared Production RAG source admission.",
            "non_applicability": "Does not approve subset-authorized or secret-bearing source material.",
            "alternatives": "Treat all HTTPS URLs as admissible evidence.",
            "trade_offs": "Verification adds review work but retains auditable support.",
            "failure_modes": "Unverified or unavailable sources can make claimed support invalid.",
            "minimum_implementation_guidance": "Validate source identity and source locator before review.",
            "minimum_validation_guidance": "Reject inaccessible or malformed source records.",
            "minimum_diagnosis_guidance": "Record a freshness event when a source changes.",
            "minimum_acceptance_guidance": "Run one supported and one Boundary query after downstream publication.",
            "conflicts": "No unresolved conflict is known for this decision.",
            "unknowns": "Future connector behavior remains outside this entry.",
            "boundary_conditions": "The recommendation assumes a team-shared authorized corpus.",
        },
        section_source_relationships=[
            {"section_id": section_id, "source_ids": ["source-rag-admission-001"]}
            for section_id in (
                "decision_query",
                "recommendation_or_reviewed_branches",
                "applicability",
                "non_applicability",
                "alternatives",
                "trade_offs",
                "failure_modes",
                "minimum_implementation_guidance",
                "minimum_validation_guidance",
                "minimum_diagnosis_guidance",
                "minimum_acceptance_guidance",
                "conflicts",
                "unknowns",
                "boundary_conditions",
            )
        ],
        claims=[
            {
                "claim_id": "claim-source-admission-policy",
                "claim_kind": "prescriptive",
                "statement": "Source admission requires verified source metadata.",
                "section_id": "recommendation_or_reviewed_branches",
                "source_ids": ["source-rag-admission-001"],
                "material": True,
                "scope": "team_shared",
            }
        ],
        relationship={},
    )


def _release_assurance_acceptance_payload(
    *,
    entry_identity: str,
    contract_identity: str,
    calibration_identity: str,
    named_gate: str,
) -> dict:
    checks = [
        {
            "check_id": check_id,
            "result": "passed",
            "evidence_links": [f"evidence://editorial-authority/{check_id.removeprefix('check:')}"],
        }
        for check_id in (
            "check:impact-declaration",
            "check:bundle-secret-scan",
            "check:product-path-impact",
            "check:configuration-impact",
        )
    ]
    checks.extend(
        {
            "check_id": check_id,
            "result": "passed",
            "evidence_links": [f"evidence://editorial-authority/{check_id.removeprefix('check:')}"],
            "identity_dependencies": [entry_identity],
        }
        for check_id in (
            "check:entry-supported-query",
            "check:entry-boundary-query",
            "check:evidence-citation-identity",
        )
    )
    return CreateDeliveryAcceptanceRecordRequest.model_validate(
        {
            "stage": "local_development",
            "affected_scope": {
                "entry_identities": [entry_identity],
                "collection_identities": [],
                "product_path_identities": [contract_identity],
                "configuration_identities": [calibration_identity],
                "protected_capability_identities": [],
                "public_claim_identities": [],
                "deployment_identity": "deployment:editorial-authority-release-assurance",
                "expected_blocking_scope": "entry_version",
                "blocking_scope_identity": entry_identity,
            },
            "content_identities": [entry_identity],
            "product_identities": [contract_identity, calibration_identity, named_gate],
            "checks": checks,
            "evidence_links": ["evidence://editorial-authority/release-assurance"],
        }
    ).model_dump(mode="json")


def _release_assurance_authority_records(
    *,
    entry_identity: str,
    contract_identity: str,
    calibration_identity: str,
    acceptance_identity: str,
    named_gate: str,
    qualified_active_status: bool,
) -> list[CanonicalRecordModel | CanonicalEventModel]:
    acceptance_payload = _release_assurance_acceptance_payload(
        entry_identity=entry_identity,
        contract_identity=contract_identity,
        calibration_identity=calibration_identity,
        named_gate=named_gate,
    )
    verified_checks = [
        {
            "check_id": check["check_id"],
            "evidence_links": check["evidence_links"],
        }
        for check in acceptance_payload["checks"]
        if check["result"] in {"passed", "carried_forward"}
    ]
    return [
        CanonicalRecordModel(
            stable_id=contract_identity,
            identity_kind=StableIdentityKind.PRODUCT_PATH.value,
            identity_value=contract_identity.removeprefix("product_path:"),
            state="active",
            record_class=CanonicalRecordClass.AUTHORITATIVE.value,
            payload={"schema": "product_path/v1"},
        ),
        CanonicalRecordModel(
            stable_id=calibration_identity,
            identity_kind=StableIdentityKind.CONFIGURATION.value,
            identity_value=calibration_identity.removeprefix("configuration:"),
            state="active",
            record_class=CanonicalRecordClass.AUTHORITATIVE.value,
            payload={"schema": "configuration/v1"},
        ),
        CanonicalRecordModel(
            stable_id=acceptance_identity,
            identity_kind=StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value,
            identity_value=acceptance_identity.removeprefix("delivery_acceptance_record:"),
            state="created",
            record_class=CanonicalRecordClass.IMMUTABLE.value,
            payload={
                "schema": "delivery_acceptance_record/v1",
                **acceptance_payload,
                "change_owner_identity": "member:release-evaluator-001",
                "evaluator_identity": "member:release-evaluator-001",
            },
        ),
        CanonicalRecordModel(
            stable_id=named_gate,
            identity_kind=StableIdentityKind.CAPABILITY.value,
            identity_value=named_gate.removeprefix("capability:"),
            state="active",
            record_class=CanonicalRecordClass.AUTHORITATIVE.value,
            payload={"schema": "capability/v1"},
        ),
        CanonicalEventModel(
            aggregate_id=acceptance_identity,
            aggregate_kind=StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value,
            event_type=(
                CanonicalEventType.STATUS_CHANGED.value
                if qualified_active_status
                else CanonicalEventType.CREATED.value
            ),
            from_state="at_risk" if qualified_active_status else None,
            to_state=AcceptanceStatus.ACTIVE.value,
            payload=(
                {
                    "schema": "delivery_acceptance_status/v1",
                    "reason_code": "checks_verified",
                    "superseding_record_identity": None,
                    "verified_checks": verified_checks,
                    "status_failure": None,
                    "reacceptance_trigger": None,
                    "invalidated_check_ids": [],
                }
                if qualified_active_status
                else {"schema": "delivery_acceptance_status/v1"}
            ),
            recorded_by="member:release-evaluator-001",
        ),
    ]


async def _prepare_editorial_review(
    authority: EditorialAuthorityService,
    entry_id: str,
    author: User,
    maintainer: User,
) -> tuple[dict, dict, dict, dict]:
    collected = await authority.collect_evidence(entry_id, author)
    accepted = await authority.accept_maintainer_responsibility(entry_id, maintainer)
    verified = await authority.record_source_availability(
        entry_id,
        "source-rag-admission-001",
        "verified_usable",
        maintainer,
    )
    awaiting_review = await authority.request_editorial_review(entry_id, author)
    return collected, accepted, verified, awaiting_review


async def test_distinct_reviewer_approves_private_revision_and_admin_exports_deterministically(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    administrator = User(username="administrator", password_hash="hash", role="admin", is_active=True)
    db_session.add_all([author, reviewer, maintainer, administrator])
    await db_session.commit()

    authority = EditorialAuthorityService(db_session)
    draft = await authority.create_draft(_review_ready_entry(), author)
    collected, maintainer_accepted, sources_verified, awaiting_review = await _prepare_editorial_review(
        authority,
        draft["entry_id"],
        author,
        maintainer,
    )

    assert collected["lifecycle_state"] == "evidence_collected"
    assert maintainer_accepted["maintainer_acceptance"]["status"] == "accepted"
    assert sources_verified["sources"][0]["availability"] == "verified_usable"
    assert awaiting_review["lifecycle_state"] == "editorial_review"
    with pytest.raises(AppError) as exc_info:
        await authority.approve_current_revision(draft["entry_id"], author)
    assert exc_info.value.code == "EDITORIAL_SELF_APPROVAL_FORBIDDEN"

    approved = await authority.approve_current_revision(draft["entry_id"], reviewer)
    exported = await authority.export_approved_revision(draft["entry_id"], administrator)
    reconstructed = await authority.reconstruct_export(draft["entry_id"], approved["revision_identity"])

    assert approved["approval"]["status"] == "approved"
    assert approved["approval"]["reviewer_identity"].startswith("member:")
    assert exported["editorial_revision_identity"] == approved["revision_identity"]
    assert exported["artifact_sha256"] == reconstructed["artifact_sha256"]
    assert exported["artifact"] == reconstructed["artifact"]
    assert "automatic_publication" not in exported["artifact"]
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0


async def test_bundle_intake_verifier_reads_only_the_retained_approved_export_and_current_source_authority(
    db_session,
) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    administrator = User(username="administrator", password_hash="hash", role="admin", is_active=True)
    db_session.add_all([author, reviewer, maintainer, administrator])
    await db_session.commit()

    authority = EditorialAuthorityService(db_session)
    draft = await authority.create_draft(_review_ready_entry(), author)
    await _prepare_editorial_review(authority, draft["entry_id"], author, maintainer)
    await authority.approve_current_revision(draft["entry_id"], reviewer)
    exported = await authority.export_approved_revision(draft["entry_id"], administrator)
    events_before = await db_session.scalar(select(func.count()).select_from(CanonicalEventModel))

    verified = await CanonicalEditorialExportVerifier(db_session).verify(
        exported["artifact"],
        exported["artifact_sha256"],
    )

    assert verified == exported["artifact"]
    assert await db_session.scalar(select(func.count()).select_from(CanonicalEventModel)) == events_before

    await authority.record_source_availability(
        draft["entry_id"],
        "source-rag-admission-001",
        "unavailable_for_new_evidence",
        maintainer,
    )

    with pytest.raises(AppError) as exc_info:
        await CanonicalEditorialExportVerifier(db_session).verify(
            exported["artifact"],
            exported["artifact_sha256"],
        )

    assert exc_info.value.code == "EDITORIAL_SOURCE_UNAVAILABLE"


async def test_export_snapshot_rejects_source_definitions_not_bound_to_the_approved_revision(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, reviewer, maintainer])
    await db_session.commit()

    authority = EditorialAuthorityService(db_session)
    draft = await authority.create_draft(_review_ready_entry(), author)
    await _prepare_editorial_review(authority, draft["entry_id"], author, maintainer)
    approved = await authority.approve_current_revision(draft["entry_id"], reviewer)
    _entry, events = await authority._entry_and_events(draft["entry_id"])
    tampered_approval = deepcopy(approved["approval"])
    source_snapshot = tampered_approval["authority_snapshot"]["sources"][0]
    source_snapshot["source"]["authority"] = "An unrelated authority"
    source_snapshot["source_definition_sha256"] = EditorialAuthorityService._sha256(source_snapshot["source"])

    with pytest.raises(AppError) as exc_info:
        authority._approval_authority_snapshot(
            tampered_approval,
            draft=_review_ready_entry(),
            entry_identity=draft["entry_identity"],
            revision_identity=approved["revision_identity"],
            events=events,
        )

    assert exc_info.value.code == "EDITORIAL_EXPORT_SNAPSHOT_MISSING"


async def test_author_cannot_assign_themself_as_the_approving_reviewer(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, maintainer])
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    entry = _review_ready_entry()
    entry.approving_reviewer_username = "author"
    draft = await authority.create_draft(entry, author)
    await authority.collect_evidence(draft["entry_id"], author)

    with pytest.raises(AppError) as exc_info:
        await authority.request_editorial_review(draft["entry_id"], author)

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "EDITORIAL_ROLE_SEPARATION_REQUIRED"


async def test_duplicate_ids_and_invalid_conditions_or_sources_return_structured_reasons(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    db_session.add(author)
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    await authority.create_draft(_review_ready_entry(), author)

    with pytest.raises(AppError) as exc_info:
        await authority.create_draft(_review_ready_entry(), author)
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "EDITORIAL_ENTRY_ID_DUPLICATE"

    invalid = _review_ready_entry(entry_id="rag-source-admission-invalid")
    invalid.applicability_conditions[0]["operator"] = "contains"
    invalid.sources[0]["public_url"] = "https://example.com/rag/source-admission?token=secret"
    invalid.sources[0]["source_tier"] = "bounded_internal_case"
    reasons = review_validation_reasons(invalid)

    assert {
        "field": "applicability_conditions[0].operator",
        "code": "invalid",
        "message": "condition operator is not supported",
    } in reasons
    assert {
        "field": "sources[0].public_url",
        "code": "invalid",
        "message": "public sources require a sanitized canonical public HTTPS URL",
    } in reasons
    assert {
        "field": "sources[0].access_scope",
        "code": "source_tier_scope_invalid",
        "message": "Bounded Internal Cases require controlled_internal access scope",
    } in reasons


def test_high_impact_claim_links_cannot_be_evaded_by_lowering_assurance() -> None:
    source_grounded = _review_ready_entry()
    source_grounded.claims[0]["source_ids"] = []

    reasons = review_validation_reasons(source_grounded)

    assert {
        "field": "claims[0].source_ids",
        "code": "claim_evidence_required",
        "message": "material high-impact claims require at least one Claim-Evidence Link",
    } in reasons

    secondary_only = _review_ready_entry()
    secondary_only.sources[0]["source_tier"] = "secondary_discovery_source"

    reasons = review_validation_reasons(secondary_only)

    assert {
        "field": "claims[0].source_ids",
        "code": "source_tier_ineligible",
        "message": "material high-impact claims cannot rely only on Secondary Discovery Sources",
    } in reasons


def test_content_indicated_high_impact_claims_require_links_even_when_mislabeled() -> None:
    mislabeled = _review_ready_entry()
    high_impact_text = "Security policy requires a score >= 0.80 before source admission."
    mislabeled.body["recommendation_or_reviewed_branches"] = high_impact_text
    mislabeled.claims = [
        {
            "claim_id": "claim-mislabeled-security-threshold",
            "claim_kind": "ordinary",
            "statement": high_impact_text,
            "section_id": "recommendation_or_reviewed_branches",
            "source_ids": [],
            "material": False,
        }
    ]

    reasons = review_validation_reasons(mislabeled)

    assert {
        "field": "claims[0].source_ids",
        "code": "claim_evidence_required",
        "message": "material high-impact claims require at least one Claim-Evidence Link",
    } in reasons
    assert {
        "field": "body.recommendation_or_reviewed_branches",
        "code": "claim_evidence_required",
        "message": "high-impact authored decision text requires a Claim-Evidence Link",
    } in reasons


def test_high_impact_body_requires_a_linked_high_impact_claim_not_an_unrelated_link() -> None:
    entry = _review_ready_entry()
    entry.body["recommendation_or_reviewed_branches"] = (
        "Security policy requires a score >= 0.80 before source admission."
    )
    entry.claims = [
        {
            "claim_id": "claim-unrelated-source-title",
            "claim_kind": "ordinary",
            "statement": "The retained source has a review title.",
            "section_id": "recommendation_or_reviewed_branches",
            "source_ids": ["source-rag-admission-001"],
            "material": False,
        }
    ]

    reasons = review_validation_reasons(entry)

    assert {
        "field": "body.recommendation_or_reviewed_branches",
        "code": "claim_evidence_required",
        "message": "high-impact authored decision text requires a Claim-Evidence Link",
    } in reasons


def test_chinese_high_impact_claims_and_authored_text_require_claim_evidence_links() -> None:
    entry = _review_ready_entry()
    high_impact_text = "安全策略必须在七天内复核生产环境的授权变更。"
    entry.body["recommendation_or_reviewed_branches"] = high_impact_text
    entry.claims = [
        {
            "claim_id": "claim-chinese-security-review",
            "claim_kind": "ordinary",
            "statement": high_impact_text,
            "section_id": "recommendation_or_reviewed_branches",
            "source_ids": [],
            "material": False,
        }
    ]

    reasons = review_validation_reasons(entry)

    assert {
        "field": "claims[0].source_ids",
        "code": "claim_evidence_required",
        "message": "material high-impact claims require at least one Claim-Evidence Link",
    } in reasons
    assert {
        "field": "body.recommendation_or_reviewed_branches",
        "code": "claim_evidence_required",
        "message": "high-impact authored decision text requires a Claim-Evidence Link",
    } in reasons


def test_duplicate_claim_and_acceptance_query_ids_return_structured_reasons() -> None:
    duplicate = _review_ready_entry()
    duplicate.claims.append(duplicate.claims[0].copy())
    duplicate.acceptance_material["boundary_queries"][0]["query_id"] = "supported-source-admission"

    reasons = review_validation_reasons(duplicate)

    assert {
        "field": "claims[1].claim_id",
        "code": "duplicate",
        "message": "claim_id must be unique within an entry",
    } in reasons
    assert {
        "field": "acceptance_material.boundary_queries[0].query_id",
        "code": "duplicate",
        "message": "query_id must be unique across accepted queries",
    } in reasons


def test_wording_only_revisions_fail_closed_for_semantic_operators() -> None:
    previous = _review_ready_entry()
    whitespace_only = previous.model_copy(deep=True)
    whitespace_only.body["trade_offs"] = "Verification  adds review work but retains auditable support."

    assert lightweight_revision_reasons(previous, whitespace_only) == []

    comparison_changed = previous.model_copy(deep=True)
    comparison_changed.body["recommendation_or_reviewed_branches"] = "Require score <= 0.80 before source admission."
    comparison_previous = previous.model_copy(deep=True)
    comparison_previous.body["recommendation_or_reviewed_branches"] = "Require score >= 0.80 before source admission."

    comparison_reasons = lightweight_revision_reasons(comparison_previous, comparison_changed)
    assert {reason["field"] for reason in comparison_reasons} == {"body.recommendation_or_reviewed_branches"}

    boolean_changed = previous.model_copy(deep=True)
    boolean_changed.body["recommendation_or_reviewed_branches"] = "Require source identity || source locator."
    boolean_previous = previous.model_copy(deep=True)
    boolean_previous.body["recommendation_or_reviewed_branches"] = "Require source identity && source locator."

    boolean_reasons = lightweight_revision_reasons(boolean_previous, boolean_changed)
    assert {reason["field"] for reason in boolean_reasons} == {"body.recommendation_or_reviewed_branches"}


def test_editorial_secret_and_export_scans_cover_compound_fields_and_imperative_autopublish_text() -> None:
    assert editorial_secret_scan_findings({"client_secret": "not-a-secret-shaped-token"}) == [
        "client_secret:secret-bearing-field"
    ]
    assert editorial_secret_scan_findings({"access_token": ["not-a-secret-shaped-token"]}) == [
        "access_token:secret-bearing-field"
    ]
    assert editorial_export_safety_findings(
        {"publication_instruction": "Automatically publish this export to production."}
    ) == ["publication_instruction:automatic-publication-instruction"]
    assert editorial_export_safety_findings({"auto_publish_enabled": True}) == [
        "auto_publish_enabled:automatic-publication-instruction"
    ]
    assert editorial_export_safety_findings(
        {"publication_instruction": "Publish this export to production now."}
    ) == ["publication_instruction:automatic-publication-instruction"]


def test_editorial_schema_version_is_explicit_and_closed() -> None:
    assert CreateEditorialEntryRequest(entry_id="entry-schema-001", title="Versioned schema").schema_version == 1
    with pytest.raises(ValidationError):
        CreateEditorialEntryRequest(entry_id="entry-schema-002", title="Unsupported version", schema_version=2)


def test_release_assured_requires_frozen_canonical_assurance_references() -> None:
    release_assured = _review_ready_entry(assurance_level="release_assured")
    release_assured.release_assurance = {
        "contract_identity": "product_path:claim-contract-v1",
        "calibration_identity": "configuration:resolver-calibration-v1",
        "frozen_acceptance_identity": "latest",
        "named_gate": "capability:release-assurance-gate-v1",
    }

    reasons = review_validation_reasons(release_assured)

    assert {
        "field": "release_assurance.frozen_acceptance_identity",
        "code": "invalid",
        "message": "frozen_acceptance_identity must use the delivery_acceptance_record identity namespace",
    } in reasons

    release_assured.release_assurance["frozen_acceptance_identity"] = "delivery_acceptance_record:editorial-preview-001"
    assert review_validation_reasons(release_assured) == []


async def test_release_assured_requires_retained_active_canonical_assurance_records(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, reviewer, maintainer])
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    entry = _review_ready_entry(assurance_level="release_assured")
    entry.release_assurance = {
        "contract_identity": "product_path:claim-contract-v1",
        "calibration_identity": "configuration:resolver-calibration-v1",
        "frozen_acceptance_identity": "delivery_acceptance_record:editorial-preview-001",
        "named_gate": "capability:release-assurance-gate-v1",
    }
    draft = await authority.create_draft(entry, author)

    with pytest.raises(AppError) as exc_info:
        await authority.collect_evidence(draft["entry_id"], author)
    assert exc_info.value.code == "EDITORIAL_RELEASE_ASSURANCE_UNVERIFIED"
    assert {reason["field"] for reason in exc_info.value.detail["reasons"]} == {
        "release_assurance.contract_identity",
        "release_assurance.calibration_identity",
        "release_assurance.frozen_acceptance_identity",
        "release_assurance.named_gate",
    }

    db_session.add_all(
        _release_assurance_authority_records(
            entry_identity="entry:rag-source-admission-001",
            contract_identity="product_path:claim-contract-v1",
            calibration_identity="configuration:resolver-calibration-v1",
            acceptance_identity="delivery_acceptance_record:editorial-preview-001",
            named_gate="capability:release-assurance-gate-v1",
            qualified_active_status=True,
        )
    )
    await db_session.commit()

    collected = await authority.collect_evidence(draft["entry_id"], author)
    assert collected["lifecycle_state"] == "evidence_collected"


async def test_release_assured_rejects_unqualified_delivery_acceptance_active_event(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, reviewer, maintainer])
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    entry = _review_ready_entry(entry_id="release-assured-unqualified-001", assurance_level="release_assured")
    entry.release_assurance = {
        "contract_identity": "product_path:claim-contract-unqualified-v1",
        "calibration_identity": "configuration:resolver-calibration-unqualified-v1",
        "frozen_acceptance_identity": "delivery_acceptance_record:editorial-preview-unqualified-001",
        "named_gate": "capability:release-assurance-gate-unqualified-v1",
    }
    draft = await authority.create_draft(entry, author)
    db_session.add_all(
        _release_assurance_authority_records(
            entry_identity=draft["entry_identity"],
            contract_identity="product_path:claim-contract-unqualified-v1",
            calibration_identity="configuration:resolver-calibration-unqualified-v1",
            acceptance_identity="delivery_acceptance_record:editorial-preview-unqualified-001",
            named_gate="capability:release-assurance-gate-unqualified-v1",
            qualified_active_status=False,
        )
    )
    await db_session.commit()

    with pytest.raises(AppError) as exc_info:
        await authority.collect_evidence(draft["entry_id"], author)

    assert exc_info.value.code == "EDITORIAL_RELEASE_ASSURANCE_UNVERIFIED"
    assert {
        "field": "release_assurance.frozen_acceptance_identity",
        "code": "acceptance_not_active",
        "message": "Release-Assured delivery acceptance must have an active retained status event",
    } in exc_info.value.detail["reasons"]


async def test_release_assured_rejects_an_active_event_not_permitted_by_delivery_acceptance_service(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, reviewer, maintainer])
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    entry = _review_ready_entry(entry_id="release-assured-suspended-001", assurance_level="release_assured")
    entry.release_assurance = {
        "contract_identity": "product_path:claim-contract-suspended-v1",
        "calibration_identity": "configuration:resolver-calibration-suspended-v1",
        "frozen_acceptance_identity": "delivery_acceptance_record:editorial-preview-suspended-001",
        "named_gate": "capability:release-assurance-gate-suspended-v1",
    }
    draft = await authority.create_draft(entry, author)
    records = _release_assurance_authority_records(
        entry_identity=draft["entry_identity"],
        contract_identity="product_path:claim-contract-suspended-v1",
        calibration_identity="configuration:resolver-calibration-suspended-v1",
        acceptance_identity="delivery_acceptance_record:editorial-preview-suspended-001",
        named_gate="capability:release-assurance-gate-suspended-v1",
        qualified_active_status=True,
    )
    status_event = next(record for record in records if isinstance(record, CanonicalEventModel))
    status_event.from_state = AcceptanceStatus.SUSPENDED.value
    db_session.add_all(records)
    await db_session.commit()

    with pytest.raises(AppError) as exc_info:
        await authority.collect_evidence(draft["entry_id"], author)

    assert exc_info.value.code == "EDITORIAL_RELEASE_ASSURANCE_UNVERIFIED"
    assert {
        "field": "release_assurance.frozen_acceptance_identity",
        "code": "acceptance_not_active",
        "message": "Release-Assured delivery acceptance must have an active retained status event",
    } in exc_info.value.detail["reasons"]


async def test_release_assured_rejects_an_active_acceptance_that_does_not_cover_the_entry(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    db_session.add(author)
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    entry = _review_ready_entry(entry_id="release-assured-unrelated-001", assurance_level="release_assured")
    entry.release_assurance = {
        "contract_identity": "product_path:claim-contract-unrelated-v1",
        "calibration_identity": "configuration:resolver-calibration-unrelated-v1",
        "frozen_acceptance_identity": "delivery_acceptance_record:editorial-preview-unrelated-001",
        "named_gate": "capability:release-assurance-gate-unrelated-v1",
    }
    draft = await authority.create_draft(entry, author)
    db_session.add_all(
        _release_assurance_authority_records(
            entry_identity="entry:other-entry-001",
            contract_identity="product_path:claim-contract-unrelated-v1",
            calibration_identity="configuration:resolver-calibration-unrelated-v1",
            acceptance_identity="delivery_acceptance_record:editorial-preview-unrelated-001",
            named_gate="capability:release-assurance-gate-unrelated-v1",
            qualified_active_status=True,
        )
    )
    await db_session.commit()

    with pytest.raises(AppError) as exc_info:
        await authority.collect_evidence(draft["entry_id"], author)

    assert exc_info.value.code == "EDITORIAL_RELEASE_ASSURANCE_UNVERIFIED"
    assert {
        "field": "release_assurance.frozen_acceptance_identity",
        "code": "acceptance_scope_mismatch",
        "message": "frozen delivery acceptance must cover this entry and every named Release-Assured authority record",
    } in exc_info.value.detail["reasons"]


async def test_material_revision_requires_new_review_while_wording_revision_uses_lightweight_acceptance(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    reviewer_two = User(username="reviewer-two", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, reviewer, reviewer_two, maintainer])
    await db_session.commit()

    authority = EditorialAuthorityService(db_session)
    draft = await authority.create_draft(_review_ready_entry(), author)
    await _prepare_editorial_review(authority, draft["entry_id"], author, maintainer)
    first_approval = await authority.approve_current_revision(draft["entry_id"], reviewer)

    incomplete = CreateEditorialEntryRequest(
        entry_id=draft["entry_id"],
        title="An incomplete material change must not enter Editorial Review",
    )
    with pytest.raises(AppError) as exc_info:
        await authority.revise_entry(
            draft["entry_id"],
            ReviseEditorialEntryRequest(entry=incomplete, change_kind="material"),
            author,
        )
    assert exc_info.value.code == "EDITORIAL_ENTRY_INVALID"

    bypass = _review_ready_entry().model_copy(deep=True)
    bypass.body["recommendation_or_reviewed_branches"] = "Do not retain an authority record before source admission."
    with pytest.raises(AppError) as exc_info:
        await authority.revise_entry(
            draft["entry_id"],
            ReviseEditorialEntryRequest(
                entry=bypass,
                change_kind="wording_only",
                lightweight_reason="caller-declared wording-only revision",
            ),
            maintainer,
        )
    assert exc_info.value.code == "EDITORIAL_LIGHTWEIGHT_CHANGE_INVALID"

    material = _review_ready_entry().model_copy(deep=True)
    material.body["recommendation_or_reviewed_branches"] = "Require an explicit authority record before source admission."
    material.approving_reviewer_username = "reviewer-two"
    revised = await authority.revise_entry(
        draft["entry_id"],
        ReviseEditorialEntryRequest(entry=material, change_kind="material"),
        reviewer,
    )

    assert revised["revision_identity"] != first_approval["revision_identity"]
    assert revised["lifecycle_state"] == "editorial_review"
    assert revised["approval"]["status"] == "pending"
    with pytest.raises(AppError) as exc_info:
        await authority.approve_current_revision(draft["entry_id"], reviewer)
    assert exc_info.value.code == "EDITORIAL_SELF_APPROVAL_FORBIDDEN"
    await authority.accept_maintainer_responsibility(draft["entry_id"], maintainer)
    second_approval = await authority.approve_current_revision(draft["entry_id"], reviewer_two)

    wording = material.model_copy(deep=True)
    wording.body["trade_offs"] = "Verification  adds review work but retains auditable support."
    wording.approving_reviewer_username = "reviewer-two"
    pending_lightweight = await authority.revise_entry(
        draft["entry_id"],
        ReviseEditorialEntryRequest(
            entry=wording,
            change_kind="wording_only",
            lightweight_reason="clarity-only wording correction",
        ),
        maintainer,
    )

    assert pending_lightweight["revision_identity"] != second_approval["revision_identity"]
    assert pending_lightweight["approval"]["status"] == "pending_lightweight_acceptance"
    await authority.accept_maintainer_responsibility(draft["entry_id"], maintainer)
    accepted = await authority.accept_wording_revision(draft["entry_id"], reviewer_two)
    assert accepted["approval"]["status"] == "lightweight_accepted"


async def test_secret_fixture_is_rejected_before_it_can_enter_retained_editorial_authority(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    db_session.add(author)
    await db_session.commit()
    unsafe = _review_ready_entry()
    unsafe.body["unknowns"] = "Never retain a token like " + "sk-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456."

    with pytest.raises(AppError) as exc_info:
        await EditorialAuthorityService(db_session).create_draft(unsafe, author)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "EDITORIAL_SECRET_REJECTED"
    assert exc_info.value.detail["findings"] == ["body.unknowns:openai-api-key"]


async def test_conflicting_stable_source_identity_is_rejected_from_private_authority(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, reviewer, maintainer])
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)

    first = await authority.create_draft(_review_ready_entry(entry_id="rag-source-admission-001"), author)
    await authority.collect_evidence(first["entry_id"], author)

    conflicting = _review_ready_entry(entry_id="rag-source-admission-002")
    conflicting.sources[0]["authority"] = "Different authority cannot rewrite the same source identity"
    second = await authority.create_draft(conflicting, author)

    with pytest.raises(AppError) as exc_info:
        await authority.collect_evidence(second["entry_id"], author)

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "EDITORIAL_SOURCE_ID_CONFLICT"
    assert exc_info.value.detail == {"source_id": "source-rag-admission-001"}


async def test_maintainer_records_source_availability_as_an_append_only_authority_event(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    administrator = User(username="administrator", password_hash="hash", role="admin", is_active=True)
    db_session.add_all([author, reviewer, maintainer, administrator])
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    draft = await authority.create_draft(_review_ready_entry(), author)
    await _prepare_editorial_review(authority, draft["entry_id"], author, maintainer)
    approved = await authority.approve_current_revision(draft["entry_id"], reviewer)
    exported_before_change = await authority.export_approved_revision(draft["entry_id"], administrator)

    unavailable = await authority.record_source_availability(
        draft["entry_id"],
        "source-rag-admission-001",
        "unavailable_for_new_evidence",
        maintainer,
    )

    source = await db_session.get(CanonicalRecordModel, "source:source-rag-admission-001")
    assert source is not None
    assert source.state == "changed_or_unreachable_awaiting_review"
    assert unavailable["sources"] == [
        {
            "source_id": "source-rag-admission-001",
            "source_identity": "source:source-rag-admission-001",
            "availability": "unavailable_for_new_evidence",
            "access_scope": "public",
        }
    ]
    assert "source_unavailable" in unavailable["eligibility"]["reasons"]
    assert "decisive_source_loss" in unavailable["eligibility"]["reasons"]
    reconstructed = await authority.reconstruct_export(draft["entry_id"], approved["revision_identity"])
    assert reconstructed["artifact_sha256"] == exported_before_change["artifact_sha256"]
    assert reconstructed["artifact"] == exported_before_change["artifact"]
    with pytest.raises(AppError) as exc_info:
        await authority.export_approved_revision(draft["entry_id"], administrator)
    assert exc_info.value.code == "EDITORIAL_SOURCE_UNAVAILABLE"


def test_malformed_maintainer_and_approval_events_do_not_create_authority() -> None:
    revision_identity = "editorial_revision:authority-event-validation-001.r1"
    roles = {
        "author_identity": "member:authority-author-001",
        "approving_reviewer_identity": "member:authority-reviewer-001",
        "accountable_maintainer_identity": "member:authority-maintainer-001",
    }
    role_assignment = CanonicalEventModel(
        aggregate_id="entry:authority-event-validation-001",
        aggregate_kind=StableIdentityKind.ENTRY.value,
        event_type=CanonicalEventType.STATE_CHANGED.value,
        from_state="draft",
        to_state="evidence_collected",
        payload={
            "schema": "editorial_authority_event/v1",
            "sequence": 1,
            "action": "evidence_collected",
            "revision_identity": revision_identity,
            "roles": roles,
        },
        recorded_by=roles["author_identity"],
    )
    malformed_maintainer_acceptance = CanonicalEventModel(
        aggregate_id="entry:authority-event-validation-001",
        aggregate_kind=StableIdentityKind.ENTRY.value,
        event_type=CanonicalEventType.CREATED.value,
        from_state=None,
        to_state="evidence_collected",
        payload={
            "schema": "editorial_authority_event/v1",
            "sequence": 2,
            "action": "maintainer_responsibility_accepted",
            "revision_identity": revision_identity,
            "roles": roles,
        },
        recorded_by=roles["accountable_maintainer_identity"],
    )
    malformed_approval = CanonicalEventModel(
        aggregate_id="entry:authority-event-validation-001",
        aggregate_kind=StableIdentityKind.ENTRY.value,
        event_type=CanonicalEventType.CREATED.value,
        from_state=None,
        to_state="editorial_review",
        payload={
            "schema": "editorial_authority_event/v1",
            "sequence": 3,
            "action": "revision_approved",
            "revision_identity": revision_identity,
            "roles": roles,
            "authority_snapshot": {},
        },
        recorded_by=roles["approving_reviewer_identity"],
    )
    events = [role_assignment, malformed_maintainer_acceptance, malformed_approval]

    assert EditorialAuthorityService._maintainer_acceptance_for_revision(events, revision_identity) is None
    assert EditorialAuthorityService._approval_for_revision(events, revision_identity) is None


async def test_author_cannot_review_or_export_a_source_until_maintainer_accepts_and_verifies_it(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, reviewer, maintainer])
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    draft = await authority.create_draft(_review_ready_entry(), author)
    collected = await authority.collect_evidence(draft["entry_id"], author)

    assert collected["sources"][0]["availability"] == "unknown"
    with pytest.raises(AppError) as exc_info:
        await authority.request_editorial_review(draft["entry_id"], author)
    assert exc_info.value.code == "EDITORIAL_MAINTAINER_ACCEPTANCE_REQUIRED"

    await authority.accept_maintainer_responsibility(draft["entry_id"], maintainer)
    with pytest.raises(AppError) as exc_info:
        await authority.request_editorial_review(draft["entry_id"], author)
    assert exc_info.value.code == "EDITORIAL_SOURCE_UNAVAILABLE"

    await authority.record_source_availability(
        draft["entry_id"],
        "source-rag-admission-001",
        "verified_usable",
        maintainer,
    )
    reviewed = await authority.request_editorial_review(draft["entry_id"], author)
    assert reviewed["lifecycle_state"] == "editorial_review"
    assert reviewed["approval"]["status"] == "pending"


async def test_missing_source_event_fails_closed_even_when_an_immutable_record_has_a_stale_state(db_session) -> None:
    source = CanonicalRecordModel(
        stable_id="source:stale-source-001",
        identity_kind=StableIdentityKind.SOURCE.value,
        identity_value="stale-source-001",
        state="verified_usable",
        record_class=CanonicalRecordClass.AUTHORITATIVE.value,
        payload={"schema": "editorial_source/v1", "source": {}},
    )
    db_session.add(source)
    await db_session.commit()

    assert await EditorialAuthorityService(db_session)._source_availability(source) == "unknown"


async def test_source_availability_fails_closed_for_unrelated_or_unaccepted_events(db_session) -> None:
    source = CanonicalRecordModel(
        stable_id="source:untrusted-source-event-001",
        identity_kind=StableIdentityKind.SOURCE.value,
        identity_value="untrusted-source-event-001",
        state="verified_usable",
        record_class=CanonicalRecordClass.AUTHORITATIVE.value,
        payload={"schema": "editorial_source/v1", "source": {}},
    )
    db_session.add_all(
        [
            source,
            CanonicalEventModel(
                aggregate_id=source.stable_id,
                aggregate_kind=StableIdentityKind.ENTRY.value,
                event_type=CanonicalEventType.STATUS_CHANGED.value,
                from_state="unknown",
                to_state="verified_usable",
                payload={"schema": "unrelated_event/v1", "action": "unrelated"},
                recorded_by="member:unrelated-001",
            ),
            CanonicalEventModel(
                aggregate_id=source.stable_id,
                aggregate_kind=StableIdentityKind.SOURCE.value,
                event_type=CanonicalEventType.STATUS_CHANGED.value,
                from_state="unknown",
                to_state="verified_usable",
                payload={
                    "schema": "editorial_source_event/v1",
                    "action": "source_availability_recorded",
                    "entry_identity": "entry:untrusted-source-event-entry-001",
                    "editorial_revision_identity": "editorial_revision:untrusted-source-event-entry-001.r1",
                    "maintainer_acceptance_event_id": "missing-maintainer-acceptance",
                },
                recorded_by="member:unaccepted-maintainer-001",
            ),
        ]
    )
    await db_session.commit()

    assert await EditorialAuthorityService(db_session)._source_availability(source) == "unknown"


async def test_source_availability_event_must_bind_a_source_of_the_accepted_revision(db_session) -> None:
    author = User(username="author", password_hash="hash", role="user", is_active=True)
    reviewer = User(username="reviewer", password_hash="hash", role="user", is_active=True)
    maintainer = User(username="maintainer", password_hash="hash", role="user", is_active=True)
    db_session.add_all([author, reviewer, maintainer])
    await db_session.commit()
    authority = EditorialAuthorityService(db_session)
    draft = await authority.create_draft(_review_ready_entry(), author)
    await authority.collect_evidence(draft["entry_id"], author)
    accepted = await authority.accept_maintainer_responsibility(draft["entry_id"], maintainer)
    unrelated_source = CanonicalRecordModel(
        stable_id="source:accepted-revision-unrelated-source-001",
        identity_kind=StableIdentityKind.SOURCE.value,
        identity_value="accepted-revision-unrelated-source-001",
        state="changed_or_unreachable_awaiting_review",
        record_class=CanonicalRecordClass.AUTHORITATIVE.value,
        payload={"schema": "editorial_source/v1", "source": {}},
    )
    db_session.add_all(
        [
            unrelated_source,
            CanonicalEventModel(
                aggregate_id=unrelated_source.stable_id,
                aggregate_kind=StableIdentityKind.SOURCE.value,
                event_type=CanonicalEventType.STATUS_CHANGED.value,
                from_state="unknown",
                to_state="verified_usable",
                payload={
                    "schema": "editorial_source_event/v1",
                    "action": "source_availability_recorded",
                    "entry_identity": accepted["entry_identity"],
                    "editorial_revision_identity": accepted["revision_identity"],
                    "maintainer_acceptance_event_id": accepted["maintainer_acceptance"]["event_id"],
                },
                recorded_by=accepted["maintainer_acceptance"]["maintainer_identity"],
            ),
        ]
    )
    await db_session.commit()

    assert await EditorialAuthorityService(db_session)._source_availability(unrelated_source) == "unknown"


def test_needs_review_grace_is_bounded_and_cannot_override_integrity_or_source_failures() -> None:
    now = datetime(2026, 9, 5, 12, tzinfo=UTC)
    eligible = evaluate_answer_eligibility(
        lifecycle_state="needs_re_review",
        approval_status="approved",
        source_availability=["verified_usable"],
        needs_review_at=now - timedelta(days=6, hours=23),
        applicability_explicit=True,
        known_contradiction=False,
        integrity_defect=False,
        decisive_source_loss=False,
        now=now,
    )
    expired = evaluate_answer_eligibility(
        lifecycle_state="needs_re_review",
        approval_status="approved",
        source_availability=["verified_usable"],
        needs_review_at=now - timedelta(days=7, seconds=1),
        applicability_explicit=True,
        known_contradiction=False,
        integrity_defect=False,
        decisive_source_loss=False,
        now=now,
    )
    contradicted = evaluate_answer_eligibility(
        lifecycle_state="needs_re_review",
        approval_status="approved",
        source_availability=["verified_usable"],
        needs_review_at=now - timedelta(days=1),
        applicability_explicit=True,
        known_contradiction=True,
        integrity_defect=False,
        decisive_source_loss=False,
        now=now,
    )
    source_lost = evaluate_answer_eligibility(
        lifecycle_state="needs_re_review",
        approval_status="approved",
        source_availability=["unavailable_for_new_evidence"],
        needs_review_at=now - timedelta(days=1),
        applicability_explicit=True,
        known_contradiction=False,
        integrity_defect=False,
        decisive_source_loss=True,
        now=now,
    )
    missing_sources = evaluate_answer_eligibility(
        lifecycle_state="published",
        approval_status="approved",
        source_availability=[],
        needs_review_at=None,
        applicability_explicit=True,
        known_contradiction=False,
        integrity_defect=False,
        decisive_source_loss=False,
        now=now,
    )

    assert eligible == {"answer_eligible": True, "reasons": ["needs_review_grace_active"]}
    assert expired["answer_eligible"] is False
    assert "needs_review_grace_expired" in expired["reasons"]
    assert contradicted == {"answer_eligible": False, "reasons": ["known_contradiction"]}
    assert source_lost == {"answer_eligible": False, "reasons": ["decisive_source_loss", "source_unavailable"]}
    assert missing_sources == {"answer_eligible": False, "reasons": ["source_availability_missing"]}
