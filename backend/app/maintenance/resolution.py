from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.editorial_authority.service import EditorialAuthorityService
from app.maintenance.diagnosis import current_publication_review, eligible_reference_reviews, evidence_required
from app.maintenance.diagnosis_policy import diagnosis_route
from app.maintenance.findings import load_finding
from app.maintenance.fixtures import load_fixture
from app.maintenance.reacceptance import verify_provider_reactivation, verify_reacceptance
from app.maintenance.replay import approved_fixture, load_replay
from app.maintenance.schemas import ResolutionInput
from app.model.canonical import CanonicalRecordModel
from app.reviewed_bundles.models import PublishedKnowledgePointer
from app.reviewed_bundles.publication import CandidatePublicationService
from app.reviewed_bundles.verifier import CanonicalEditorialExportVerifier


def require_complete_scope(item: dict, fixtures: list[dict]) -> None:
    if not fixtures:
        raise evidence_required()
    for kind, targets in item["affected_scope"].items():
        covered = [target for fixture in fixtures for target in fixture["affected_scope"][kind]]
        if any(target not in covered for target in targets):
            raise evidence_required()


async def verify_confirmation(session: AsyncSession, item: dict, *, current: bool) -> list[str]:
    if item["classification"] != "confirmation" or item["disposition"] != "confirmation" or item["severity"] in {"p0", "p1"}:
        raise evidence_required()
    fixtures = []
    links = []
    for identity in item.get("finding_identities", []):
        finding = await load_finding(session, item, identity)
        if finding["pattern"]["observation"] != "confirmation":
            continue
        fixture = await load_fixture(session, item, finding["fixture_identity"])
        if (
            diagnosis_route(fixture["verified_observation"], fixture["diagnosis_facts"]) != ("confirmation", "confirmation")
            or fixture["expected_outcome"] != "evidence_gated_answer"
            or fixture["observed_outcome"] != "evidence_gated_answer"
            or any(target["publication_identity"] not in fixture["evidence_publication_identities"]
                   for target in fixture["affected_scope"]["entry_versions"])
        ):
            raise evidence_required()
        fixtures.append(fixture)
        links.extend(finding["result_links"])
    require_complete_scope(item, fixtures)
    if current:
        publications = sorted({identity for fixture in fixtures for identity in fixture["evidence_publication_identities"]})
        revisions = set()
        for identity in publications:
            publication = await session.get(CanonicalRecordModel, identity)
            if publication is None or not isinstance(publication.payload.get("editorial_revision_identity"), str):
                raise evidence_required()
            revisions.add(publication.payload["editorial_revision_identity"])
        entries = await EditorialAuthorityService(session).lock_retrieval_authority_for_revisions(revisions)
        await session.execute(
            select(PublishedKnowledgePointer).where(PublishedKnowledgePointer.entry_identity.in_(entries))
            .order_by(PublishedKnowledgePointer.entry_identity).with_for_update()
        )
        await eligible_reference_reviews(session, publications)
    return list(dict.fromkeys(links))


async def verify_content_publication_lineage(session: AsyncSession, current: dict, prior: dict) -> None:
    service = CandidatePublicationService(session, editorial_export_verifier=CanonicalEditorialExportVerifier(session))
    identity = current["publication_identity"]
    visited = set()
    while identity != prior["publication_identity"]:
        if not isinstance(identity, str) or identity in visited:
            raise evidence_required()
        visited.add(identity)
        record = await session.get(CanonicalRecordModel, identity)
        if record is None or not isinstance(record.payload.get("candidate_id"), str):
            raise evidence_required()
        publication = await service.get_publication(record.payload["candidate_id"])
        version = publication["published_knowledge_version"]
        if (
            version is None or version["identity"] != identity or version["entry_identity"] != prior["entry_identity"]
            or (identity == current["publication_identity"] and publication["is_current_for_entry"] is not True)
        ):
            raise evidence_required()
        identity = version["supersedes_published_knowledge_version_identity"]


async def coverage_publication_artifacts(session: AsyncSession, fixture: dict, replay: dict) -> set[str]:
    if (
        fixture["verified_observation"] != "coverage_gap"
        or diagnosis_route(fixture["verified_observation"], fixture["diagnosis_facts"]) != ("coverage-gap", "coverage-work")
        or fixture["expected_outcome"] != "evidence_gated_answer"
        or replay["observed_outcome"] != "evidence_gated_answer"
        or not set(replay["evidence_publication_identities"]) - set(fixture["active_publication_identities"])
    ):
        raise evidence_required()
    service = CandidatePublicationService(session, editorial_export_verifier=CanonicalEditorialExportVerifier(session))
    artifacts = set()
    for identity in replay["evidence_publication_identities"]:
        record = await session.get(CanonicalRecordModel, identity)
        if record is None or not isinstance(record.payload.get("candidate_id"), str):
            raise evidence_required()
        publication = await service.get_publication(record.payload["candidate_id"])
        version = publication["published_knowledge_version"]
        if version is None or version["identity"] != identity or publication["is_current_for_entry"] is not True:
            raise evidence_required()
        current = await current_publication_review(session, version["entry_identity"])
        authority = await EditorialAuthorityService(session).get_retrieval_authority_for_revision(
            current["entry_identity"].removeprefix("entry:"), current["revision_identity"],
        )
        if current["publication_identity"] != identity or authority["answer_eligible"] is not True:
            raise evidence_required()
        artifacts.update((identity, current["revision_identity"]))
    return artifacts


async def verify_resolution(session: AsyncSession, item: dict, payload: ResolutionInput) -> list[str]:
    if item["state"] != "in_progress" or not item.get("finding_identities"):
        raise evidence_required()
    if len(set(payload.artifact_identities)) != len(payload.artifact_identities):
        raise evidence_required()
    if payload.disposition == "boundary-query" and (
        item["classification"] != "coverage-gap" or item["disposition"] != "coverage-work"
    ):
        raise evidence_required()
    if payload.disposition == "source-change" and (
        item["classification"] != "source-freshness" or item["disposition"] != "source-change"
    ):
        raise evidence_required()
    if payload.disposition == "provider-work" and (
        item["classification"] != "product-privacy-operations" or item["disposition"] != "provider-work"
    ):
        raise evidence_required()
    if payload.disposition == "retrieval-experiment" and (
        item["classification"] != "retrieval-answer-behavior" or item["disposition"] != "retrieval-experiment"
    ):
        raise evidence_required()
    if payload.disposition == "product-repair" and (
        item["classification"] != "product-privacy-operations" or item["disposition"] != "product-repair"
    ):
        raise evidence_required()
    if payload.disposition == "entry-revision" and (item["classification"], item["disposition"]) not in {
        ("content-integrity", "entry-revision"), ("coverage-gap", "coverage-work"),
    }:
        raise evidence_required()
    fixtures: dict[str, dict] = {}
    sources = set()
    replayed_fixtures = set()
    provider_replays = []
    provider_artifacts = set()
    content_artifacts = set()
    reacceptance_records = []
    links = []
    for identity in payload.artifact_identities:
        record = await session.get(CanonicalRecordModel, identity)
        if record is None:
            raise evidence_required()
        if record.payload.get("schema") == "maintenance_fixture/v1":
            fixture = await approved_fixture(session, item, identity)
            if payload.disposition == "boundary-query" and (
                fixture["expected_outcome"] != "insufficient_evidence_reply"
                or fixture["observed_outcome"] != "insufficient_evidence_reply"
                or diagnosis_route(fixture["verified_observation"], fixture["diagnosis_facts"]) != (
                    item["classification"], item["disposition"]
                )
            ):
                raise evidence_required()
            fixtures[identity] = fixture
        elif record.payload.get("schema") == "maintenance_replay/v1":
            replay = await load_replay(session, item, identity, current=True)
            if not replay["passed"]:
                raise evidence_required()
            replayed_fixtures.add(replay["fixture_identity"])
            provider_replays.append(replay)
        elif record.payload.get("schema") == "delivery_acceptance_record/v1":
            if payload.disposition == "provider-work" and identity != item.get("containment_record_identity"):
                provider_artifacts.add(identity)
                if item["severity"] in {"p0", "p1"}:
                    reacceptance_records.append(identity)
                continue
            reacceptance_records.append(identity)
        elif record.identity_kind == "source" and payload.disposition == "source-change":
            sources.add(identity)
            continue
        elif record.identity_kind == "provider_route" and payload.disposition == "provider-work":
            provider_artifacts.add(identity)
            continue
        elif payload.disposition == "entry-revision" and record.identity_kind in {"editorial_revision", "published_knowledge_version"}:
            content_artifacts.add(identity)
        else:
            raise evidence_required()
        links.append(f"evidence://maintenance/artifacts/{identity}")
    if not fixtures or set(fixtures) != replayed_fixtures:
        raise evidence_required()
    require_complete_scope(item, list(fixtures.values()))
    if item["severity"] in {"p0", "p1"}:
        links.append(await verify_reacceptance(session, item, reacceptance_records, provider_replays))
    elif reacceptance_records:
        raise evidence_required()
    if payload.disposition == "entry-revision":
        expected_artifacts = set()
        for replay in provider_replays:
            fixture = fixtures.get(replay["fixture_identity"])
            if item["classification"] == "coverage-gap":
                if fixture is None:
                    raise evidence_required()
                expected_artifacts.update(await coverage_publication_artifacts(session, fixture, replay))
                continue
            prior = fixture["publication_review"] if fixture else None
            current = replay["publication_review"]
            if (
                fixture is None or fixture["verified_observation"] != "wrong_content"
                or prior is None or not prior.get("integrity_review_event_id") or current is None
                or fixture["expected_outcome"] != "evidence_gated_answer"
                or replay["observed_outcome"] != "evidence_gated_answer"
                or current["entry_identity"] != prior["entry_identity"]
                or current["revision_identity"] == prior["revision_identity"]
                or current["publication_identity"] == prior["publication_identity"]
                or current.get("integrity_review_event_id") is not None
                or current["publication_identity"] not in replay["evidence_publication_identities"]
            ):
                raise evidence_required()
            revision = await session.get(CanonicalRecordModel, current["revision_identity"])
            publication = await session.get(CanonicalRecordModel, current["publication_identity"])
            if (
                revision is None or revision.record_class != "authoritative"
                or revision.payload.get("schema") != "editorial_revision/v1"
                or revision.payload.get("entry_identity") != prior["entry_identity"]
                or revision.payload.get("change_kind") != "material"
                or publication is None or not isinstance(publication.payload.get("candidate_id"), str)
                or publication.payload.get("editorial_revision_identity") != revision.stable_id
            ):
                raise evidence_required()
            await verify_content_publication_lineage(session, current, prior)
            authority = await EditorialAuthorityService(session).get_retrieval_authority_for_revision(
                current["entry_identity"].removeprefix("entry:"), revision.stable_id,
            )
            if (
                authority["answer_eligible"] is not True
            ):
                raise evidence_required()
            expected_artifacts.update((revision.stable_id, publication.stable_id))
        if not expected_artifacts or content_artifacts != expected_artifacts:
            raise evidence_required()
    if payload.disposition in {"retrieval-experiment", "product-repair"}:
        observations = (
            {"product_failure"} if payload.disposition == "product-repair"
            else {"citation_drift", "retrieval_miss", "condition_loss"}
        )
        for replay in provider_replays:
            fixture = fixtures.get(replay["fixture_identity"])
            if (
                fixture is None
                or fixture["verified_observation"] not in observations
                or fixture["expected_outcome"] != "evidence_gated_answer"
                or replay["observed_outcome"] != "evidence_gated_answer"
                or set(replay["evidence_publication_identities"]) != set(fixture["reference_evidence_publication_identities"])
            ):
                raise evidence_required()
            if diagnosis_route(fixture["verified_observation"], fixture["diagnosis_facts"]) != (
                item["classification"], payload.disposition
            ):
                raise evidence_required()
    if payload.disposition == "source-change":
        repaired_sources = set()
        for fixture in fixtures.values():
            prior = fixture["publication_review"]
            if (
                fixture["verified_observation"] != "stale_source"
                or fixture["expected_outcome"] != "evidence_gated_answer"
                or prior is None
            ):
                raise evidence_required()
            current = await current_publication_review(session, prior["entry_identity"])
            if (
                current["publication_identity"] != prior["publication_identity"]
                or current["revision_identity"] != prior["revision_identity"]
                or current["source_states"] != ["verified_usable"]
            ):
                raise evidence_required()
            current_facts = {fact["source_identity"]: fact for fact in current["source_facts"]}
            for fact in prior["source_facts"]:
                if fact["availability"] == "verified_usable":
                    continue
                repaired = current_facts.get(fact["source_identity"])
                if repaired is None or repaired["status_event_id"] == fact["status_event_id"]:
                    raise evidence_required()
                repaired_sources.add(fact["source_identity"])
                links.append(f"evidence://maintenance/artifacts/{fact['source_identity']}:{repaired['status_event_id']}")
        if not repaired_sources or sources != repaired_sources:
            raise evidence_required()
    if payload.disposition == "provider-work":
        expected_artifacts = set()
        for replay in provider_replays:
            fixture = fixtures.get(replay["fixture_identity"])
            context = replay["generation_context"]
            if (
                fixture is None
                or fixture["verified_observation"] != "provider_failure"
                or fixture["observed_outcome"] != "generation_unavailable"
                or fixture["generation_context"] is None
                or replay["observed_outcome"] != "evidence_gated_answer"
                or context is None
            ):
                raise evidence_required()
            if item["severity"] in {"p0", "p1"}:
                context = await verify_provider_reactivation(session, item, replay, reacceptance_records[0])
            expected_artifacts.update((context["route_identity"], context["acceptance_record_identity"]))
            links.extend(
                (
                    f"evidence://maintenance/artifacts/{context['route_identity']}:{context['activation_event_id']}",
                    f"evidence://maintenance/artifacts/{context['acceptance_record_identity']}",
                )
            )
        if provider_artifacts != expected_artifacts:
            raise evidence_required()
    return links
