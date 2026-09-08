from __future__ import annotations

import asyncio
import threading

import pytest
from sqlalchemy import select

from app.common.config import get_settings
from app.editorial_authority.service import EditorialAuthorityService
from app.model.user import User
from app.rag.answer_execution import AnswerExecutionOutcome, AnswerOutcomeKind
from app.rag.evidence_sufficiency import decide_answer_evidence
from app.repository.chat_repository import ChatRepository
from app.retrieval.candidate_pool import AuthorizedRetrievalCandidatePool
from app.service.answer_execution_store import AnswerExecutionStore
from app.service.identity_audit_service import IdentityAuditService
from tests.browser_acceptance_api import (
    _browser_ticket24_entry,
    _build_ticket24_candidate,
    _ticket24_manifest,
)
from tests.integration.test_ticket24_candidate_publication_api import (
    _headers,
    _inspect_and_accept,
    _publish_selection,
)
from tests.integration.test_ticket24_candidate_publication_api import client as client


async def _reviewed_candidate(client, headers, *, entry_id="ticket25-entry"):
    del headers
    factory = client.app.state.test_auth_session_factory
    async with factory() as session:
        publisher = await session.scalar(select(User).where(User.username == "ticket25-admin"))
        assert publisher is not None
        author = User(username=f"{entry_id}-author", password_hash="fixture", role="user", is_active=True)
        reviewer = User(username=f"{entry_id}-reviewer", password_hash="fixture", role="user", is_active=True)
        maintainer = User(username=f"{entry_id}-maintainer", password_hash="fixture", role="user", is_active=True)
        session.add_all([author, reviewer, maintainer])
        await session.flush()
        entry = _browser_ticket24_entry(entry_id).model_copy(update={
            "approving_reviewer_username": reviewer.username,
            "accountable_maintainer_username": maintainer.username,
        })
        authority = EditorialAuthorityService(session)
        await authority.create_draft(entry, author)
        await authority.collect_evidence(entry_id, author)
        await authority.accept_maintainer_responsibility(entry_id, maintainer)
        await authority.record_source_availability(entry_id, entry.sources[0]["source_id"], "verified_usable", maintainer)
        await authority.request_editorial_review(entry_id, author)
        await authority.approve_current_revision(entry_id, reviewer)
        exported = await authority.export_approved_revision(entry_id, publisher)
        actor_identity = await IdentityAuditService(session).ensure_member_record(publisher, admission_path="test")
        return await _build_ticket24_candidate(
            session, artifact=exported["artifact"], bundle_id=f"{entry_id}-bundle",
            operation="create", actor_identity=actor_identity,
        )


def _publish(client, headers, *, entry_id="ticket25-entry"):
    candidate = asyncio.run(_reviewed_candidate(client, headers, entry_id=entry_id))
    _inspect_and_accept(client, candidate_id=candidate, headers=headers)
    eligible = client.get(
        f"/api/v1/reviewed-release-bundles/candidates/{candidate}/publication-eligibility", headers=headers,
    ).json()["data"]
    response = client.post("/api/v1/reviewed-release-bundles/publication-batches", headers=headers, json={
        "confirmation_id": f"{entry_id}-confirmation",
        "selected_items": [_publish_selection(candidate, eligible)],
    })
    assert response.status_code == 200, response.text
    result = response.json()["data"]
    assert result["batch_complete"], result
    return result["published"][0]["publication_identity"]


def test_withdrawal_retains_exact_published_identity_and_original_audit(client):
    headers = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, headers)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    payload = {"reason_code": "integrity_defect", "trigger": "integrity"}
    response = client.post(url, headers=headers, json=payload)
    assert response.status_code == 200, response.text
    record = response.json()["data"]
    assert record["publication_identity"] == publication
    assert record["entry_identity"] == "entry:ticket25-entry"
    assert record["state"] == "withdrawn"
    assert record["reason_code"] == "integrity_defect"
    assert record["trigger"] == "integrity"
    assert record["actor_identity"].startswith("member:")
    assert record["occurred_at"]
    assert record["event_identity"]
    assert record["affected_scope"] == {"scope": "entry_version", "identity": publication}
    replay = client.post(url, headers=headers, json=payload)
    assert replay.status_code == 200, replay.text
    assert replay.json()["data"] == record
    read = client.get(url, headers=headers)
    assert read.status_code == 200
    assert read.json()["data"] == record


@pytest.mark.parametrize("with_rejection", [False, True])
def test_withdrawal_completes_successor_bundle_lifecycle(client, with_rejection):
    from app.common.canonical_json import canonical_json_sha256

    headers = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, headers)

    async def export():
        async with client.app.state.test_auth_session_factory() as session:
            admin = await session.scalar(select(User).where(User.username == "ticket25-admin"))
            return await EditorialAuthorityService(session).export_approved_revision("ticket25-entry", admin)

    artifact = asyncio.run(export())["artifact"]
    manifest = _ticket24_manifest(bundle_id="ticket25-lifecycle", artifact=artifact, operation="replace")
    if with_rejection:
        rejected = _ticket24_manifest(
            bundle_id="ticket25-lifecycle-rejected",
            artifact={**artifact, "entry_identity": "entry:ticket25-unapproved"}, operation="create",
        )
        manifest["items"].extend(rejected["items"])
        manifest["bundle_sha256"] = canonical_json_sha256({
            key: value for key, value in manifest.items() if key != "bundle_sha256"
        })
    imported = client.post("/api/v1/reviewed-release-bundles/import", headers=headers, json=manifest)
    assert imported.status_code == 200, imported.text
    assert imported.json()["data"]["state"] == "processing"
    assert [item["state"] for item in imported.json()["data"]["items"]] == (
        ["admitted", "rejected"] if with_rejection else ["admitted"]
    )
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    for _ in range(2):
        assert client.post(url, headers=headers, json={
            "reason_code": "integrity_defect", "trigger": "integrity",
        }).status_code == 200
        bundle = client.get("/api/v1/reviewed-release-bundles/ticket25-lifecycle", headers=headers)
        assert bundle.json()["data"]["state"] == ("completed_with_rejections" if with_rejection else "completed")
        assert client.post(f"{url}/reconciliation", headers=headers).json()["data"]["state"] == "completed"


@pytest.mark.parametrize("cleanup_first", [False, True])
def test_withdrawal_rejects_new_build_admission_for_retained_export(client, cleanup_first):
    headers = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, headers)
    _publish(client, headers, entry_id="ticket25-unrelated")

    async def export(entry_id):
        async with client.app.state.test_auth_session_factory() as session:
            admin = await session.scalar(select(User).where(User.username == "ticket25-admin"))
            return await EditorialAuthorityService(session).export_approved_revision(entry_id, admin)

    artifact = asyncio.run(export("ticket25-entry"))["artifact"]
    unrelated = asyncio.run(export("ticket25-unrelated"))["artifact"]
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    payload = {"reason_code": "integrity_defect", "trigger": "integrity"}
    original = client.post(url, headers=headers, json=payload).json()["data"]
    if cleanup_first:
        assert client.post(f"{url}/reconciliation", headers=headers).json()["data"]["state"] == "completed"
    imported = client.post("/api/v1/reviewed-release-bundles/import", headers=headers, json=_ticket24_manifest(
        bundle_id="ticket25-post-withdrawal", artifact=artifact, operation="replace",
    ))
    assert imported.status_code == 200, imported.text
    item = imported.json()["data"]["items"][0]
    assert item["state"] == "rejected"
    assert "job_id" not in item
    assert item["failure_reason"]["code"] == "ENTRY_WITHDRAWN"
    sibling = client.post("/api/v1/reviewed-release-bundles/import", headers=headers, json=_ticket24_manifest(
        bundle_id="ticket25-post-withdrawal-unrelated", artifact=unrelated, operation="replace",
    ))
    assert sibling.status_code == 200, sibling.text
    assert sibling.json()["data"]["items"][0]["state"] == "admitted"
    assert client.post(url, headers=headers, json=payload).json()["data"] == original
    for _ in range(2):
        assert client.post(f"{url}/reconciliation", headers=headers).json()["data"]["state"] == "completed"


@pytest.mark.parametrize("canceled_successor", [False, True])
def test_withdrawal_invalidates_successor_jobs_and_excludes_only_affected_entry(client, monkeypatch, canceled_successor):
    headers = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, headers)
    _publish(client, headers, entry_id="ticket25-unrelated")

    async def export():
        async with client.app.state.test_auth_session_factory() as session:
            admin = await session.scalar(select(User).where(User.username == "ticket25-admin"))
            return await EditorialAuthorityService(session).export_approved_revision("ticket25-entry", admin)

    artifact = asyncio.run(export())["artifact"]
    imported = client.post("/api/v1/reviewed-release-bundles/import", headers=headers, json=_ticket24_manifest(
        bundle_id="ticket25-successor", artifact=artifact, operation="replace",
    ))
    assert imported.status_code == 200, imported.text
    job_id = imported.json()["data"]["items"][0]["job_id"]
    if canceled_successor:
        from app.documents.dense_index_service import DenseIndexService
        from app.reviewed_bundles.runtime import candidate_build_runtime

        async def unavailable(self, **kwargs):
            raise OSError("successor vector cleanup unavailable")

        async def reached_worker(job_id):
            return True

        monkeypatch.setattr(candidate_build_runtime, "cancel", reached_worker)
        monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", unavailable)
        canceled = client.post(f"/api/v1/reviewed-release-bundles/jobs/{job_id}/cancel", headers=headers)
        assert canceled.status_code == 200, canceled.text
        assert canceled.json()["data"]["status"] == "canceled"
        assert canceled.json()["data"]["derived_cleanup_pending"]
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=headers, json={"reason_code": "integrity_defect", "trigger": "integrity"},
    )
    assert response.status_code == 200, response.text
    job = client.get(f"/api/v1/reviewed-release-bundles/jobs/{job_id}", headers=headers).json()["data"]
    assert job["status"] == "superseded"
    assert job["allowed_next_action"] == "none"
    assert job["derived_cleanup_pending"]
    assert job["failure_reason"]["code"] == "PUBLICATION_WITHDRAWN"

    async def pool():
        async with client.app.state.test_auth_session_factory() as session:
            return await AuthorizedRetrievalCandidatePool(session, settings=get_settings()).retrieve("Candidate publication contract", 20)

    candidates = asyncio.run(pool())
    assert candidates.items
    assert {item["entry_identity"] for item in candidates.items} == {"entry:ticket25-unrelated"}
    if canceled_successor:
        async def successor_unavailable(self, **kwargs):
            if kwargs["generation"] > 1:
                raise OSError("successor vector cleanup unavailable")

        monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", successor_unavailable)
        failed_cleanup = client.post(
            f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal/reconciliation", headers=headers,
        )
        assert failed_cleanup.json()["data"]["state"] == "suspended"

        async def recovered(self, **kwargs):
            pass

        monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", recovered)
    cleanup = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal/reconciliation", headers=headers,
    )
    assert cleanup.status_code == 200, cleanup.text
    assert cleanup.json()["data"]["state"] == "completed"
    reconciled = client.get(f"/api/v1/reviewed-release-bundles/jobs/{job_id}", headers=headers).json()["data"]
    assert reconciled["derived_cleanup_pending"] is False
    assert reconciled["status"] == "superseded"
    assert reconciled["allowed_next_action"] == "none"


async def _frozen_execution(client, *, session_id="ticket25-history", complete=True):
    async with client.app.state.test_auth_session_factory() as session:
        repository = ChatRepository(session)
        await repository.get_or_create_session(session_id, "ticket25-reader")
        store = AnswerExecutionStore(session, repository)
        question = "Which Candidate publication contract applies?"
        resolution = await store.resolve_query_conditions(
            user_id="ticket25-reader", session_id=session_id, question=question, inherit_conditions=False,
            explicit_conditions=[{
                "condition_id": "ticket25-production", "field": "deployment", "operator": "equals", "value": "production",
            }],
        )
        handle = await store.admit(
            request_id=session_id, user_id="ticket25-reader", session_id=session_id,
            question=question, resolution=resolution,
        )
        pool = await AuthorizedRetrievalCandidatePool(session, settings=get_settings()).retrieve(question, 20)
        decision = decide_answer_evidence(
            normalized_question=question, query_conditions=handle.query_conditions, candidates=pool.items,
        )
        assert decision.evidence_set is not None, decision
        outcome = AnswerExecutionOutcome(
            kind=AnswerOutcomeKind.EVIDENCE_GATED_ANSWER, text="An inspected and explicitly confirmed Candidate is required. [1]",
            evidence=decision.evidence_set.items, question=question, query_conditions=handle.query_conditions,
            request_id=handle.request_id, session_id=handle.session_id, gate_passed=True,
            gate_reason="sufficient_evidence", steps=(), runtime={},
            evidence_set=decision.evidence_set, sufficiency_decision=decision,
        )
        loaded = None
        if complete:
            _, loaded = await store.complete(handle=handle, outcome=outcome)
        await session.commit()
        return handle, outcome, loaded


def test_withdrawal_redacts_private_history_and_preserves_frozen_identity(client):
    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    reader = asyncio.run(_headers(client, username="ticket25-reader", role="user"))
    publication = _publish(client, admin)
    _, _, original = asyncio.run(_frozen_execution(client))
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=admin, json={"reason_code": "privacy_defect", "trigger": "integrity"},
    )
    assert response.status_code == 200, response.text
    withdrawal = response.json()["data"]

    async def reload():
        async with client.app.state.test_auth_session_factory() as session:
            return await AnswerExecutionStore(session, ChatRepository(session)).load(
                execution_id=original.projection["id"], user_id="ticket25-reader",
            )

    loaded = asyncio.run(reload())
    for key in ("evidence_set_identity", "item_identities", "snapshot_ids", "knowledge_version_identities"):
        assert loaded.result[key] == original.result[key]
    source = loaded.projection["evidence_summary"]["sources"][0]
    assert source["withdrawal_notice"] == "This source has been withdrawn."
    assert source["withdrawal"] == withdrawal
    assert "excerpt" not in source
    assert "source_url" not in source
    assert "controlled_locator" not in source
    history = client.get("/api/v1/sessions/ticket25-history", headers=reader)
    assert history.status_code == 200, history.text
    messages = history.json()["data"]["messages"]
    assert messages[-1]["evidence_summary"] == loaded.projection["evidence_summary"]
    assert client.get("/api/v1/sessions/ticket25-history", headers=admin).json()["data"]["messages"] == []
    other = asyncio.run(_headers(client, username="ticket25-other-reader", role="user"))
    assert client.get("/api/v1/sessions/ticket25-history", headers=other).json()["data"]["messages"] == []


@pytest.mark.parametrize("restored_compatibility_projection", [False, True])
def test_withdrawal_before_final_completion_rejects_frozen_supported_answer(client, restored_compatibility_projection):
    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    handle, outcome, _ = asyncio.run(_frozen_execution(client, complete=False))
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=admin, json={"reason_code": "integrity_defect", "trigger": "integrity"},
    )
    assert response.status_code == 200, response.text
    if restored_compatibility_projection:
        from app.model.document import Document

        async def legacy_recovery_fixture():
            async with client.app.state.test_auth_session_factory() as session:
                document = await session.get(Document, response.json()["data"]["document_identity"])
                document.deleted_at = None
                await session.commit()

        asyncio.run(legacy_recovery_fixture())

    async def finish():
        async with client.app.state.test_auth_session_factory() as session:
            _, loaded = await AnswerExecutionStore(session, ChatRepository(session)).complete(handle=handle, outcome=outcome)
            await session.commit()
            return loaded

    loaded = asyncio.run(finish())
    assert loaded.projection["state"] == "failed"
    assert loaded.projection.get("outcome") is None
    assert loaded.projection.get("answer_text") in (None, "")
    assert loaded.projection.get("evidence_summary") is None


@pytest.mark.parametrize("transport", ["normal", "sse"])
def test_completed_response_refreshes_withdrawal_at_the_final_projection(client, monkeypatch, transport):
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.rag.answer_execution import EvidenceGatedAnswerExecutor
    from tests.integration.test_ticket20_answer_execution_persistence import _sse_event_data

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    reader = asyncio.run(_headers(client, username="ticket25-reader", role="user"))
    publication = _publish(client, admin)
    question = "Which Candidate publication contract applies?"
    original_complete = AnswerExecutionStore.complete
    original_commit = AsyncSession.commit
    frozen = {}
    withdrawals = []

    async def execute(self, **kwargs):
        async with client.app.state.test_auth_session_factory() as session:
            pool = await AuthorizedRetrievalCandidatePool(session, settings=get_settings()).retrieve(question, 20)
        decision = decide_answer_evidence(
            normalized_question=question, query_conditions=kwargs["query_conditions"], candidates=pool.items,
        )
        assert decision.evidence_set is not None
        return AnswerExecutionOutcome(
            kind=AnswerOutcomeKind.EVIDENCE_GATED_ANSWER,
            text="An inspected and explicitly confirmed Candidate is required. [1]",
            evidence=decision.evidence_set.items, question=question, query_conditions=kwargs["query_conditions"],
            request_id=kwargs["request_id"], session_id=kwargs["session_id"], gate_passed=True,
            gate_reason="sufficient_evidence", steps=(), runtime={},
            evidence_set=decision.evidence_set, sufficiency_decision=decision,
        )

    async def complete(store, **kwargs):
        result = await original_complete(store, **kwargs)
        frozen.update(result[1].projection)
        store.session.info["ticket25_completed_commit"] = True
        return result

    async def commit(session):
        await original_commit(session)
        if session.info.pop("ticket25_completed_commit", False):
            withdrawn = await asyncio.to_thread(
                client.post, f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
                headers=admin, json={"reason_code": "privacy_defect", "trigger": "integrity"},
            )
            assert withdrawn.status_code == 200, withdrawn.text
            withdrawals.append(withdrawn.json()["data"])

    monkeypatch.setattr(EvidenceGatedAnswerExecutor, "execute", execute)
    monkeypatch.setattr(AnswerExecutionStore, "complete", complete)
    monkeypatch.setattr(AsyncSession, "commit", commit)
    response = client.post("/api/v1/chat/stream" if transport == "sse" else "/api/v1/chat", headers=reader, json={
        "message": question, "session_id": "ticket25-final-response",
        "query_conditions": [{
            "condition_id": "ticket25-production", "field": "deployment", "operator": "equals", "value": "production",
        }],
    })
    assert response.status_code == 200, response.text
    projected = (
        _sse_event_data(response.text, "answer_execution")["answer_execution"]
        if transport == "sse" else response.json()["data"]["answer_execution"]
    )
    assert len(withdrawals) == 1
    assert projected["state"] == frozen["state"] == "completed"
    assert projected["outcome"] == frozen["outcome"]
    for key in ("evidence_set_identity", "item_identities", "snapshot_ids", "knowledge_version_identities"):
        assert projected[key] == frozen[key]
    source = projected["evidence_summary"]["sources"][0]
    assert source["withdrawal"]["event_identity"] == withdrawals[0]["event_identity"]
    assert "excerpt" not in source
    assert not source.get("source_url")
    history = client.get("/api/v1/sessions/ticket25-final-response", headers=reader)
    assert history.status_code == 200, history.text
    assert history.json()["data"]["messages"][-1]["answer_execution"] == projected


def test_redaction_failure_retains_containment_and_scoped_reconciliation(client, monkeypatch):
    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    reader = asyncio.run(_headers(client, username="ticket25-reader", role="user"))
    publication = _publish(client, admin)
    asyncio.run(_frozen_execution(client))

    async def unavailable(*args, **kwargs):
        raise OSError("private redaction storage unavailable")

    monkeypatch.setattr(AnswerExecutionStore, "redact_document_evidence", unavailable)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    response = client.post(url, headers=admin, json={"reason_code": "privacy_defect", "trigger": "integrity"})
    assert response.status_code == 200, response.text
    state = client.get(f"{url}/reconciliation", headers=admin)
    assert state.status_code == 200, state.text
    assert state.json()["data"]["state"] == "suspended"
    assert state.json()["data"]["affected_scope"] == {"scope": "entry_version", "identity": publication}
    assert state.json()["data"]["allowed_next_action"] == "retry_reconciliation"
    history = client.get("/api/v1/sessions/ticket25-history", headers=reader)
    assert history.status_code == 200
    source = history.json()["data"]["messages"][-1]["evidence_summary"]["sources"][0]
    assert "excerpt" not in source
    assert source["withdrawal_notice"]


@pytest.mark.parametrize("field,value", [
    ("document_identity", "runtime-document:foreign"),
    ("entry_identity", None),
    ("generation", True),
    ("reason_code", "unbounded-reason"),
    ("supersedes_version_id", "published_knowledge_version:foreign"),
])
def test_malformed_withdrawal_fact_fails_closed_on_history_and_admin_reads(client, monkeypatch, field, value):
    from app.model.canonical import CanonicalEventModel

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    reader = asyncio.run(_headers(client, username="ticket25-reader", role="user"))
    publication = _publish(client, admin)
    asyncio.run(_frozen_execution(client))

    async def unavailable(*args, **kwargs):
        raise OSError("private redaction storage unavailable")

    monkeypatch.setattr(AnswerExecutionStore, "redact_document_evidence", unavailable)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    response = client.post(url, headers=admin, json={"reason_code": "privacy_defect", "trigger": "integrity"})
    assert response.status_code == 200, response.text
    record = response.json()["data"]

    async def corrupt_fixture():
        payload = {**record, field: value}
        if value is None:
            payload.pop(field)
        async with client.app.state.test_auth_session_factory() as session:
            table = CanonicalEventModel.__table__
            await session.execute(table.update().where(
                table.c.id == record["event_identity"].removeprefix("event:"),
            ).values(payload=payload))
            await session.commit()

    asyncio.run(corrupt_fixture())
    for read_url, headers in ((url, admin), ("/api/v1/sessions/ticket25-history", reader)):
        denied = client.get(read_url, headers=headers)
        assert denied.status_code == 409, denied.text
        assert denied.json()["code"] == "WITHDRAWAL_FACT_INVALID"
        assert "excerpt" not in denied.text


def test_reconciliation_retries_failed_cleanup_without_reversing_withdrawal(client, monkeypatch):
    from app.documents.dense_index_service import DenseIndexService

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    record = client.post(url, headers=admin, json={
        "reason_code": "integrity_defect", "trigger": "integrity",
    }).json()["data"]
    calls = []

    async def unavailable(self, **kwargs):
        calls.append(kwargs)
        raise OSError("private vector backend detail")

    monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", unavailable)
    failed = client.post(f"{url}/reconciliation", headers=admin)
    assert failed.status_code == 200, failed.text
    assert failed.json()["data"]["state"] == "suspended"
    assert "private vector backend" not in failed.text

    async def recovered(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", recovered)
    success = client.post(f"{url}/reconciliation", headers=admin)
    assert success.status_code == 200, success.text
    assert success.json()["data"]["state"] == "completed"
    assert success.json()["data"]["allowed_next_action"] == "none"
    assert calls[0] == calls[1]
    assert calls[0]["generation"] == record["generation"]
    assert client.get(url, headers=admin).json()["data"] == record
    assert client.post(f"{url}/reconciliation", headers=admin).json()["data"] == success.json()["data"]


@pytest.mark.parametrize("corruption", ["withdrawal_identity", "scope", "actor", "unproven_completion"])
def test_reconciliation_rejects_malformed_or_unproven_terminal_facts(client, corruption):
    from app.model.canonical import CanonicalEventModel

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    assert client.post(url, headers=admin, json={
        "reason_code": "integrity_defect", "trigger": "integrity",
    }).status_code == 200

    async def corrupt_fixture():
        async with client.app.state.test_auth_session_factory() as session:
            event = await session.scalar(select(CanonicalEventModel).where(
                CanonicalEventModel.aggregate_id == publication,
                CanonicalEventModel.event_type == "status_changed",
            ))
            payload = dict(event.payload)
            updates = {}
            if corruption == "withdrawal_identity":
                payload["withdrawal_event_identity"] = "event:foreign"
            elif corruption == "scope":
                payload.pop("affected_scope")
            elif corruption == "actor":
                updates["recorded_by"] = "member:foreign"
            else:
                payload.update(state="completed", reason_code="derived_cleanup_completed", allowed_next_action="none")
                updates["to_state"] = "completed"
            table = CanonicalEventModel.__table__
            await session.execute(table.update().where(table.c.id == event.id).values(payload=payload, **updates))
            await session.commit()

    asyncio.run(corrupt_fixture())
    for method in ("get", "post"):
        response = getattr(client, method)(f"{url}/reconciliation", headers=admin)
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "WITHDRAWAL_RECONCILIATION_INVALID"


@pytest.mark.parametrize("changed_fingerprint", [False, True])
def test_withdrawal_reconciles_backfilled_runtime_vectors_before_completion(client, monkeypatch, changed_fingerprint):
    from app.documents.dense_index_service import DenseIndexResult, DenseIndexService
    from app.documents.dense_maintenance_service import DenseMaintenanceService
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from tests.unit.test_dense_maintenance_service import _dense_settings

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    _publish(client, admin, entry_id="ticket25-unrelated")
    settings = _dense_settings()
    fingerprint = build_embedding_contract_fingerprint(settings)
    vectors = set()
    failing = True

    async def index(self, *, document_id, generation, chunks, **kwargs):
        assert chunks
        indexed_fingerprint = build_embedding_contract_fingerprint(self._settings)
        vectors.add((document_id, generation, indexed_fingerprint))
        return DenseIndexResult(active=True, fingerprint=indexed_fingerprint)

    async def delete(self, *, document_id, generation, embedding_fingerprint):
        if document_id == "runtime-document:ticket25-entry" and failing:
            raise OSError("runtime vector backend unavailable")
        vectors.discard((document_id, generation, embedding_fingerprint))

    monkeypatch.setattr(DenseIndexService, "index_candidate_generation", index)
    monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", delete)

    async def backfill():
        async with client.app.state.test_auth_session_factory() as session:
            return await DenseMaintenanceService(settings=settings).backfill_published_documents(
                session=session, limit=10,
            )

    assert asyncio.run(backfill()).indexed_documents == 2
    if changed_fingerprint:
        settings = _dense_settings(EMBEDDING_MODEL="text-embedding-3-small")
        fingerprint = build_embedding_contract_fingerprint(settings)
        blocked_switch = asyncio.run(backfill())
        assert blocked_switch.failed_documents == 1
        assert blocked_switch.indexed_documents == 1
        failing = False
        assert asyncio.run(backfill()).indexed_documents == 1
        failing = True
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    withdrawal = client.post(url, headers=admin, json={"reason_code": "integrity_defect", "trigger": "integrity"})
    assert withdrawal.status_code == 200, withdrawal.text
    failed = client.post(f"{url}/reconciliation", headers=admin)
    assert failed.json()["data"]["state"] == "suspended"
    failing = False
    completed = client.post(f"{url}/reconciliation", headers=admin)
    assert completed.json()["data"]["state"] == "completed"
    assert vectors == {("runtime-document:ticket25-unrelated", 1, fingerprint)}
    assert asyncio.run(backfill()).indexed_documents == 0

    async def restore_legacy_projection():
        from app.model.document import Document

        async with client.app.state.test_auth_session_factory() as session:
            document = await session.get(Document, "runtime-document:ticket25-entry")
            document.deleted_at = None
            await session.commit()

    asyncio.run(restore_legacy_projection())
    recovery = asyncio.run(backfill())
    assert recovery.failed_documents == 0
    assert recovery.indexed_documents == 0
    assert [(item.outcome, item.reason) for item in recovery.documents] == [("skipped", "publication_withdrawn")]


@pytest.mark.parametrize("failure", ["response_lost", "commit_failed", "cancelled", "unsettled"])
def test_withdrawal_cleans_runtime_writes_even_without_readiness_commit(client, monkeypatch, failure):
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.documents.dense_index_service import DenseIndexResult, DenseIndexService
    from app.documents.dense_maintenance_service import DenseMaintenanceService
    from app.infra.milvus_document_index import MilvusUpsertCancelledAfterDrain
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from tests.unit.test_dense_maintenance_service import _dense_settings

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    settings = _dense_settings()
    fingerprint = build_embedding_contract_fingerprint(settings)
    vectors = set()
    written = False
    original_commit = AsyncSession.commit

    async def index(self, *, document_id, generation, chunks, **kwargs):
        nonlocal written
        vectors.add((document_id, generation, fingerprint))
        written = True
        if failure == "response_lost":
            raise OSError("upsert response lost after effect")
        if failure == "cancelled":
            raise MilvusUpsertCancelledAfterDrain()
        if failure == "unsettled":
            raise asyncio.CancelledError()
        return DenseIndexResult(active=True, fingerprint=fingerprint)

    async def commit(session):
        nonlocal written
        if written and failure == "commit_failed":
            written = False
            raise OSError("readiness commit unavailable")
        await original_commit(session)

    async def delete(self, *, document_id, generation, embedding_fingerprint):
        vectors.discard((document_id, generation, embedding_fingerprint))

    monkeypatch.setattr(DenseIndexService, "index_candidate_generation", index)
    monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", delete)
    monkeypatch.setattr(AsyncSession, "commit", commit)

    async def backfill():
        async with client.app.state.test_auth_session_factory() as session:
            return await DenseMaintenanceService(settings=settings).backfill_published_documents(session=session, limit=10)

    if failure in {"cancelled", "unsettled"}:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(backfill())
    else:
        assert asyncio.run(backfill()).failed_documents == 1
    assert vectors
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    assert client.post(url, headers=admin, json={
        "reason_code": "integrity_defect", "trigger": "integrity",
    }).status_code == 200
    response = client.post(f"{url}/reconciliation", headers=admin)
    assert response.status_code == 200, response.text
    if failure == "unsettled":
        assert response.json()["data"]["state"] == "suspended"
        assert response.json()["data"]["allowed_next_action"] == "retry_reconciliation"
    else:
        assert response.json()["data"]["state"] == "completed"
        assert vectors == set()


@pytest.mark.parametrize("failure", ["read_failed", "read_cancelled", "intent_failed", "intent_response_lost"])
def test_backfill_prewrite_exit_retains_a_settled_cleanup_target(client, monkeypatch, failure):
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.documents.dense_index_service import DenseIndexService
    from app.documents.dense_maintenance_service import DenseMaintenanceService
    from tests.unit.test_dense_maintenance_service import _dense_settings

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    original_execute = AsyncSession.execute
    original_commit = AsyncSession.commit
    index_calls = []
    commit_fault = False

    async def execute(session, statement, *args, **kwargs):
        if failure.startswith("read_") and "FROM document_chunks" in str(statement):
            if failure == "read_cancelled":
                raise asyncio.CancelledError()
            raise OSError("chunks temporarily unavailable")
        return await original_execute(session, statement, *args, **kwargs)

    async def commit(session):
        nonlocal commit_fault
        if failure.startswith("intent_") and not commit_fault:
            commit_fault = True
            if failure == "intent_response_lost":
                await original_commit(session)
            raise OSError("intent commit failed or its response was lost")
        await original_commit(session)

    async def index(self, **kwargs):
        index_calls.append(kwargs)
        raise AssertionError("no external write may start")

    async def delete(self, **kwargs):
        pass

    monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", delete)

    async def backfill():
        async with client.app.state.test_auth_session_factory() as session:
            return await DenseMaintenanceService(settings=_dense_settings()).backfill_published_documents(
                session=session, limit=10,
            )

    with monkeypatch.context() as fault:
        fault.setattr(AsyncSession, "execute", execute)
        fault.setattr(AsyncSession, "commit", commit)
        fault.setattr(DenseIndexService, "index_candidate_generation", index)
        if failure == "read_cancelled":
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(backfill())
        else:
            result = asyncio.run(backfill())
            assert result.failed_documents == 1
    assert index_calls == []
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    assert client.post(url, headers=admin, json={
        "reason_code": "integrity_defect", "trigger": "integrity",
    }).status_code == 200
    for _ in range(2):
        response = client.post(f"{url}/reconciliation", headers=admin)
        assert response.status_code == 200, response.text
        assert response.json()["data"]["state"] == "completed"


def test_backfill_rechecks_withdrawal_after_durable_intent_commit(client, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.documents.dense_index_service import DenseIndexResult, DenseIndexService
    from app.documents.dense_maintenance_service import DenseMaintenanceService
    from tests.unit.test_dense_maintenance_service import _dense_settings

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    original_commit = AsyncSession.commit
    armed = True
    writes = []

    async def commit(session):
        nonlocal armed
        await original_commit(session)
        if armed:
            armed = False
            response = await asyncio.to_thread(client.post, url, headers=admin, json={
                "reason_code": "integrity_defect", "trigger": "integrity",
            })
            assert response.status_code == 200, response.text

    async def index(self, **kwargs):
        writes.append(kwargs)
        return DenseIndexResult(active=True, fingerprint=kwargs.get("embedding_fingerprint"))

    async def delete(self, **kwargs):
        pass

    monkeypatch.setattr(AsyncSession, "commit", commit)
    monkeypatch.setattr(DenseIndexService, "index_candidate_generation", index)
    monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", delete)

    async def backfill():
        async with client.app.state.test_auth_session_factory() as session:
            return await DenseMaintenanceService(settings=_dense_settings()).backfill_published_documents(
                session=session, limit=10,
            )

    result = asyncio.run(backfill())
    assert result.indexed_documents == 0
    assert result.skipped_documents == 1
    assert writes == []
    cleanup = client.post(f"{url}/reconciliation", headers=admin)
    assert cleanup.json()["data"]["state"] == "completed"


def test_backfill_cannot_downgrade_missing_publication_projection_to_legacy(client, monkeypatch):
    from app.documents.dense_index_service import DenseIndexResult, DenseIndexService
    from app.documents.dense_maintenance_service import DenseMaintenanceService
    from app.model.document import Document
    from app.reviewed_bundles.models import PublishedKnowledgeVersion
    from tests.unit.test_dense_maintenance_service import _dense_settings

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    assert client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=admin, json={"reason_code": "integrity_defect", "trigger": "integrity"},
    ).status_code == 200
    writes = []

    async def index(self, **kwargs):
        writes.append(kwargs)
        return DenseIndexResult(active=True, fingerprint=kwargs.get("embedding_fingerprint"))

    monkeypatch.setattr(DenseIndexService, "index_candidate_generation", index)

    async def restore_damaged_projection():
        async with client.app.state.test_auth_session_factory() as session:
            document = await session.get(Document, "runtime-document:ticket25-entry")
            document.deleted_at = None
            table = PublishedKnowledgeVersion.__table__
            await session.execute(table.delete().where(table.c.id == publication))
            await session.commit()
        async with client.app.state.test_auth_session_factory() as session:
            return await DenseMaintenanceService(settings=_dense_settings()).backfill_published_documents(
                session=session, limit=10,
            )

    result = asyncio.run(restore_damaged_projection())
    assert result.indexed_documents == 0
    assert result.skipped_documents == 1
    assert result.documents[0].reason == "publication_withdrawn"
    assert writes == []


@pytest.mark.parametrize("before_external_write,startup_recovery,corrupt_pending", [
    (False, False, False), (True, False, False), (False, True, False), (True, True, False), (False, False, True),
])
def test_successor_cleanup_waits_for_its_exact_writer_attempt_to_exit(
    client, monkeypatch, before_external_write, startup_recovery, corrupt_pending,
):
    from app.documents.dense_index_service import DenseIndexResult, DenseIndexService
    from app.infra.milvus_document_index import MilvusDocumentIndex
    from app.reviewed_bundles.build_service import CandidateBuildService
    from app.reviewed_bundles.models import CandidateBuildJob
    from app.reviewed_bundles.recovery import CandidateBuildRecoveryService
    from app.reviewed_bundles.runtime import candidate_build_runtime
    from app.reviewed_bundles.verifier import CanonicalEditorialExportVerifier

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)

    async def export():
        async with client.app.state.test_auth_session_factory() as session:
            actor = await session.scalar(select(User).where(User.username == "ticket25-admin"))
            return await EditorialAuthorityService(session).export_approved_revision("ticket25-entry", actor)

    imported = client.post("/api/v1/reviewed-release-bundles/import", headers=admin, json=_ticket24_manifest(
        bundle_id="ticket25-inflight-successor", artifact=asyncio.run(export())["artifact"], operation="replace",
    ))
    assert imported.status_code == 200, imported.text
    job_id = imported.json()["data"]["items"][0]["job_id"]
    started = threading.Event()
    release = threading.Event()
    rows = []
    if before_external_write:
        original_write_chunks = CandidateBuildService._write_chunks

        async def paused_chunks(self, **kwargs):
            result = await original_write_chunks(self, **kwargs)
            started.set()
            assert await asyncio.to_thread(release.wait, 10)
            return result

        monkeypatch.setattr(CandidateBuildService, "_write_chunks", paused_chunks)

    class BlockingClient:
        def upsert(self, collection_name, payload):
            started.set()
            assert release.wait(10)
            rows.extend(payload)

    adapter = MilvusDocumentIndex(client=BlockingClient())

    async def delete(self, *, document_id, generation, embedding_fingerprint):
        rows[:] = [row for row in rows if (row["document_id"], row["generation"]) != (document_id, generation)]

    class Indexer:
        async def index_candidate_generation(self, *, document_id, generation, chunks, embedding_fingerprint):
            await adapter.upsert_generation(collection_name="ticket25-successor", rows=[{
                "document_id": document_id, "generation": generation, "chunk_index": chunk.chunk_index,
                "content_sha256": chunk.content_sha256, "vector": [1.0],
            } for chunk in chunks])
            return DenseIndexResult(active=False, fingerprint=None)

        delete_candidate_generation = delete

    async def unreachable(job_id):
        return False

    monkeypatch.setattr(candidate_build_runtime, "cancel", unreachable)
    monkeypatch.setattr(DenseIndexService, "delete_candidate_generation", delete)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"

    async def recover():
        async def forbidden_enqueue(job_id):
            raise AssertionError("withdrawn successor cannot be requeued")

        async with client.app.state.test_auth_session_factory() as session:
            return await CandidateBuildRecoveryService(session, dense_index_service=Indexer()).recover(
                enqueue=forbidden_enqueue, recovery_owner="ticket25-other-runtime",
            )

    async def race():
        async with client.app.state.test_auth_session_factory() as session:
            actor = await session.scalar(select(User).where(User.username == "ticket25-admin"))
            identity = await IdentityAuditService(session).ensure_member_record(actor, admission_path="test")
            builder = CandidateBuildService(
                session, dense_index_service=Indexer(), editorial_export_verifier=CanonicalEditorialExportVerifier(session),
            )
            await builder.dispatch_job(job_id, actor_identity=identity)
            worker = asyncio.create_task(builder.process_job(job_id))
            try:
                assert await asyncio.to_thread(started.wait, 10)
                withdrawal = await asyncio.to_thread(client.post, url, headers=admin, json={
                    "reason_code": "integrity_defect", "trigger": "integrity",
                })
                assert withdrawal.status_code == 200, withdrawal.text
                if startup_recovery:
                    recovery = await recover()
                    assert job_id in recovery["cleanup_pending_job_ids"]
                    assert job_id not in recovery["reconciled_cleanup_job_ids"]
                if corrupt_pending:
                    async with client.app.state.test_auth_session_factory() as legacy_session:
                        job = await legacy_session.get(CandidateBuildJob, job_id)
                        job.derived_cleanup_pending = False
                        await legacy_session.commit()
                early = await asyncio.to_thread(client.post, f"{url}/reconciliation", headers=admin)
                assert early.json()["data"]["state"] == "suspended"
                release.set()
                await worker
                assert bool(rows) is not before_external_write
                if startup_recovery:
                    recovery = await recover()
                    assert job_id in recovery["reconciled_cleanup_job_ids"]
                settled = await asyncio.to_thread(client.post, f"{url}/reconciliation", headers=admin)
                assert settled.json()["data"]["state"] == "completed"
                assert rows == []
            finally:
                release.set()
                await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(race())


@pytest.mark.parametrize("route", ["single", "batch"])
def test_legacy_published_deletion_requires_explicit_withdrawal(client, route):
    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    document = client.get("/api/v1/documents", headers=admin).json()["data"]["items"][0]
    if route == "single":
        response = client.delete(f"/api/v1/documents/{document['filename']}", headers=admin)
    else:
        response = client.post("/api/v1/documents/batch-delete", headers=admin, json={
            "document_ids": [document["document_id"]],
        })
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "EXPLICIT_WITHDRAWAL_REQUIRED"
    assert client.get(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal", headers=admin,
    ).status_code == 404


@pytest.mark.parametrize("method,path", [
    ("post", ""), ("get", ""), ("get", "/reconciliation"), ("post", "/reconciliation"),
])
def test_withdrawal_administration_denies_members_and_anonymous(client, method, path):
    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    member = asyncio.run(_headers(client, username="ticket25-other", role="user"))
    publication = _publish(client, admin)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal{path}"
    kwargs = {"json": {"reason_code": "integrity_defect", "trigger": "integrity"}} if method == "post" and not path else {}
    for headers, expected in (({}, 401), (member, 403)):
        response = getattr(client, method)(url, headers=headers, **kwargs)
        assert response.status_code == expected, response.text


def test_repeated_withdrawal_cannot_change_original_reason(client):
    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    original = client.post(url, headers=admin, json={
        "reason_code": "integrity_defect", "trigger": "integrity",
    }).json()["data"]
    conflict = client.post(url, headers=admin, json={
        "reason_code": "source_unavailable", "trigger": "source",
    })
    assert conflict.status_code == 409
    assert client.get(url, headers=admin).json()["data"] == original


async def _replacement_candidate(client):
    async with client.app.state.test_auth_session_factory() as session:
        admin = await session.scalar(select(User).where(User.username == "ticket25-admin"))
        exported = await EditorialAuthorityService(session).export_approved_revision("ticket25-entry", admin)
        actor = await IdentityAuditService(session).ensure_member_record(admin, admission_path="test")
        return await _build_ticket24_candidate(
            session, artifact=exported["artifact"], bundle_id="ticket25-replacement",
            operation="replace", actor_identity=actor,
        )


@pytest.mark.parametrize("legacy_completion", [
    "current", "valid", "bad_attempt", "missing_candidate", "missing_completion", "bad_candidate_binding",
    "pre0018", "modern_missing_hash", "legacy_wrong_hash",
])
def test_withdrawal_invalidates_ready_successor_and_rejects_retry(client, legacy_completion):
    from app.model.canonical import CanonicalEventModel, CanonicalRecordModel

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    candidate = asyncio.run(_replacement_candidate(client))
    detail = client.get(f"/api/v1/reviewed-release-bundles/candidates/{candidate}/inspection", headers=admin)
    assert detail.status_code == 200, detail.text
    bundle = client.get("/api/v1/reviewed-release-bundles/ticket25-replacement", headers=admin)
    assert bundle.status_code == 200, bundle.text
    job_id = bundle.json()["data"]["items"][0]["job_id"]
    if legacy_completion != "current":
        async def pre_upgrade_fixture():
            async with client.app.state.test_auth_session_factory() as session:
                connection = await session.connection()
                await connection.execute(CanonicalEventModel.__table__.delete().where(
                    CanonicalEventModel.aggregate_id == f"build_generation:{job_id}",
                    CanonicalEventModel.payload["schema"].as_string() == "candidate_writer_exit/v1",
                ))
                if legacy_completion in {"pre0018", "modern_missing_hash", "legacy_wrong_hash"}:
                    records = [candidate]
                    if legacy_completion != "modern_missing_hash":
                        records.append(f"build_generation:{job_id}")
                    for identity in records:
                        record = await session.get(CanonicalRecordModel, identity)
                        await connection.execute(CanonicalRecordModel.__table__.update().where(
                            CanonicalRecordModel.stable_id == identity,
                        ).values(payload={
                            key: value for key, value in record.payload.items() if key != "frozen_input_sha256"
                        }))
                    events = (await session.scalars(select(CanonicalEventModel).where(
                        CanonicalEventModel.aggregate_id == f"build_generation:{job_id}",
                    ))).all()
                    for event in events:
                        old_payload = {key: value for key, value in event.payload.items() if key != "frozen_input_sha256"}
                        if legacy_completion == "legacy_wrong_hash" and event.payload.get("action") == "completed":
                            old_payload["frozen_input_sha256"] = "0" * 64
                        await connection.execute(CanonicalEventModel.__table__.update().where(
                            CanonicalEventModel.id == event.id,
                        ).values(payload=old_payload))
                if legacy_completion == "bad_attempt":
                    event = await session.scalar(select(CanonicalEventModel).where(
                        CanonicalEventModel.aggregate_id == f"build_generation:{job_id}",
                        CanonicalEventModel.payload["action"].as_string() == "completed",
                    ))
                    await connection.execute(CanonicalEventModel.__table__.update().where(
                        CanonicalEventModel.id == event.id,
                    ).values(payload={**event.payload, "attempt": 2}))
                if legacy_completion == "missing_candidate":
                    await connection.execute(CanonicalRecordModel.__table__.delete().where(
                        CanonicalRecordModel.stable_id == candidate,
                    ))
                if legacy_completion == "missing_completion":
                    await connection.execute(CanonicalEventModel.__table__.delete().where(
                        CanonicalEventModel.aggregate_id == f"build_generation:{job_id}",
                        CanonicalEventModel.payload["action"].as_string() == "completed",
                    ))
                if legacy_completion == "bad_candidate_binding":
                    record = await session.get(CanonicalRecordModel, candidate)
                    await connection.execute(CanonicalRecordModel.__table__.update().where(
                        CanonicalRecordModel.stable_id == candidate,
                    ).values(payload={**record.payload, "requested_generation": 999}))
                await session.commit()

        asyncio.run(pre_upgrade_fixture())
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=admin, json={"reason_code": "integrity_defect", "trigger": "integrity"},
    )
    assert response.status_code == 200
    job = client.get(f"/api/v1/reviewed-release-bundles/jobs/{job_id}", headers=admin).json()["data"]
    assert job["status"] == "superseded"
    assert job["allowed_next_action"] == "none"
    assert client.post(f"/api/v1/reviewed-release-bundles/jobs/{job_id}/retry", headers=admin).status_code == 409
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal/reconciliation"
    for _ in range(2):
        result = client.post(url, headers=admin)
        assert result.status_code == 200, result.text
        assert result.json()["data"]["state"] == (
            "completed" if legacy_completion in {"current", "valid", "pre0018"} else "suspended"
        )


@pytest.mark.parametrize("missing_predecessor_projection", ["none", "publication", "candidate_index", "both"])
def test_replacement_withdrawal_preserves_predecessor_history_and_lineage(client, missing_predecessor_projection):
    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    reader = asyncio.run(_headers(client, username="ticket25-reader", role="user"))
    predecessor = _publish(client, admin)
    _, _, old = asyncio.run(_frozen_execution(client, session_id="ticket25-old"))
    candidate = asyncio.run(_replacement_candidate(client))
    _inspect_and_accept(client, candidate_id=candidate, headers=admin)
    eligibility = client.get(
        f"/api/v1/reviewed-release-bundles/candidates/{candidate}/publication-eligibility", headers=admin,
    ).json()["data"]
    published = client.post("/api/v1/reviewed-release-bundles/publication-batches", headers=admin, json={
        "confirmation_id": "ticket25-replacement-confirmation",
        "selected_items": [_publish_selection(candidate, eligibility)],
    }).json()["data"]
    assert published["batch_complete"], published
    successor = published["published"][0]["publication_identity"]
    asyncio.run(_frozen_execution(client, session_id="ticket25-new"))
    payload = {"reason_code": "editorial_withdrawal", "trigger": "editorial"}
    stale = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{predecessor}/withdrawal", headers=admin, json=payload,
    )
    assert stale.status_code == 409
    original_bundle = client.get("/api/v1/reviewed-release-bundles/ticket25-entry-bundle", headers=admin).json()["data"]
    original_job_id = original_bundle["items"][0]["job_id"]
    original_job_url = f"/api/v1/reviewed-release-bundles/jobs/{original_job_id}"
    original_job = client.get(original_job_url, headers=admin).json()["data"]
    if missing_predecessor_projection != "none":
        from app.reviewed_bundles.models import CandidateBuildJob, PublishedKnowledgeVersion

        async def lose_projection():
            async with client.app.state.test_auth_session_factory() as session:
                if missing_predecessor_projection in {"publication", "both"}:
                    table = PublishedKnowledgeVersion.__table__
                    await session.execute(table.delete().where(table.c.id == predecessor))
                if missing_predecessor_projection in {"candidate_index", "both"}:
                    job = await session.get(CandidateBuildJob, original_job_id)
                    job.candidate_id = None
                await session.commit()

        asyncio.run(lose_projection())
        original_job = client.get(original_job_url, headers=admin).json()["data"]
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{successor}/withdrawal", headers=admin, json=payload,
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["supersedes_version_id"] == predecessor
    assert client.get(original_job_url, headers=admin).json()["data"] == original_job
    reconciled = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{successor}/withdrawal/reconciliation", headers=admin,
    )
    assert reconciled.status_code == 200, reconciled.text
    assert reconciled.json()["data"]["state"] == "completed"

    async def startup():
        from app.documents.dense_index_service import DenseIndexService
        from app.reviewed_bundles.recovery import CandidateBuildRecoveryService

        async def forbidden_enqueue(job_id):
            raise AssertionError("unexpected requeue")

        async with client.app.state.test_auth_session_factory() as session:
            return await CandidateBuildRecoveryService(session, dense_index_service=DenseIndexService()).recover(
                enqueue=forbidden_enqueue,
            )

    assert original_job_id not in asyncio.run(startup())["reconciled_cleanup_job_ids"]
    assert client.get(original_job_url, headers=admin).json()["data"] == original_job
    old_history = client.get("/api/v1/sessions/ticket25-old", headers=reader).json()["data"]["messages"][-1]
    assert old_history["evidence_summary"] == old.projection["evidence_summary"]
    new_history = client.get("/api/v1/sessions/ticket25-new", headers=reader).json()["data"]["messages"][-1]
    assert new_history["evidence_summary"]["sources"][0]["withdrawal"]["publication_identity"] == successor


@pytest.mark.parametrize("mixed_version_binding", [False, True])
def test_withdrawal_suspends_bound_acceptance_without_suspending_siblings(client, monkeypatch, mixed_version_binding):
    from tests.integration.test_delivery_acceptance import _activate, _record

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    records = []
    for entry in ("entry:ticket25-entry", "entry:ticket25-unrelated"):
        payload = _record()
        payload["content_identities"] = [entry]
        if mixed_version_binding:
            payload["content_identities"].append("published_knowledge_version:unrelated-entry-v1")
        payload["affected_scope"]["entry_identities"] = [entry]
        payload["affected_scope"]["blocking_scope_identity"] = entry
        for check in payload["checks"]:
            check["identity_dependencies"] = [
                entry if identity == "entry:decision-entry-001" else identity
                for identity in check.get("identity_dependencies", [])
                if not identity.startswith("published_knowledge_version:")
            ]
        response = client.post("/api/v1/acceptance/records", headers=admin, json=payload)
        assert response.status_code == 200, response.text
        records.append(response.json()["data"]["record_id"])
        _activate(client, admin, records[-1])

    async def unavailable(*args, **kwargs):
        raise OSError("private history failure")

    monkeypatch.setattr(AnswerExecutionStore, "redact_document_evidence", unavailable)
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=admin, json={"reason_code": "privacy_defect", "trigger": "integrity"},
    )
    assert response.status_code == 200, response.text
    affected = client.get(f"/api/v1/acceptance/records/{records[0]}", headers=admin).json()["data"]
    sibling = client.get(f"/api/v1/acceptance/records/{records[1]}", headers=admin).json()["data"]
    assert affected["current_status"] == "suspended"
    assert affected["accepted_scope"]["entry_identities"] == []
    assert sibling["current_status"] == "active"
    assert sibling["accepted_scope"]["entry_identities"] == ["entry:ticket25-unrelated"]


def test_withdrawal_rejects_boolean_canonical_generation(client):
    from app.model.canonical import CanonicalRecordModel

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)

    async def corrupt_fixture():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, publication)
            table = CanonicalRecordModel.__table__
            await session.execute(table.update().where(
                table.c.stable_id == publication,
            ).values(payload={**record.payload, "generation": True}))
            await session.commit()

    asyncio.run(corrupt_fixture())
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=admin, json={"reason_code": "integrity_defect", "trigger": "integrity"},
    )
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "WITHDRAWAL_BINDING_INVALID"


def test_withdrawal_rejects_contradictory_predecessor_projection(client):
    from app.reviewed_bundles.models import PublishedKnowledgeVersion

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)

    async def corrupt_fixture():
        async with client.app.state.test_auth_session_factory() as session:
            table = PublishedKnowledgeVersion.__table__
            await session.execute(table.update().where(table.c.id == publication).values(
                supersedes_version_id="published_knowledge_version:foreign-predecessor",
            ))
            await session.commit()

    asyncio.run(corrupt_fixture())
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=admin, json={"reason_code": "integrity_defect", "trigger": "integrity"},
    )
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "WITHDRAWAL_BINDING_INVALID"


def test_needs_re_review_source_loss_can_be_explicitly_withdrawn(client):
    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)

    async def lose_source():
        async with client.app.state.test_auth_session_factory() as session:
            maintainer = await session.scalar(select(User).where(User.username == "ticket25-entry-maintainer"))
            source_id = _browser_ticket24_entry("ticket25-entry").sources[0]["source_id"]
            return await EditorialAuthorityService(session).record_source_availability(
                "ticket25-entry", source_id, "unavailable_for_new_evidence", maintainer,
            )

    assert asyncio.run(lose_source())["lifecycle_state"] == "needs_re_review"
    response = client.post(
        f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
        headers=admin, json={"reason_code": "source_unavailable", "trigger": "source"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["state"] == "withdrawn"
    assert response.json()["data"]["trigger"] == "source"


def test_withdrawal_serializes_with_acceptance_first_publication_lock(client):
    from sqlalchemy import event
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm import Session

    from app.model.canonical import CanonicalRecordModel
    from tests.integration.test_delivery_acceptance import _activate, _record

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    created = client.post("/api/v1/acceptance/records", headers=admin, json=_record())
    assert created.status_code == 200, created.text
    acceptance = created.json()["data"]["record_id"]
    _activate(client, admin, acceptance)
    publication_holds_acceptance = True

    def competing_publication(execution):
        nonlocal publication_holds_acceptance
        statement = execution.statement
        if (
            not execution.is_select or statement._for_update_arg is None
            or execution.bind_mapper is None or execution.bind_mapper.class_ is not CanonicalRecordModel
        ):
            return
        frozen = execution.invoke_statement().freeze()
        # Model a finalizer holding acceptance and next requesting entry. It may
        # finish before withdrawal acquires entry only if withdrawal waits first.
        for record in frozen().scalars():
            if record.stable_id == acceptance:
                publication_holds_acceptance = False
            elif record.stable_id == "entry:ticket25-entry" and publication_holds_acceptance:
                raise OperationalError("canonical authority lock", {}, Exception("simulated publication deadlock"))
        return frozen()

    event.listen(Session, "do_orm_execute", competing_publication, retval=True)
    try:
        response = client.post(
            f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal",
            headers=admin, json={"reason_code": "integrity_defect", "trigger": "integrity"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["state"] == "withdrawn"
    finally:
        event.remove(Session, "do_orm_execute", competing_publication)


def test_reconciliation_serializes_with_document_first_answer_finalization(client):
    from sqlalchemy import event
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm import Session

    from app.model.answer_execution import AnswerExecutionModel
    from app.model.document import Document

    admin = asyncio.run(_headers(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, admin)
    asyncio.run(_frozen_execution(client, complete=False))
    url = f"/api/v1/reviewed-release-bundles/publications/{publication}/withdrawal"
    assert client.post(url, headers=admin, json={
        "reason_code": "integrity_defect", "trigger": "integrity",
    }).status_code == 200
    answer_holds_document = True

    def competing_answer(execution):
        nonlocal answer_holds_document
        if (
            not execution.is_select or execution.statement._for_update_arg is None
            or execution.bind_mapper is None
        ):
            return
        model = execution.bind_mapper.class_
        if model is Document:
            answer_holds_document = False
        elif model is AnswerExecutionModel and answer_holds_document:
            raise OperationalError("answer finalization lock", {}, Exception("simulated reconciliation deadlock"))

    event.listen(Session, "do_orm_execute", competing_answer)
    try:
        response = client.post(f"{url}/reconciliation", headers=admin)
        assert response.status_code == 200, response.text
        assert response.json()["data"]["state"] == "completed"
    finally:
        event.remove(Session, "do_orm_execute", competing_answer)
