import json
from dataclasses import replace

import pytest

from app.rag import answer_execution as answer_execution_module
from app.retrieval.candidate_pool import AuthorizedRetrievalCandidatePool
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


@pytest.mark.parametrize(
    ("observation", "failure"),
    [(observation, failure) for observation in ("citation_drift", "retrieval_miss", "condition_loss", "product_failure")
     for failure in ("repaired", "provider_timeout", "unrepaired")]
    + [("condition_loss", "question_drift")],
)
def test_retrieval_diagnosis_requires_observed_failure_and_replays_after_repair(client, monkeypatch, failure, observation):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="maintenance-citation")
    admin, maintainer, worker = _maintainer(client)
    reporter = _headers(_register(client, username="citation-reporter"))
    provider = _activate_route(client, monkeypatch, admin)
    question = "Which Candidate publication contract applies? deployment=production"
    reported = _gap(client, reporter, "citation-report", question)
    independent = _gap(client, worker, "citation-reference", question)
    assert reported["answer_execution"]["outcome"] == "evidence_gated_answer"
    assert independent["answer_execution"]["outcome"] == "evidence_gated_answer"
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=reporter,
        json={"answer_id": reported["id"], "entry_id": "maintenance-citation", "label": "insufficient_evidence"},
    )
    assert signal.status_code == 200, signal.text
    classification = "product-privacy-operations" if observation == "product_failure" else "retrieval-answer-behavior"
    disposition = "product-repair" if observation == "product_failure" else "retrieval-experiment"
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": classification, "severity": "p2", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signal.json()["data"]["id"]],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    assert client.post(base + "/administrator", headers=admin, json={"expected_revision": 1}).status_code == 200
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 2, "state": "triaged"}).status_code == 200
    original = provider.complete
    original_retrieve = AuthorizedRetrievalCandidatePool.retrieve
    original_prompt = answer_execution_module.build_generation_prompt

    async def invalid_citation(prompt, *, system_prompt=None):
        text = await original(prompt, system_prompt=system_prompt)
        return text.replace("[S1]", "[S99]")

    async def miss_eligible_candidates(self, query, top_k):
        result = await original_retrieve(self, query, top_k)
        assert result.items, "Fault injection requires genuinely eligible published evidence"
        return replace(result, items=[], merged_count=0)

    def lose_conditions(question, evidence):
        prompt = original_prompt(question, evidence)
        envelope = json.loads(prompt.user_prompt)
        assert envelope["query_condition_set"]["conditions"]
        if failure == "question_drift" or observation == "product_failure":
            envelope["user_question"] = "A different synthetic question."
        else:
            envelope["query_condition_set"]["conditions"] = []
        return replace(prompt, user_prompt=json.dumps(envelope))

    if failure == "provider_timeout":
        provider.unavailable = True
    elif observation in {"condition_loss", "product_failure"}:
        monkeypatch.setattr(answer_execution_module, "build_generation_prompt", lose_conditions)
    elif observation == "retrieval_miss":
        monkeypatch.setattr(AuthorizedRetrievalCandidatePool, "retrieve", miss_eligible_candidates)
    else:
        monkeypatch.setattr(provider, "complete", invalid_citation)
    reproduced = client.post(
        base + "/reproductions", headers=worker,
        json={
            "expected_revision": 3, "answer_id": independent["id"],
            "signal_id": signal.json()["data"]["id"], "expected_outcome": "evidence_gated_answer",
            "confirmed_synthetic_fixture": True, "verified_observation": observation,
            "reference_answer_id": independent["id"],
        },
    )
    if failure in {"provider_timeout", "question_drift"}:
        assert reproduced.status_code == 409, reproduced.text
        return
    assert reproduced.status_code == 200, reproduced.text
    fixture = reproduced.json()["data"]
    if observation == "citation_drift":
        assert fixture["observed_outcome"] == "generation_unavailable"
        assert fixture["diagnosis_facts"].get("verified_difference") is True, fixture
    elif observation == "retrieval_miss":
        assert fixture["observed_outcome"] == "insufficient_evidence_reply"
        assert fixture["diagnosis_facts"]["frozen_evidence"] is False, fixture
    else:
        assert fixture["observed_outcome"] is None
        assert fixture["observed_state"] == "failed"
        if observation == "condition_loss":
            assert fixture["diagnosis_facts"].get("verified_difference") is True, fixture
    assert fixture["reference_evidence_publication_identities"] == [publication]
    diagnosed = client.post(
        base + "/diagnosis", headers=maintainer,
        json={"expected_revision": 4, "fixture_identity": fixture["id"], "observation": observation},
    )
    assert diagnosed.status_code == 200, diagnosed.text
    assert diagnosed.json()["data"]["disposition"] == disposition
    approved = client.post(base + "/findings", headers=maintainer, json={"expected_revision": 5, "fixture_identity": fixture["id"]})
    assert approved.status_code == 200, approved.text
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 6, "state": "in_progress"}).status_code == 200
    if failure != "unrepaired":
        monkeypatch.setattr(provider, "complete", original)
        monkeypatch.setattr(AuthorizedRetrievalCandidatePool, "retrieve", original_retrieve)
        monkeypatch.setattr(answer_execution_module, "build_generation_prompt", original_prompt)
    replayed = client.post(
        base + "/replays", headers=worker,
        json={"expected_revision": 7, "fixture_identity": fixture["id"], "answer_id": independent["id"]},
    )
    assert replayed.status_code == 200, replayed.text
    replay = replayed.json()["data"]
    assert replay["passed"] is (failure != "unrepaired"), replay
    assert replay["query_condition_set_identity"] == fixture["query_condition_set_identity"]
    expected_publications = [] if observation != "citation_drift" and failure == "unrepaired" else [publication]
    assert replay["evidence_publication_identities"] == expected_publications
    resolved = client.post(
        base + "/resolution", headers=maintainer,
        json={
            "expected_revision": 8, "disposition": disposition,
            "artifact_identities": [fixture["id"], replay["id"]],
        },
    )
    if failure == "unrepaired":
        assert resolved.status_code == 409, resolved.text
        assert client.get(base, headers=maintainer).json()["data"]["state"] == "in_progress"
        return
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["data"]["state"] == "resolved"
    closed = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 9, "state": "closed_confirmation"})
    assert closed.status_code == 200, closed.text
