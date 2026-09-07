import asyncio
import hashlib
import json

import pytest

from app.extensions.provider_router import ProviderRouter
from app.rag.answer_evidence import (
    evidence_snapshot_id,
    evidence_summary_from_execution,
    evidence_summary_from_trace,
)
from app.rag.answer_execution import AnswerOutcomeKind, EvidenceGatedAnswerExecutor
from app.rag.evidence_sufficiency import QueryConditionSet
from app.rag.interfaces import GenerationCompletion, RetrieveResult
from app.retrieval.policy import PILOT_RETRIEVAL_PROFILE_ID
from tests.support.generation import approved_test_route


def _candidate(*, section_id: str) -> dict:
    entry_id = "decision-execution-001"
    source_id = f"{entry_id}-source"
    content_sha256 = hashlib.sha256(f"{section_id}-content".encode()).hexdigest()
    return {
        "chunk_id": f"chunk-{section_id}",
        "document_id": f"{entry_id}-document",
        "generation": 1,
        "chunk_index": 0,
        "content_preview": "This is an authorized but non-governing reviewed snapshot.",
        "content_length": len("This is an authorized but non-governing reviewed snapshot."),
        "content_sha256": content_sha256,
        "score": 999.0,
        "answer_evidence_eligible": True,
        "entry_id": entry_id,
        "entry_identity": f"entry:{entry_id}",
        "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
        "publication_identity": f"published_knowledge_version:{entry_id}.v1",
        "publication_version": "v1",
        "section_id": section_id,
        "section_identity": f"entry:{entry_id}#{section_id}",
        "decision_query": "Which reviewed operating decision applies for environment=production?",
        "chunk_identity": {
            "document_id": f"{entry_id}-document",
            "generation": 1,
            "chunk_index": 0,
            "content_sha256": content_sha256,
        },
        "source_relationships": [
            {
                "source_identity": f"source:{source_id}",
                "availability": "verified_usable",
                "access_scope": "controlled_internal",
            }
        ],
        "assurance_level": "source_grounded",
        "applicability_conditions": [
            {
                "condition_id": "environment-production",
                "field": "environment",
                "operator": "equals",
                "value": "production",
            }
        ],
        "freshness_triggers": [{"trigger_id": "source-change"}],
        "lifecycle_state": "published",
        "metadata": {
            "title": "Execute evidence sufficiency",
            "publication_version": "v1",
            "entry_id": entry_id,
            "entry_identity": f"entry:{entry_id}",
            "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
            "entry_title": "Execute evidence sufficiency",
            "domain": "evidence-sufficiency",
            "section_id": section_id,
            "source_id": source_id,
            "source_title": "Reviewed controlled source",
            "source_authority": "ZhoMind editorial authority",
            "source_url": f"controlled://knowledge/{source_id}",
            "source_version": "2026-09-06",
            "review_date": "2026-09-06",
            "review_status": "approved",
            "source_availability": "verified",
            "source_access_scope": "controlled_internal",
            "source_tier": "primary_evidence_source",
            "evidence_conflict": "none",
            "decision_query": "Which reviewed operating decision applies for environment=production?",
        },
    }


class _PilotRetriever:
    async def retrieve(self, _query: str, top_k: int) -> RetrieveResult:
        del top_k
        return RetrieveResult(
            items=[_candidate(section_id="alternatives")],
            strategy="sparse_bm25",
            lexical_candidate_count=1,
            merged_count=1,
            profile_identity=PILOT_RETRIEVAL_PROFILE_ID,
            candidate_pool_scope="published_knowledge",
        )


class _SufficientPilotRetriever:
    async def retrieve(self, _query: str, top_k: int) -> RetrieveResult:
        del top_k
        return RetrieveResult(
            items=[_candidate(section_id="recommendation_or_reviewed_branches")],
            strategy="sparse_bm25",
            lexical_candidate_count=1,
            merged_count=1,
            profile_identity=PILOT_RETRIEVAL_PROFILE_ID,
            candidate_pool_scope="published_knowledge",
        )


class _WrongScopePilotRetriever:
    async def retrieve(self, _query: str, top_k: int) -> RetrieveResult:
        del top_k
        return RetrieveResult(
            items=[_candidate(section_id="recommendation_or_reviewed_branches")],
            strategy="sparse_bm25",
            lexical_candidate_count=1,
            merged_count=1,
            profile_identity=PILOT_RETRIEVAL_PROFILE_ID,
            candidate_pool_scope="candidate_preview",
        )


class _ProfilelessPilotRetriever:
    async def retrieve(self, _query: str, top_k: int) -> RetrieveResult:
        del top_k
        return RetrieveResult(
            items=[_candidate(section_id="recommendation_or_reviewed_branches")],
            strategy="sparse_bm25",
            lexical_candidate_count=1,
            merged_count=1,
            candidate_pool_scope="published_knowledge",
        )


class _IdentityReranker:
    async def rerank(self, _query: str, items: list[dict]) -> list[dict]:
        return items


class _FailIfCalledJudge:
    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, _query: str, _context: list[dict]) -> bool:
        self.calls += 1
        raise AssertionError("Pilot sufficiency must not call the online relevance judge")


class _RecordingProvider:
    def __init__(self, *, answer: str = "This must not be generated.") -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.answer = answer

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        del system_prompt
        self.calls += 1
        self.prompts.append(prompt)
        return self.answer


class _ObservedRecordingProvider(_RecordingProvider):
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
        from app.rag.generation_observation import (
            generation_envelope_observation,
            provider_visible_snapshot_ids,
        )

        self.calls += 1
        self.prompts.append(prompt)
        return GenerationCompletion(
            text=self.answer,
            generation_envelope=generation_envelope_observation(
                user_prompt=prompt,
                system_prompt=system_prompt,
                snapshot_ids=provider_visible_snapshot_ids(prompt),
            ),
        )


def _valid_answer(*, recommendation: str = "Use the deterministic reviewed default. [S1]") -> str:
    return f"""## Recommendation
{recommendation}

## Applicability Limits
This applies only when environment=production. [S1]

## Alternatives
Ask a narrower question when the condition is absent. [S1]

## Minimal Implementation or Acceptance Check
Keep the immutable snapshot and citation together. [S1]"""


def test_pilot_nonempty_context_without_a_governing_section_is_closed_before_judge_or_provider() -> None:
    judge = _FailIfCalledJudge()
    provider = _RecordingProvider()
    executor = EvidenceGatedAnswerExecutor(
        retriever=_PilotRetriever(),
        reranker=_IdentityReranker(),
        judge=judge,
        provider_router=ProviderRouter(providers={"approved": provider}, approved_route=approved_test_route("approved")),
        primary_provider="approved",
        retriever_name="pilot-retriever",
        reranker_name="identity-reranker",
        judge_name="must-not-run",
        retrieval_top_k=20,
        max_evidence_items=3,
        max_excerpt_chars=1200,
    )

    outcome = asyncio.run(
        executor.execute(
            request_id="ticket19-insufficient",
            user_id="knowledge-user",
            session_id="ticket19-session",
            question="What is the reviewed default for environment=production?",
        )
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "decision_not_covered"
    assert outcome.evidence == ()
    trace = outcome.to_rag_trace()
    assert trace["insufficient_evidence_reply"] == {
        "outcome": "insufficient_evidence_reply",
        "reason": "decision_not_covered",
        "query_condition_set_identity": trace["evidence_sufficiency_decision"]["query_conditions"]["identity"],
    }
    assert "answer_evidence_set" not in trace
    assert judge.calls == 0
    assert provider.calls == 0


def test_pilot_rejects_an_authorized_looking_candidate_from_a_nonproduction_pool_scope() -> None:
    judge = _FailIfCalledJudge()
    provider = _RecordingProvider()
    executor = EvidenceGatedAnswerExecutor(
        retriever=_WrongScopePilotRetriever(),
        reranker=_IdentityReranker(),
        judge=judge,
        provider_router=ProviderRouter(providers={"approved": provider}, approved_route=approved_test_route("approved")),
        primary_provider="approved",
        retriever_name="wrong-scope-retriever",
        reranker_name="identity-reranker",
        judge_name="must-not-run",
        retrieval_top_k=20,
        max_evidence_items=3,
        max_excerpt_chars=1200,
    )

    outcome = asyncio.run(
        executor.execute(
            request_id="ticket19-wrong-scope",
            user_id="knowledge-user",
            session_id="ticket19-session",
            question="What is the reviewed default for environment=production?",
        )
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "no_eligible_published_evidence"
    assert outcome.evidence_set is None
    assert judge.calls == 0
    assert provider.calls == 0


def test_pilot_rejects_a_profileless_retrieval_result_before_generation() -> None:
    judge = _FailIfCalledJudge()
    provider = _RecordingProvider()
    executor = EvidenceGatedAnswerExecutor(
        retriever=_ProfilelessPilotRetriever(),
        reranker=_IdentityReranker(),
        judge=judge,
        provider_router=ProviderRouter(providers={"approved": provider}, approved_route=approved_test_route("approved")),
        primary_provider="approved",
        retriever_name="profileless-pilot-retriever",
        reranker_name="identity-reranker",
        judge_name="must-not-run",
        retrieval_top_k=20,
        max_evidence_items=3,
        max_excerpt_chars=1200,
    )

    outcome = asyncio.run(
        executor.execute(
            request_id="ticket19-profileless",
            user_id="knowledge-user",
            session_id="ticket19-session",
            question="What is the reviewed default for environment=production?",
        )
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "no_eligible_published_evidence"
    assert outcome.evidence_set is None
    assert judge.calls == 0
    assert provider.calls == 0


def test_pilot_rejects_an_unresolved_pool_conflict_before_provider_generation() -> None:
    class _ConflictedPilotRetriever:
        async def retrieve(self, _query: str, top_k: int) -> RetrieveResult:
            del top_k
            governing = _candidate(section_id="recommendation_or_reviewed_branches")
            conflict = _candidate(section_id="alternatives")
            conflict["metadata"]["evidence_conflict"] = "unresolved"
            return RetrieveResult(
                items=[governing, conflict],
                strategy="sparse_bm25",
                lexical_candidate_count=2,
                merged_count=2,
                profile_identity=PILOT_RETRIEVAL_PROFILE_ID,
                candidate_pool_scope="published_knowledge",
            )

    judge = _FailIfCalledJudge()
    provider = _RecordingProvider()
    executor = EvidenceGatedAnswerExecutor(
        retriever=_ConflictedPilotRetriever(),
        reranker=_IdentityReranker(),
        judge=judge,
        provider_router=ProviderRouter(providers={"approved": provider}, approved_route=approved_test_route("approved")),
        primary_provider="approved",
        retriever_name="conflicted-pilot-retriever",
        reranker_name="identity-reranker",
        judge_name="must-not-run",
        retrieval_top_k=20,
        max_evidence_items=3,
        max_excerpt_chars=1200,
    )

    outcome = asyncio.run(
        executor.execute(
            request_id="ticket19-conflict",
            user_id="knowledge-user",
            session_id="ticket19-session",
            question="What is the reviewed default for environment=production?",
        )
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "material_evidence_conflict"
    assert outcome.evidence_set is None
    assert judge.calls == 0
    assert provider.calls == 0


def test_pilot_provider_prompt_uses_only_the_frozen_evidence_set_and_visible_conditions() -> None:
    provider = _ObservedRecordingProvider(answer=_valid_answer())
    executor = EvidenceGatedAnswerExecutor(
        retriever=_SufficientPilotRetriever(),
        reranker=_IdentityReranker(),
        judge=_FailIfCalledJudge(),
        provider_router=ProviderRouter(providers={"approved": provider}, approved_route=approved_test_route("approved")),
        primary_provider="approved",
        retriever_name="pilot-retriever",
        reranker_name="identity-reranker",
        judge_name="must-not-run",
        retrieval_top_k=20,
        max_evidence_items=3,
        max_excerpt_chars=1200,
    )

    question = "What is the reviewed default for environment=production?"
    query_conditions = QueryConditionSet.from_records(
        normalized_question=question,
        records=[
            {
                "condition_id": "environment-production",
                "field": "environment",
                "operator": "equals",
                "value": "production",
            }
        ],
    )
    outcome = asyncio.run(
        executor.execute(
            request_id="ticket19-sufficient",
            user_id="knowledge-user",
            session_id="ticket19-session",
            question=question,
            query_conditions=query_conditions,
        )
    )

    assert outcome.kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
    assert outcome.evidence_set is not None
    assert outcome.query_conditions == query_conditions
    assert provider.calls == 1
    envelope = json.loads(provider.prompts[0])
    assert envelope["query_condition_set"] == {
        "identity": outcome.evidence_set.query_conditions.identity,
        "conditions": [
            {
                "condition_id": "environment-production",
                "field": "environment",
                "operator": "equals",
                "value": "production",
            }
        ],
    }
    assert envelope["response_contract"]["governing_citation_id"] == outcome.evidence_set.governing_citation.marker
    assert envelope["response_contract"]["governing_section_id"] == "recommendation_or_reviewed_branches"
    assert envelope["evidence_sources"][0]["citation_id"] == outcome.evidence_set.citations[0].marker
    serialized = json.dumps(envelope, ensure_ascii=False)
    for forbidden in ("score", "999.0", "chunk-"):
        assert forbidden not in serialized
    assert envelope["evidence_sources"][0]["snapshot_id"] == outcome.evidence_set.citations[0].snapshot_id
    assert envelope["evidence_sources"][0]["item_identity"] == outcome.evidence_set.citations[0].item_identity
    assert envelope["evidence_sources"][0]["citation_identity"] == outcome.evidence_set.citations[0].identity
    trace = outcome.to_rag_trace()
    frozen_item = trace["answer_evidence_set"]["items"][0]
    summary = outcome.evidence_summary()
    assert summary["sources"][0]["citation_id"] == outcome.evidence_set.citations[0].marker
    assert summary["sources"][0]["citation_identity"] == outcome.evidence_set.citations[0].identity
    assert summary["sources"][0]["snapshot_id"] == frozen_item["snapshot_id"]
    assert summary["sources"][0]["source_access_scope"] == "controlled_internal"
    assert summary["provider_prompt_snapshot_ids"] == [frozen_item["snapshot_id"]]
    assert outcome.runtime["provider_prompt_snapshot_ids"] == (frozen_item["snapshot_id"],)
    assert summary["provider_generation_envelope"]["identity"] == outcome.runtime["provider_generation_envelope"]["identity"]


def test_frozen_evidence_display_metadata_cannot_be_mutated_by_candidates_or_projections() -> None:
    candidate = _candidate(section_id="recommendation_or_reviewed_branches")
    candidate["metadata"]["applicability_conditions"] = [
        {
            "condition_id": "environment-production",
            "field": "environment",
            "operator": "equals",
            "value": "production",
        }
    ]
    candidate["metadata"]["non_applicability_conditions"] = [
        {
            "condition_id": "environment-staging",
            "field": "environment",
            "operator": "equals",
            "value": "staging",
        }
    ]

    class _DisplayMetadataRetriever:
        async def retrieve(self, _query: str, top_k: int) -> RetrieveResult:
            del top_k
            return RetrieveResult(
                items=[candidate],
                strategy="sparse_bm25",
                lexical_candidate_count=1,
                merged_count=1,
                profile_identity=PILOT_RETRIEVAL_PROFILE_ID,
                candidate_pool_scope="published_knowledge",
            )

    provider = _ObservedRecordingProvider(answer=_valid_answer())
    executor = EvidenceGatedAnswerExecutor(
        retriever=_DisplayMetadataRetriever(),
        reranker=_IdentityReranker(),
        judge=_FailIfCalledJudge(),
        provider_router=ProviderRouter(providers={"approved": provider}, approved_route=approved_test_route("approved")),
        primary_provider="approved",
        retriever_name="display-metadata-retriever",
        reranker_name="identity-reranker",
        judge_name="must-not-run",
        retrieval_top_k=20,
        max_evidence_items=3,
        max_excerpt_chars=1200,
    )
    question = "What is the reviewed default for environment=production?"
    outcome = asyncio.run(
        executor.execute(
            request_id="ticket21-display-metadata",
            user_id="knowledge-user",
            session_id="ticket21-session",
            question=question,
        )
    )

    assert outcome.kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
    assert outcome.evidence_set is not None
    candidate["metadata"]["applicability_conditions"][0]["value"] = "mutated-candidate"
    frozen_record = outcome.evidence_set.to_record()
    assert frozen_record["items"][0]["evidence"]["metadata"]["applicability_conditions"][0]["value"] == "production"

    frozen_record["items"][0]["evidence"]["metadata"]["applicability_conditions"][0]["value"] = "mutated-record"
    assert (
        outcome.evidence_set.to_record()["items"][0]["evidence"]["metadata"]["applicability_conditions"][0]["value"]
        == "production"
    )

    execution_result = {
        "outcome": outcome.kind.value,
        "evidence_set": outcome.evidence_set.to_record(),
    }
    summary = evidence_summary_from_execution(execution_result)
    summary["sources"][0]["applicability_conditions"][0]["value"] = "mutated-summary"
    assert (
        evidence_summary_from_execution(execution_result)["sources"][0]["applicability_conditions"][0]["value"]
        == "production"
    )


def test_frozen_evidence_summary_rejects_snapshot_substitution_even_when_the_snapshot_hash_is_recomputed() -> None:
    provider = _ObservedRecordingProvider(answer=_valid_answer())
    executor = EvidenceGatedAnswerExecutor(
        retriever=_SufficientPilotRetriever(),
        reranker=_IdentityReranker(),
        judge=_FailIfCalledJudge(),
        provider_router=ProviderRouter(providers={"approved": provider}, approved_route=approved_test_route("approved")),
        primary_provider="approved",
        retriever_name="pilot-retriever",
        reranker_name="identity-reranker",
        judge_name="must-not-run",
        retrieval_top_k=20,
        max_evidence_items=3,
        max_excerpt_chars=1200,
    )
    outcome = asyncio.run(
        executor.execute(
            request_id="ticket19-snapshot-substitution",
            user_id="knowledge-user",
            session_id="ticket19-session",
            question="What is the reviewed default for environment=production?",
        )
    )
    trace = json.loads(json.dumps(outcome.to_rag_trace()))
    frozen_item = trace["answer_evidence_set"]["items"][0]
    evidence = frozen_item["evidence"]
    metadata = evidence["metadata"]
    evidence["content_preview"] = "A substituted snapshot must not keep the original item identity."
    substituted_snapshot_id = evidence_snapshot_id(
        title=metadata["title"],
        publication_version=metadata["publication_version"],
        excerpt=evidence["content_preview"],
        citation_metadata=metadata,
    )
    evidence["snapshot_id"] = substituted_snapshot_id
    frozen_item["snapshot_id"] = substituted_snapshot_id
    frozen_item["citation"]["snapshot_id"] = substituted_snapshot_id

    summary = evidence_summary_from_trace(trace)

    assert summary["coverage"] == "unavailable"
    assert summary["sources"] == []


@pytest.mark.parametrize(
    ("recommendation", "source_tier"),
    [
        ("I know from general knowledge that an unreviewed path is safe. [S1]", "primary_evidence_source"),
        ("This also applies when environment=staging. [S1]", "primary_evidence_source"),
        ("JWT_SECRET=ticket19-not-a-real-secret [S1]", "primary_evidence_source"),
        ("The API key is ticket19-not-a-real-secret. [S1]", "primary_evidence_source"),
        ("This also applies when region=us-east-1. [S1]", "primary_evidence_source"),
        ("This guarantees 99.99% availability. [S1]", "primary_evidence_source"),
        ("This always applies to every system. [S1]", "bounded_internal_case"),
        ("This holds without exception across every deployment. [S1]", "bounded_internal_case"),
        (
            "Use the deterministic reviewed default. [S1]\n"
            "This separate material conclusion has no citation.",
            "primary_evidence_source",
        ),
        ("Use the deterministic reviewed default. [S9]", "primary_evidence_source"),
    ],
)
def test_pilot_rejects_deterministic_adversarial_generation_contract_violations(
    recommendation: str,
    source_tier: str,
) -> None:
    class _AdversarialPilotRetriever:
        async def retrieve(self, _query: str, top_k: int) -> RetrieveResult:
            del top_k
            candidate = _candidate(section_id="recommendation_or_reviewed_branches")
            candidate["metadata"]["source_tier"] = source_tier
            return RetrieveResult(
                items=[candidate],
                strategy="sparse_bm25",
                lexical_candidate_count=1,
                merged_count=1,
                profile_identity=PILOT_RETRIEVAL_PROFILE_ID,
                candidate_pool_scope="published_knowledge",
            )

    provider = _RecordingProvider(answer=_valid_answer(recommendation=recommendation))
    executor = EvidenceGatedAnswerExecutor(
        retriever=_AdversarialPilotRetriever(),
        reranker=_IdentityReranker(),
        judge=_FailIfCalledJudge(),
        provider_router=ProviderRouter(providers={"approved": provider}, approved_route=approved_test_route("approved")),
        primary_provider="approved",
        retriever_name="adversarial-pilot-retriever",
        reranker_name="identity-reranker",
        judge_name="must-not-run",
        retrieval_top_k=20,
        max_evidence_items=3,
        max_excerpt_chars=1200,
    )

    outcome = asyncio.run(
        executor.execute(
            request_id="ticket19-adversarial",
            user_id="knowledge-user",
            session_id="ticket19-session",
            question="What is the reviewed default for environment=production?",
        )
    )

    assert outcome.kind is AnswerOutcomeKind.GENERATION_UNAVAILABLE
    assert outcome.evidence_set is not None
    assert provider.calls == 1
