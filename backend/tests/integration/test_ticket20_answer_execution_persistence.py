import asyncio
import hashlib
import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.requests import ClientDisconnect

import app.api.v1.chat as chat_api_module
from app.api.v1.chat import chat_stream
from app.chat.schemas import ChatRequest
from app.common.config import get_settings
from app.common.deps import get_current_user
from app.editorial_authority.service import EditorialAuthorityService
from app.extensions.provider_router import ProviderRouter
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.answer_execution import AnswerExecutionEventModel, AnswerExecutionModel
from app.model.base import Base
from app.model.chat import ChatMessage, ChatSession
from app.model.document import Document, DocumentChunk
from app.rag import answer_execution as answer_execution_module
from app.rag.answer_evidence import evidence_summary_from_execution
from app.rag.answer_execution import AnswerExecutionOutcome, AnswerOutcomeKind
from app.rag.evidence_sufficiency import QueryConditionSet, decide_answer_evidence
from app.rag.generation_observation import generation_envelope_observation, provider_visible_snapshot_ids
from app.rag.interfaces import GenerationCompletion, RetrieveResult
from app.repository.chat_repository import ChatRepository
from app.retrieval.policy import PILOT_RETRIEVAL_PROFILE_ID
from app.service import chat_service as chat_service_module
from app.service.answer_execution_store import AnswerExecutionStore
from app.service.chat_service import ChatService
from app.service.document_retrieval_service import MixedModeDocumentRetrieverService
from app.settings.runtime import get_system_settings_runtime
from tests.support.auth import create_authenticated_test_token
from tests.support.generation import approved_test_route

_KNOWLEDGE_QUESTION = "Which reviewed operating decision applies for environment=production?"
_PRODUCTION_CONDITION = {
    "condition_id": "environment-production",
    "field": "environment",
    "operator": "equals",
    "value": "production",
}


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(name): str(value) for name, value in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return 1 if key in self.hashes else 0


def test_answer_execution_event_sequence_is_unique_per_execution() -> None:
    unique_column_sets = {
        tuple(column.name for column in constraint.columns)
        for constraint in AnswerExecutionEventModel.__table__.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }
    assert ("execution_id", "sequence") in unique_column_sets


def _sse_event_data(payload: str, event: str) -> dict:
    for block in payload.split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0] == f"event: {event}" and lines[1].startswith("data: "):
            return json.loads(lines[1][6:])
    raise AssertionError(f"missing SSE {event!r} event")


def _semantic_execution(value: dict) -> dict:
    projection = {
        key: value[key]
        for key in (
            "state",
            "question",
            "query_condition_set",
            "condition_provenance",
            "outcome",
            "evidence_set_identity",
            "item_identities",
            "snapshot_ids",
            "knowledge_version_identities",
        )
    }
    if "insufficient_evidence_reply" in value:
        projection["insufficient_evidence_reply"] = value["insufficient_evidence_reply"]
    return projection


@contextmanager
def _client_context() -> Generator[tuple[TestClient, Any, _InMemoryRedis], None, None]:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    async def _init_db() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    asyncio.run(_init_db())
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.state.test_auth_redis = fake_redis
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, session_factory, fake_redis
    finally:
        app.dependency_overrides.clear()
        for attribute in ("test_auth_session_factory", "test_auth_redis"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)
        asyncio.run(db_engine.dispose())


def _headers(session_factory: Any, redis: _InMemoryRedis, *, username: str = "ticket20-user", role: str = "user") -> dict[str, str]:
    token = asyncio.run(
        create_authenticated_test_token(
            session_factory,
            redis,
            username=username,
            role=role,
        )
    )
    return {"Authorization": f"Bearer {token}"}


def _frozen_candidate(question: str, query_conditions: QueryConditionSet) -> dict:
    condition = (
        query_conditions.conditions[0].to_record()
        if query_conditions.conditions
        else dict(_PRODUCTION_CONDITION)
    )
    content = "Use the reviewed production decision and preserve the frozen evidence identity."
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    entry_id = "ticket20-decision"
    source_id = "ticket20-reviewed-source"
    return {
        "chunk_id": "ticket20-reviewed-chunk",
        "document_id": "ticket20-reviewed-document",
        "generation": 1,
        "chunk_index": 0,
        "content_preview": content,
        "content_length": len(content),
        "content_sha256": content_sha256,
        "answer_evidence_eligible": True,
        "entry_id": entry_id,
        "entry_identity": f"entry:{entry_id}",
        "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
        "publication_identity": f"published_knowledge_version:{entry_id}.v1",
        "publication_version": "v1",
        "section_id": "recommendation_or_reviewed_branches",
        "section_identity": f"entry:{entry_id}#recommendation_or_reviewed_branches",
        "decision_query": question,
        "chunk_identity": {
            "document_id": "ticket20-reviewed-document",
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
        "applicability_conditions": [condition],
        "freshness_triggers": [{"trigger_id": "source-change"}],
        "lifecycle_state": "published",
        "metadata": {
            "title": "Ticket 20 reviewed decision",
            "publication_version": "v1",
            "entry_id": entry_id,
            "entry_identity": f"entry:{entry_id}",
            "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
            "entry_title": "Ticket 20 reviewed decision",
            "domain": "answer-execution",
            "section_id": "recommendation_or_reviewed_branches",
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
            "decision_query": question,
        },
    }


def _sufficient_decision(question: str, query_conditions: QueryConditionSet):
    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=query_conditions,
        candidates=[_frozen_candidate(question, query_conditions)],
    )
    assert decision.is_sufficient
    return decision


def _outcome(
    kind: AnswerOutcomeKind,
    *,
    request_id: str,
    session_id: str,
    question: str,
    query_conditions: QueryConditionSet,
) -> AnswerExecutionOutcome:
    runtime = {
        "request_id": request_id,
        "session_id": session_id,
        "graph_alias": "ticket20-scripted",
        "steps": [],
        "provider_trace": {},
        "provider_attempts": [],
        "fallback_hops": 0,
    }
    if kind in {
        AnswerOutcomeKind.EVIDENCE_GATED_ANSWER,
        AnswerOutcomeKind.GENERATION_UNAVAILABLE,
    }:
        decision = _sufficient_decision(question, query_conditions)
        assert decision.evidence_set is not None
        return AnswerExecutionOutcome(
            kind=kind,
            text=(
                "A supported answer bounded by the frozen evidence."
                if kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
                else "Generation is unavailable after frozen evidence was accepted."
            ),
            evidence=decision.evidence_set.items,
            question=question,
            query_conditions=query_conditions,
            request_id=request_id,
            session_id=session_id,
            gate_passed=True,
            gate_reason="sufficient_evidence",
            steps=(),
            runtime=runtime,
            evidence_set=decision.evidence_set,
            sufficiency_decision=decision,
        )
    if kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY:
        decision = decide_answer_evidence(
            normalized_question=question,
            query_conditions=query_conditions,
            candidates=[],
        )
        assert decision.insufficient_reply is not None
        return AnswerExecutionOutcome(
            kind=kind,
            text="There is not enough reviewed evidence for this request.",
            evidence=(),
            question=question,
            query_conditions=query_conditions,
            request_id=request_id,
            session_id=session_id,
            gate_passed=False,
            gate_reason=decision.insufficient_reply.reason,
            steps=(),
            runtime=runtime,
            sufficiency_decision=decision,
        )
    assert kind is AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY
    return AnswerExecutionOutcome(
        kind=kind,
        text="A narrow non-knowledge-base reply.",
        evidence=(),
        question=question,
        query_conditions=query_conditions,
        request_id=request_id,
        session_id=session_id,
        gate_passed=None,
        gate_reason="not_applicable_non_knowledge_base",
        steps=(),
        runtime=runtime,
    )


def _scripted_executor(kind: AnswerOutcomeKind):
    class _ScriptedExecutor:
        calls: list[dict[str, object]] = []

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def execute(
            self,
            *,
            request_id: str,
            user_id: str,
            session_id: str,
            question: str,
            query_conditions: QueryConditionSet | None = None,
            progress=None,
        ) -> AnswerExecutionOutcome:
            del user_id, progress
            assert query_conditions is not None
            type(self).calls.append(
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "question": question,
                    "query_conditions": query_conditions.to_record(),
                }
            )
            return _outcome(
                kind,
                request_id=request_id,
                session_id=session_id,
                question=question,
                query_conditions=query_conditions,
            )

    return _ScriptedExecutor


def _assistant_execution(client: TestClient, headers: dict[str, str], session_id: str, execution_id: str) -> tuple[dict, dict]:
    history = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert history.status_code == 200
    assistant = next(
        message
        for message in history.json()["data"]["messages"]
        if message["type"] == "assistant" and message["answer_execution"]["id"] == execution_id
    )
    return history.json()["data"], assistant


def _stored_execution_result(session_factory: Any, *, execution_id: str, user_id: str) -> dict:
    async def _load() -> dict:
        async with session_factory() as session:
            loaded = await AnswerExecutionStore(session, ChatRepository(session)).load(
                execution_id=execution_id,
                user_id=user_id,
            )
            assert loaded is not None and loaded.result is not None
            return loaded.result

    return asyncio.run(_load())


def _blocking_executor(started: asyncio.Event):
    class _BlockingExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def execute(
            self,
            *,
            request_id: str,
            user_id: str,
            session_id: str,
            question: str,
            query_conditions: QueryConditionSet | None = None,
            progress=None,
        ) -> AnswerExecutionOutcome:
            del request_id, user_id, session_id, question, query_conditions, progress
            started.set()
            await asyncio.Future()
            raise AssertionError("the blocked executor cannot complete")

    return _BlockingExecutor


class _SufficientPilotRetriever:
    async def retrieve(self, query: str, top_k: int) -> RetrieveResult:
        return RetrieveResult(
            items=[_frozen_candidate(query, QueryConditionSet.from_records(
                normalized_question=query,
                records=[dict(_PRODUCTION_CONDITION)],
            ))][:top_k],
            strategy="sparse_bm25",
            lexical_candidate_count=1,
            merged_count=1,
            profile_identity=PILOT_RETRIEVAL_PROFILE_ID,
            candidate_pool_scope="published_knowledge",
        )


class _DeterministicProvider:
    def __init__(self, *, unavailable: bool) -> None:
        self.unavailable = unavailable
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
        self.prompts.append(prompt)
        generation_envelope = generation_envelope_observation(
            user_prompt=prompt,
            system_prompt=system_prompt,
            snapshot_ids=provider_visible_snapshot_ids(prompt),
        )
        if self.unavailable:
            return GenerationCompletion(text="", generation_envelope=generation_envelope)
        return GenerationCompletion(
            text=(
                "## Recommendation\n"
                "Use the reviewed production decision. [S1]\n\n"
                "## Applicability Limits\n"
                "This applies only when environment=production. [S1]\n\n"
                "## Alternatives\n"
                "Use a reviewed branch when the condition changes. [S1]\n\n"
                "## Minimal Implementation or Acceptance Check\n"
                "Keep the frozen snapshot and citation together. [S1]"
            ),
            generation_envelope=generation_envelope,
        )


class _MalformedEnvelopeProvider(_DeterministicProvider):
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
        completion = await super().complete(prompt, system_prompt=system_prompt)
        return GenerationCompletion(
            text=completion.text,
            generation_envelope={
                "identity": "not-a-valid-envelope",
                "snapshot_ids": ["not-a-snapshot"],
                "source_count": 1,
            },
        )


class _ExplodingProvider:
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
        del prompt, system_prompt
        raise RuntimeError("ticket20 provider exploded")


def _stream_request() -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/chat/stream",
            "raw_path": b"/api/v1/chat/stream",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )


def test_non_knowledge_execution_is_frozen_once_for_normal_sse_and_history() -> None:
    conditions = [
        {
            "condition_id": "environment-production",
            "field": "environment",
            "operator": "equals",
            "value": "production",
        }
    ]
    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        payload = {
            "message": "hello",
            "session_id": "ticket20-private-session",
            "query_conditions": conditions,
        }

        normal = client.post("/api/v1/chat", headers=headers, json=payload)
        assert normal.status_code == 200
        normal_data = normal.json()["data"]
        normal_execution = normal_data["answer_execution"]
        assert normal_data["message"]["answer_execution"] == normal_execution

        streamed = client.post("/api/v1/chat/stream", headers=headers, json=payload)
        assert streamed.status_code == 200
        stream_execution = _sse_event_data(streamed.text, "answer_execution")["answer_execution"]

        expected = {
            "state": "completed",
            "question": "hello",
            "query_condition_set": {
                "identity": normal_execution["query_condition_set"]["identity"],
                "normalized_question": "hello",
                "conditions": conditions,
            },
            "condition_provenance": {"mode": "explicit"},
            "outcome": "non_knowledge_base_reply",
            "evidence_set_identity": None,
            "item_identities": [],
            "snapshot_ids": [],
            "knowledge_version_identities": [],
        }
        assert _semantic_execution(normal_execution) == expected
        assert _semantic_execution(stream_execution) == expected
        assert normal_execution["id"] != stream_execution["id"]

        history = client.get("/api/v1/sessions/ticket20-private-session", headers=headers)
        assert history.status_code == 200
        assistant_messages = [
            message
            for message in history.json()["data"]["messages"]
            if message["type"] == "assistant"
        ]
        by_execution_id = {
            message["answer_execution"]["id"]: message["answer_execution"]
            for message in assistant_messages
        }
        assert by_execution_id[normal_execution["id"]] == normal_execution
        assert by_execution_id[stream_execution["id"]] == stream_execution


@pytest.mark.parametrize(
    "kind",
    (
        AnswerOutcomeKind.EVIDENCE_GATED_ANSWER,
        AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY,
        AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY,
        AnswerOutcomeKind.GENERATION_UNAVAILABLE,
    ),
)
def test_each_completed_outcome_has_one_cross_surface_execution_projection(
    monkeypatch: pytest.MonkeyPatch,
    kind: AnswerOutcomeKind,
) -> None:
    executor_type = _scripted_executor(kind)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)
    question = "hello" if kind is AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY else _KNOWLEDGE_QUESTION
    payload = {
        "message": question,
        "query_conditions": [dict(_PRODUCTION_CONDITION)],
    }

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        normal_session_id = f"ticket20-{kind.value}-normal"
        stream_session_id = f"ticket20-{kind.value}-stream"

        normal = client.post("/api/v1/chat", headers=headers, json={**payload, "session_id": normal_session_id})
        assert normal.status_code == 200
        normal_data = normal.json()["data"]
        normal_execution = normal_data["answer_execution"]
        assert normal_data["message"]["answer_execution"] == normal_execution
        assert normal_data["outcome"] == kind.value

        streamed = client.post("/api/v1/chat/stream", headers=headers, json={**payload, "session_id": stream_session_id})
        assert streamed.status_code == 200
        stream_execution = _sse_event_data(streamed.text, "answer_execution")["answer_execution"]
        assert _sse_event_data(streamed.text, "outcome") == {"outcome": kind.value}
        stream_evidence = _sse_event_data(streamed.text, "evidence_summary")["evidence_summary"]

        _normal_history, normal_assistant = _assistant_execution(
            client,
            headers,
            normal_session_id,
            normal_execution["id"],
        )
        _stream_history, stream_assistant = _assistant_execution(
            client,
            headers,
            stream_session_id,
            stream_execution["id"],
        )

        assert _semantic_execution(normal_execution) == _semantic_execution(stream_execution)
        assert normal_assistant["answer_execution"] == normal_execution
        assert stream_assistant["answer_execution"] == stream_execution
        assert normal_assistant["outcome"] == kind.value
        assert stream_assistant["outcome"] == kind.value
        assert normal_assistant["evidence_summary"] == stream_evidence
        assert stream_assistant["evidence_summary"] == stream_evidence
        if kind is AnswerOutcomeKind.GENERATION_UNAVAILABLE:
            assert stream_evidence == {
                "coverage": "unavailable",
                "source_count": 0,
                "sources": [],
            }
            assert "citation_id" not in json.dumps(stream_evidence)
            assert "snapshot_id" not in json.dumps(stream_evidence)
            assert "evidence_preview" not in stream_evidence
        if kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY:
            insufficient_reply = normal_execution["insufficient_evidence_reply"]
            assert normal_data["insufficient_evidence_reply"] == insufficient_reply
            assert normal_data["message"]["insufficient_evidence_reply"] == insufficient_reply
            assert stream_execution["insufficient_evidence_reply"] == insufficient_reply
            assert _sse_event_data(streamed.text, "insufficient_evidence_reply") == {
                "insufficient_evidence_reply": insufficient_reply
            }
            assert normal_assistant["insufficient_evidence_reply"] == insufficient_reply
            assert stream_assistant["insufficient_evidence_reply"] == insufficient_reply
        else:
            assert "insufficient_evidence_reply" not in normal_execution
            assert "insufficient_evidence_reply" not in normal_data
            assert "event: insufficient_evidence_reply" not in streamed.text
            assert "insufficient_evidence_reply" not in normal_assistant
            assert "insufficient_evidence_reply" not in stream_assistant
        normal_result = _stored_execution_result(
            session_factory,
            execution_id=normal_execution["id"],
            user_id="ticket20-user",
        )
        stream_result = _stored_execution_result(
            session_factory,
            execution_id=stream_execution["id"],
            user_id="ticket20-user",
        )
        for execution, result in (
            (normal_execution, normal_result),
            (stream_execution, stream_result),
        ):
            assert result["question"] == execution["question"]
            assert result["query_condition_set"] == execution["query_condition_set"]
            assert result["outcome"] == execution["outcome"]
            assert result["evidence_set_identity"] == execution["evidence_set_identity"]
            assert result["item_identities"] == execution["item_identities"]
            assert result["snapshot_ids"] == execution["snapshot_ids"]
            assert result["knowledge_version_identities"] == execution["knowledge_version_identities"]

        if kind in {
            AnswerOutcomeKind.EVIDENCE_GATED_ANSWER,
            AnswerOutcomeKind.GENERATION_UNAVAILABLE,
        }:
            expected_provider_input = {
                "question": normal_execution["question"],
                "query_condition_set_identity": normal_execution["query_condition_set"]["identity"],
                "evidence_set_identity": normal_execution["evidence_set_identity"],
                "item_identities": normal_execution["item_identities"],
                "snapshot_ids": normal_execution["snapshot_ids"],
                "knowledge_version_identities": normal_execution["knowledge_version_identities"],
            }
            assert normal_result["provider_input"] == expected_provider_input
            assert stream_result["provider_input"] == expected_provider_input
        else:
            assert normal_result["provider_input"] is None
            assert stream_result["provider_input"] is None
        assert len(executor_type.calls) == 2


@pytest.mark.parametrize(
    ("unavailable", "expected_outcome"),
    (
        (False, AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
        (True, AnswerOutcomeKind.GENERATION_UNAVAILABLE),
    ),
)
def test_real_executor_reuses_the_provider_payload_across_normal_sse_and_history(
    monkeypatch: pytest.MonkeyPatch,
    unavailable: bool,
    expected_outcome: AnswerOutcomeKind,
) -> None:
    provider = _DeterministicProvider(unavailable=unavailable)
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-deterministic-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_SufficientPilotRetriever(), "ticket20-sufficient-pilot-retriever"),
    )
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(
        ChatService,
        "_provider_router",
        lambda _self: ProviderRouter(
            providers={"ticket20-deterministic-provider": provider}, approved_route=approved_test_route("ticket20-deterministic-provider"),
        ),
    )

    try:
        with _client_context() as (client, session_factory, redis):
            headers = _headers(session_factory, redis)
            payload = {
                "message": _KNOWLEDGE_QUESTION,
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            }
            normal = client.post(
                "/api/v1/chat",
                headers=headers,
                json={**payload, "session_id": f"ticket20-real-{expected_outcome.value}-normal"},
            )
            streamed = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={**payload, "session_id": f"ticket20-real-{expected_outcome.value}-stream"},
            )
            assert normal.status_code == 200
            assert streamed.status_code == 200

            normal_data = normal.json()["data"]
            normal_execution = normal_data["answer_execution"]
            stream_execution = _sse_event_data(streamed.text, "answer_execution")["answer_execution"]
            assert normal_data["outcome"] == expected_outcome.value
            assert _sse_event_data(streamed.text, "outcome") == {"outcome": expected_outcome.value}
            assert _semantic_execution(normal_execution) == _semantic_execution(stream_execution)

            normal_result = _stored_execution_result(
                session_factory,
                execution_id=normal_execution["id"],
                user_id="ticket20-user",
            )
            stream_result = _stored_execution_result(
                session_factory,
                execution_id=stream_execution["id"],
                user_id="ticket20-user",
            )
            _normal_history, normal_assistant = _assistant_execution(
                client,
                headers,
                f"ticket20-real-{expected_outcome.value}-normal",
                normal_execution["id"],
            )
            _stream_history, stream_assistant = _assistant_execution(
                client,
                headers,
                f"ticket20-real-{expected_outcome.value}-stream",
                stream_execution["id"],
            )
            assert normal_assistant["answer_execution"] == normal_execution
            assert stream_assistant["answer_execution"] == stream_execution
            assert normal_assistant["outcome"] == expected_outcome.value
            assert stream_assistant["outcome"] == expected_outcome.value

            for result, execution, prompt in zip(
                (normal_result, stream_result),
                (normal_execution, stream_execution),
                provider.prompts,
                strict=True,
            ):
                provider_envelope = json.loads(prompt)
                assert result["provider_input"] == {
                    "question": execution["question"],
                    "query_condition_set_identity": execution["query_condition_set"]["identity"],
                    "evidence_set_identity": execution["evidence_set_identity"],
                    "item_identities": execution["item_identities"],
                    "snapshot_ids": execution["snapshot_ids"],
                    "knowledge_version_identities": execution["knowledge_version_identities"],
                }
                assert provider_envelope["user_question"] == result["question"]
                assert provider_envelope["query_condition_set"]["identity"] == result["query_condition_set"]["identity"]
                assert [source["item_identity"] for source in provider_envelope["evidence_sources"]] == result[
                    "item_identities"
                ]
                assert [source["snapshot_id"] for source in provider_envelope["evidence_sources"]] == result[
                    "snapshot_ids"
                ]
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def test_real_published_retrieval_projects_one_supported_execution_across_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DeterministicProvider(unavailable=False)
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-deterministic-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()

    async def _authority(
        _self,
        entry_id: str,
        *,
        now=None,
    ) -> dict:
        del now
        assert entry_id == "ticket20-decision"
        return {
            "entry_id": entry_id,
            "entry_identity": f"entry:{entry_id}",
            "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
            "lifecycle_state": "published",
            "answer_eligible": True,
            "eligibility_reasons": [],
            "section_source_relationships": {
                "recommendation_or_reviewed_branches": [
                    {
                        "source_identity": "source:ticket20-reviewed-source",
                        "availability": "verified_usable",
                        "access_scope": "controlled_internal",
                    }
                ]
            },
            "assurance_level": "source_grounded",
            "applicability_conditions": [dict(_PRODUCTION_CONDITION)],
            "freshness_triggers": [{"trigger_id": "source-change"}],
            "decision_query": _KNOWLEDGE_QUESTION,
        }

    monkeypatch.setattr(EditorialAuthorityService, "get_retrieval_authority", _authority)
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(
        ChatService,
        "_provider_router",
        lambda _self: ProviderRouter(
            providers={"ticket20-deterministic-provider": provider}, approved_route=approved_test_route("ticket20-deterministic-provider"),
        ),
    )

    try:
        with _client_context() as (client, session_factory, redis):
            async def _seed_published_source() -> None:
                query_conditions = QueryConditionSet.from_records(
                    normalized_question=_KNOWLEDGE_QUESTION,
                    records=[dict(_PRODUCTION_CONDITION)],
                )
                candidate = _frozen_candidate(_KNOWLEDGE_QUESTION, query_conditions)
                chunk_metadata = dict(candidate["metadata"])
                chunk_metadata.update(
                    {
                        "source_relationships": list(candidate["source_relationships"]),
                        "assurance_level": candidate["assurance_level"],
                        "applicability_conditions": list(candidate["applicability_conditions"]),
                        "freshness_triggers": list(candidate["freshness_triggers"]),
                    }
                )
                async with session_factory() as session:
                    session.add(
                        Document(
                            id="ticket20-reviewed-document",
                            filename="ticket20-reviewed-document.md",
                            file_type="md",
                            file_size=len(candidate["content_preview"]),
                            status="ready",
                            chunk_strategy="section-aware",
                            chunk_count=1,
                            published_generation=1,
                            next_generation=2,
                            latest_requested_generation=1,
                        )
                    )
                    session.add(
                        DocumentChunk(
                            id="ticket20-reviewed-chunk",
                            document_id="ticket20-reviewed-document",
                            generation=1,
                            chunk_index=0,
                            content=candidate["content_preview"],
                            keywords=["reviewed", "production", "decision"],
                            generated_questions=[_KNOWLEDGE_QUESTION],
                            chunk_metadata=chunk_metadata,
                        )
                    )
                    await session.commit()
                async with session_factory() as session:
                    retrieved = await MixedModeDocumentRetrieverService(session).retrieve(
                        _KNOWLEDGE_QUESTION,
                        top_k=3,
                    )
                    assert retrieved.items, retrieved.candidate_exclusions

            asyncio.run(_seed_published_source())
            headers = _headers(session_factory, redis)
            payload = {
                "message": _KNOWLEDGE_QUESTION,
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            }
            normal = client.post(
                "/api/v1/chat",
                headers=headers,
                json={**payload, "session_id": "ticket20-real-published-normal"},
            )
            streamed = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={**payload, "session_id": "ticket20-real-published-stream"},
            )
            assert normal.status_code == 200
            assert streamed.status_code == 200

            normal_execution = normal.json()["data"]["answer_execution"]
            stream_execution = _sse_event_data(streamed.text, "answer_execution")["answer_execution"]
            assert (
                normal_execution["outcome"] == AnswerOutcomeKind.EVIDENCE_GATED_ANSWER.value
            ), normal_execution.get("insufficient_evidence_reply")
            assert _semantic_execution(normal_execution) == _semantic_execution(stream_execution)
            assert normal_execution["snapshot_ids"] == stream_execution["snapshot_ids"]
            assert normal_execution["knowledge_version_identities"] == stream_execution[
                "knowledge_version_identities"
            ]

            _normal_history, normal_assistant = _assistant_execution(
                client,
                headers,
                "ticket20-real-published-normal",
                normal_execution["id"],
            )
            _stream_history, stream_assistant = _assistant_execution(
                client,
                headers,
                "ticket20-real-published-stream",
                stream_execution["id"],
            )
            assert normal_assistant["answer_execution"] == normal_execution
            assert stream_assistant["answer_execution"] == stream_execution
            assert normal_assistant["evidence_summary"]["source_count"] == 1
            assert stream_assistant["evidence_summary"] == normal_assistant["evidence_summary"]
            assert len(provider.prompts) == 2
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def test_provider_snapshot_contract_failure_is_persisted_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DeterministicProvider(unavailable=False)
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-deterministic-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_SufficientPilotRetriever(), "ticket20-sufficient-pilot-retriever"),
    )
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(
        ChatService,
        "_provider_router",
        lambda _self: ProviderRouter(
            providers={"ticket20-deterministic-provider": provider}, approved_route=approved_test_route("ticket20-deterministic-provider"),
        ),
    )
    provider_visible_generation_input = answer_execution_module.provider_visible_generation_input

    def _substitute_snapshot_identity(prompt: str):
        observed = provider_visible_generation_input(prompt)
        assert observed is not None
        return {**observed, "snapshot_ids": ["forged-snapshot"]}

    monkeypatch.setattr(
        answer_execution_module,
        "provider_visible_generation_input",
        _substitute_snapshot_identity,
    )

    try:
        with _client_context() as (client, session_factory, redis):
            headers = _headers(session_factory, redis)
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": _KNOWLEDGE_QUESTION,
                    "session_id": "ticket20-provider-input-contract-failure",
                    "query_conditions": [dict(_PRODUCTION_CONDITION)],
                },
            )
            assert response.status_code == 500
            assert "outcome" not in response.text
            assert provider.prompts == []

            history = client.get(
                "/api/v1/sessions/ticket20-provider-input-contract-failure",
                headers=headers,
            )
            assert history.status_code == 200
            assistant = next(
                message
                for message in history.json()["data"]["messages"]
                if message["type"] == "assistant"
            )
            assert assistant["answer_execution"]["state"] == "failed"
            assert "outcome" not in assistant
            assert "evidence_summary" not in assistant
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def test_provider_visible_input_contract_failure_is_persisted_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DeterministicProvider(unavailable=False)
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-deterministic-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_SufficientPilotRetriever(), "ticket20-sufficient-pilot-retriever"),
    )
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(
        ChatService,
        "_provider_router",
        lambda _self: ProviderRouter(
            providers={"ticket20-deterministic-provider": provider}, approved_route=approved_test_route("ticket20-deterministic-provider"),
        ),
    )
    build_generation_prompt = answer_execution_module.build_generation_prompt

    def _substitute_visible_input(question: str, evidence: object):
        prompt = build_generation_prompt(question, evidence)
        envelope = json.loads(prompt.user_prompt)
        envelope["user_question"] = "A substituted provider-visible question."
        envelope["query_condition_set"]["conditions"][0]["value"] = "staging"
        envelope["evidence_sources"][0]["item_identity"] = "f" * 64
        envelope["evidence_sources"][0]["snapshot_id"] = "e" * 64
        return type(prompt)(
            system_prompt=prompt.system_prompt,
            user_prompt=json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
        )

    monkeypatch.setattr(answer_execution_module, "build_generation_prompt", _substitute_visible_input)

    try:
        with _client_context() as (client, session_factory, redis):
            headers = _headers(session_factory, redis)
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": _KNOWLEDGE_QUESTION,
                    "session_id": "ticket20-provider-visible-input-contract-failure",
                    "query_conditions": [dict(_PRODUCTION_CONDITION)],
                },
            )
            assert response.status_code == 500
            assert "outcome" not in response.text
            assert provider.prompts == []

            history = client.get(
                "/api/v1/sessions/ticket20-provider-visible-input-contract-failure",
                headers=headers,
            )
            assert history.status_code == 200
            assistant = next(
                message
                for message in history.json()["data"]["messages"]
                if message["type"] == "assistant"
            )
            assert assistant["answer_execution"]["state"] == "failed"
            assert "outcome" not in assistant
            assert "evidence_summary" not in assistant
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def test_duplicate_provider_visible_json_keys_are_persisted_as_application_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DeterministicProvider(unavailable=False)
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-deterministic-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_SufficientPilotRetriever(), "ticket20-sufficient-pilot-retriever"),
    )
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(
        ChatService,
        "_provider_router",
        lambda _self: ProviderRouter(
            providers={"ticket20-deterministic-provider": provider}, approved_route=approved_test_route("ticket20-deterministic-provider"),
        ),
    )
    build_generation_prompt = answer_execution_module.build_generation_prompt

    def _duplicate_visible_input(question: str, evidence: object):
        prompt = build_generation_prompt(question, evidence)
        return type(prompt)(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt.replace(
                '"user_question":',
                '"user_question":"forged","user_question":',
                1,
            ),
        )

    monkeypatch.setattr(answer_execution_module, "build_generation_prompt", _duplicate_visible_input)

    try:
        with _client_context() as (client, session_factory, redis):
            headers = _headers(session_factory, redis)
            session_id = "ticket20-duplicate-provider-visible-key"
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": _KNOWLEDGE_QUESTION,
                    "session_id": session_id,
                    "query_conditions": [dict(_PRODUCTION_CONDITION)],
                },
            )
            assert response.status_code == 500
            assert "outcome" not in response.text
            assert provider.prompts == []

            history = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert history.status_code == 200
            assistant = next(
                message
                for message in history.json()["data"]["messages"]
                if message["type"] == "assistant"
            )
            assert assistant["answer_execution"]["state"] == "failed"
            assert "outcome" not in assistant
            assert "evidence_summary" not in assistant
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "mutation",
    (
        "citation_marker",
        "response_contract",
        "noncanonical_condition_value",
    ),
)
def test_provider_visible_input_rejects_all_frozen_envelope_mutations(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    provider = _DeterministicProvider(unavailable=False)
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-deterministic-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_SufficientPilotRetriever(), "ticket20-sufficient-pilot-retriever"),
    )
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(
        ChatService,
        "_provider_router",
        lambda _self: ProviderRouter(
            providers={"ticket20-deterministic-provider": provider}, approved_route=approved_test_route("ticket20-deterministic-provider"),
        ),
    )
    build_generation_prompt = answer_execution_module.build_generation_prompt

    def _substitute_visible_input(question: str, evidence: object):
        prompt = build_generation_prompt(question, evidence)
        envelope = json.loads(prompt.user_prompt)
        if mutation == "citation_marker":
            envelope["evidence_sources"][0]["citation_id"] = "S99"
        elif mutation == "response_contract":
            contract = envelope["response_contract"]
            assert isinstance(contract, dict)
            contract["citation_markers"] = ["S99"]
            contract["governing_citation_id"] = "S99"
        else:
            envelope["query_condition_set"]["conditions"][0]["value"] = " production "
        return type(prompt)(
            system_prompt=prompt.system_prompt,
            user_prompt=json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
        )

    monkeypatch.setattr(answer_execution_module, "build_generation_prompt", _substitute_visible_input)

    try:
        with _client_context() as (client, session_factory, redis):
            headers = _headers(session_factory, redis)
            session_id = f"ticket20-provider-envelope-{mutation}"
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": _KNOWLEDGE_QUESTION,
                    "session_id": session_id,
                    "query_conditions": [dict(_PRODUCTION_CONDITION)],
                },
            )
            assert response.status_code == 500
            assert provider.prompts == []

            history = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert history.status_code == 200
            assistant = next(
                message
                for message in history.json()["data"]["messages"]
                if message["type"] == "assistant"
            )
            assert assistant["answer_execution"]["state"] == "failed"
            assert "outcome" not in assistant
            assert "evidence_summary" not in assistant
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def test_malformed_provider_generation_envelope_is_persisted_as_application_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _MalformedEnvelopeProvider(unavailable=False)
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-deterministic-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_SufficientPilotRetriever(), "ticket20-sufficient-pilot-retriever"),
    )
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(
        ChatService,
        "_provider_router",
        lambda _self: ProviderRouter(
            providers={"ticket20-deterministic-provider": provider}, approved_route=approved_test_route("ticket20-deterministic-provider"),
        ),
    )

    try:
        with _client_context() as (client, session_factory, redis):
            headers = _headers(session_factory, redis)
            session_id = "ticket20-malformed-provider-envelope"
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": _KNOWLEDGE_QUESTION,
                    "session_id": session_id,
                    "query_conditions": [dict(_PRODUCTION_CONDITION)],
                },
            )
            assert response.status_code == 500
            assert "outcome" not in response.text

            history = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert history.status_code == 200
            assistant = next(
                message
                for message in history.json()["data"]["messages"]
                if message["type"] == "assistant"
            )
            assert assistant["answer_execution"]["state"] == "failed"
            assert "outcome" not in assistant
            assert "evidence_summary" not in assistant
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def test_unrecovered_provider_exception_is_persisted_as_application_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ExplodingProvider()
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-deterministic-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_SufficientPilotRetriever(), "ticket20-sufficient-pilot-retriever"),
    )
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(
        ChatService,
        "_provider_router",
        lambda _self: ProviderRouter(
            providers={"ticket20-deterministic-provider": provider}, approved_route=approved_test_route("ticket20-deterministic-provider"),
        ),
    )

    try:
        with _client_context() as (client, session_factory, redis):
            headers = _headers(session_factory, redis)
            session_id = "ticket20-provider-exception"
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": _KNOWLEDGE_QUESTION,
                    "session_id": session_id,
                    "query_conditions": [dict(_PRODUCTION_CONDITION)],
                },
            )
            assert response.status_code == 500
            assert "outcome" not in response.text

            history = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert history.status_code == 200
            assistant = next(
                message
                for message in history.json()["data"]["messages"]
                if message["type"] == "assistant"
            )
            assert assistant["answer_execution"]["state"] == "failed"
            assert "outcome" not in assistant
            assert "evidence_summary" not in assistant
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def test_missing_approved_route_is_persisted_as_generation_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", PILOT_RETRIEVAL_PROFILE_ID)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ticket20-missing-provider")
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_SufficientPilotRetriever(), "ticket20-sufficient-pilot-retriever"),
    )
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda _self: (None, "disabled-for-ticket20"))
    monkeypatch.setattr(ChatService, "_provider_router", lambda _self: ProviderRouter(providers={}))

    try:
        with _client_context() as (client, session_factory, redis):
            headers = _headers(session_factory, redis)
            session_id = "ticket20-missing-provider"
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": _KNOWLEDGE_QUESTION,
                    "session_id": session_id,
                    "query_conditions": [dict(_PRODUCTION_CONDITION)],
                },
            )
            assert response.status_code == 200
            assert response.json()["data"]["outcome"] == "generation_unavailable"

            history = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert history.status_code == 200
            assistant = next(
                message
                for message in history.json()["data"]["messages"]
                if message["type"] == "assistant"
            )
            assert assistant["answer_execution"]["state"] == "completed"
            assert assistant["outcome"] == "generation_unavailable"
            assert assistant["evidence_summary"]["sources"] == []
            assert assistant["answer_execution"]["snapshot_ids"] == response.json()["data"]["answer_execution"]["snapshot_ids"]
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def test_private_query_conditions_can_be_inherited_edited_and_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)
    production = dict(_PRODUCTION_CONDITION)
    staging = {
        "condition_id": "environment-staging",
        "field": "environment",
        "operator": "equals",
        "value": "staging",
    }

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        first = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "hello",
                "session_id": "ticket20-condition-session",
                "query_conditions": [production],
            },
        )
        assert first.status_code == 200
        first_execution = first.json()["data"]["answer_execution"]

        inherited_same_question = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "hello",
                "session_id": "ticket20-condition-session",
                "inherit_conditions": True,
            },
        )
        assert inherited_same_question.status_code == 200
        inherited_same_question_execution = inherited_same_question.json()["data"]["answer_execution"]
        assert inherited_same_question_execution["condition_provenance"] == {
            "mode": "inherited",
            "source_execution_id": first_execution["id"],
        }
        assert inherited_same_question_execution["query_condition_set"]["normalized_question"] == "hello"
        assert inherited_same_question_execution["query_condition_set"]["conditions"] == [production]
        assert (
            inherited_same_question_execution["query_condition_set"]["identity"]
            != first_execution["query_condition_set"]["identity"]
        )

        inherited = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "hello again",
                "session_id": "ticket20-condition-session",
                "inherit_conditions": True,
            },
        )
        assert inherited.status_code == 200
        inherited_execution = inherited.json()["data"]["answer_execution"]
        assert inherited_execution["condition_provenance"] == {
            "mode": "inherited",
            "source_execution_id": inherited_same_question_execution["id"],
        }
        assert inherited_execution["query_condition_set"]["normalized_question"] == "hello again"
        assert inherited_execution["query_condition_set"]["conditions"] == [production]
        assert inherited_execution["query_condition_set"]["identity"] != first_execution["query_condition_set"]["identity"]

        edited = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "hello after editing",
                "session_id": "ticket20-condition-session",
                "query_conditions": [staging],
            },
        )
        assert edited.status_code == 200
        edited_execution = edited.json()["data"]["answer_execution"]
        assert edited_execution["condition_provenance"] == {"mode": "explicit"}
        assert edited_execution["query_condition_set"]["conditions"] == [staging]

        no_conditions = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "hello without a hidden profile", "session_id": "ticket20-empty-conditions"},
        )
        assert no_conditions.status_code == 200
        empty_execution = no_conditions.json()["data"]["answer_execution"]
        assert empty_execution["condition_provenance"] == {"mode": "question_normalized"}
        assert empty_execution["query_condition_set"]["conditions"] == []

        reset = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "hello in another private conversation",
                "session_id": "ticket20-other-session",
                "inherit_conditions": True,
            },
        )
        assert reset.status_code == 422
        assert reset.json()["code"] == "QUERY_CONDITION_INHERITANCE_UNAVAILABLE"

        _history, first_assistant = _assistant_execution(
            client,
            headers,
            "ticket20-condition-session",
            first_execution["id"],
        )
        assert first_assistant["answer_execution"]["query_condition_set"]["conditions"] == [production]
        assert first_assistant["answer_execution"]["condition_provenance"] == {"mode": "explicit"}


def test_duplicate_query_condition_id_is_rejected_before_answer_execution_admission() -> None:
    duplicate_identity_conditions = [
        dict(_PRODUCTION_CONDITION),
        {
            "condition_id": _PRODUCTION_CONDITION["condition_id"],
            "field": "region",
            "operator": "equals",
            "value": "us-east",
        },
    ]

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "Which operating decision applies?",
                "session_id": "ticket20-duplicate-condition-id",
                "query_conditions": duplicate_identity_conditions,
            },
        )

    assert response.status_code == 422
    assert response.json()["code"] == "QUERY_CONDITION_INVALID"


def test_persisted_diagnostic_trace_excludes_private_turn_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-private-diagnostic-trace",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert response.status_code == 200

        async def _load_trace() -> dict:
            async with session_factory() as session:
                message = (
                    await session.scalars(
                        select(ChatMessage).where(
                            ChatMessage.session_id == "ticket20-private-diagnostic-trace",
                            ChatMessage.type == "assistant",
                        )
                    )
                ).one()
                assert isinstance(message.rag_trace, dict)
                return message.rag_trace

        trace = asyncio.run(_load_trace())
        serialized = json.dumps(trace, sort_keys=True)
        assert trace["outcome"] == AnswerOutcomeKind.EVIDENCE_GATED_ANSWER.value
        assert trace["gate"] == {"passed": True, "reason": "sufficient_evidence"}
        assert "query" not in trace
        assert "query_condition_set" not in trace
        assert "answer_preview" not in trace
        assert "evidence" not in trace
        assert "answer_evidence_set" not in trace
        assert "provider_generation_envelope" not in serialized
        assert _KNOWLEDGE_QUESTION not in serialized
        assert _PRODUCTION_CONDITION["value"] not in serialized
        assert "A supported answer bounded by the frozen evidence." not in serialized


def test_supported_answer_feedback_uses_the_frozen_execution_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        answer = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-feedback-frozen-execution",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert answer.status_code == 200

        feedback = client.post(
            "/api/v1/knowledge-feedback",
            headers=headers,
            json={
                "answer_id": answer.json()["data"]["message"]["id"],
                "entry_id": "ticket20-decision",
                "label": "helpful",
            },
        )
        assert feedback.status_code == 200
        assert feedback.json()["data"]["knowledge_edition"] == "publication:v1"


def test_insufficient_gap_feedback_uses_only_the_closed_execution_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        admin_headers = _headers(session_factory, redis, username="ticket20-feedback-admin", role="admin")
        answer = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-insufficient-gap-feedback",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert answer.status_code == 200
        answer_data = answer.json()["data"]
        execution = answer_data["answer_execution"]
        answer_id = answer_data["message"]["id"]
        gap_context = {
            "outcome": AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY.value,
            "reason": execution["insufficient_evidence_reply"]["reason"],
            "query_condition_set_identity": execution["query_condition_set"]["identity"],
        }

        invalid_scope = client.post(
            "/api/v1/knowledge-feedback",
            headers=headers,
            json={"answer_id": answer_id, "label": "helpful"},
        )
        assert invalid_scope.status_code == 422

        submitted = client.post(
            "/api/v1/knowledge-feedback",
            headers=headers,
            json={
                "answer_id": answer_id,
                "label": "insufficient_evidence",
                "note": "Missing reproducible deployment validation evidence.",
            },
        )
        assert submitted.status_code == 200
        signal = submitted.json()["data"]
        assert signal["answer_id"] == answer_id
        assert signal["entry_id"] is None
        assert signal["knowledge_edition"] is None
        assert signal["label"] == "insufficient_evidence"
        assert signal["gap_context"] == gap_context
        assert signal["duplicate"] is False
        serialized_signal = json.dumps(signal, sort_keys=True)
        assert _KNOWLEDGE_QUESTION not in serialized_signal
        assert "There is not enough reviewed evidence for this request." not in serialized_signal

        duplicate = client.post(
            "/api/v1/knowledge-feedback",
            headers=headers,
            json={
                "answer_id": answer_id,
                "label": "insufficient_evidence",
                "note": "Missing reproducible deployment validation evidence.",
            },
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["data"]["id"] == signal["id"]
        assert duplicate.json()["data"]["duplicate"] is True

        queue = client.get("/api/v1/knowledge-review-queue", headers=admin_headers)
        assert queue.status_code == 200
        item = next(
            item
            for item in queue.json()["data"]["items"]
            if item["metadata"].get("answer_id") == answer_id
        )
        assert item["subject_id"] == f"gap:{gap_context['reason']}"
        assert item["metadata"]["answer_id"] == answer_id
        assert item["metadata"]["gap_context"] == gap_context
        assert item["metadata"]["label"] == "insufficient_evidence"
        assert item["metadata"]["note"] == "Missing reproducible deployment validation evidence."
        serialized_queue = json.dumps(item, sort_keys=True)
        assert _KNOWLEDGE_QUESTION not in serialized_queue
        assert "There is not enough reviewed evidence for this request." not in serialized_queue


@pytest.mark.parametrize(
    "question",
    (
        " ".join(f"f{index}=v" for index in range(33)),
        f"{'f' * 80}={'v' * 160}",
    ),
)
def test_question_derived_conditions_respect_explicit_retry_limits(question: str) -> None:
    with _client_context() as (client, session_factory, redis):
        response = client.post(
            "/api/v1/chat",
            headers=_headers(session_factory, redis),
            json={"message": question, "session_id": "ticket20-implicit-condition-limit"},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "QUERY_CONDITION_INVALID"


def test_history_ignores_legacy_trace_outcome_and_source_count_for_new_executions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-trace-not-authoritative",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert response.status_code == 200
        execution = response.json()["data"]["answer_execution"]

        async def _corrupt_compatibility_trace() -> None:
            async with session_factory() as session:
                message = (
                    await session.scalars(
                        select(ChatMessage).where(
                            ChatMessage.answer_execution_id == execution["id"],
                            ChatMessage.type == "assistant",
                        )
                    )
                ).one()
                message.rag_trace = {
                    "outcome": "evidence_gated_answer",
                    "gate": {"passed": True},
                    "evidence": [
                        {
                            "document_id": "forged-document",
                            "content_preview": "This must never become a supported answer.",
                        }
                    ],
                }
                await session.commit()

        asyncio.run(_corrupt_compatibility_trace())
        _history, assistant = _assistant_execution(
            client,
            headers,
            "ticket20-trace-not-authoritative",
            execution["id"],
        )
        assert assistant["outcome"] == AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY.value
        assert assistant["evidence_summary"] == {"coverage": "insufficient", "source_count": 0, "sources": []}
        assert assistant["answer_execution"] == execution


def test_history_rejects_a_message_text_contradicting_the_closed_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "hello", "session_id": "ticket20-contradictory-text"},
        )
        assert response.status_code == 200
        execution = response.json()["data"]["answer_execution"]

        async def _corrupt_message_text() -> None:
            async with session_factory() as session:
                message = (
                    await session.scalars(
                        select(ChatMessage).where(
                            ChatMessage.answer_execution_id == execution["id"],
                            ChatMessage.type == "assistant",
                        )
                    )
                ).one()
                message.content = "A mutable message must not replace the closed answer."
                await session.commit()

        asyncio.run(_corrupt_message_text())
        history = client.get("/api/v1/sessions/ticket20-contradictory-text", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_history_rejects_a_user_question_contradicting_the_closed_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "hello", "session_id": "ticket20-contradictory-user-question"},
        )
        assert response.status_code == 200
        execution = response.json()["data"]["answer_execution"]

        async def _corrupt_user_question() -> None:
            async with session_factory() as session:
                message = (
                    await session.scalars(
                        select(ChatMessage).where(
                            ChatMessage.answer_execution_id == execution["id"],
                            ChatMessage.type == "user",
                        )
                    )
                ).one()
                message.content = "A mutable user message must not replace the admitted question."
                await session.commit()

        asyncio.run(_corrupt_user_question())
        history = client.get("/api/v1/sessions/ticket20-contradictory-user-question", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_history_rejects_a_missing_terminal_event_instead_of_projecting_a_partial_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "hello", "session_id": "ticket20-missing-terminal"},
        )
        assert response.status_code == 200
        execution = response.json()["data"]["answer_execution"]

        async def _remove_terminal_event() -> None:
            async with session_factory() as session:
                await session.execute(
                    delete(AnswerExecutionEventModel).where(
                        AnswerExecutionEventModel.execution_id == execution["id"],
                        AnswerExecutionEventModel.to_state == "completed",
                    )
                )
                await session.commit()

        asyncio.run(_remove_terminal_event())
        history = client.get("/api/v1/sessions/ticket20-missing-terminal", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_history_rejects_a_terminal_payload_that_contradicts_its_event_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": "hello", "session_id": "ticket20-terminal-state-mismatch"},
        )
        assert response.status_code == 200
        execution = response.json()["data"]["answer_execution"]

        async def _tamper_terminal_state() -> None:
            async with session_factory() as session:
                terminal = (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.execution_id == execution["id"],
                            AnswerExecutionEventModel.to_state == "completed",
                        )
                    )
                ).one()
                payload = json.loads(json.dumps(terminal.payload))
                payload["data"]["state"] = "failed"
                await session.execute(
                    AnswerExecutionEventModel.__table__.update()
                    .where(AnswerExecutionEventModel.__table__.c.id == terminal.id)
                    .values(payload=payload)
                )
                await session.commit()

        asyncio.run(_tamper_terminal_state())
        history = client.get("/api/v1/sessions/ticket20-terminal-state-mismatch", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_history_rejects_an_assistant_rebound_to_a_different_same_text_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)
    staging_condition = {**_PRODUCTION_CONDITION, "condition_id": "environment-staging", "value": "staging"}

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        first = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "hello",
                "session_id": "ticket20-message-rebinding",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        second = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "hello",
                "session_id": "ticket20-message-rebinding",
                "query_conditions": [staging_condition],
            },
        )
        assert first.status_code == 200
        assert second.status_code == 200
        first_execution = first.json()["data"]["answer_execution"]
        second_execution = second.json()["data"]["answer_execution"]

        async def _rebind_second_assistant() -> None:
            async with session_factory() as session:
                assistant = (
                    await session.scalars(
                        select(ChatMessage).where(
                            ChatMessage.answer_execution_id == second_execution["id"],
                            ChatMessage.type == "assistant",
                        )
                    )
                ).one()
                assistant.answer_execution_id = first_execution["id"]
                await session.commit()

        asyncio.run(_rebind_second_assistant())
        history = client.get("/api/v1/sessions/ticket20-message-rebinding", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_history_rejects_a_tampered_query_condition_set_in_the_closed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-tampered-conditions",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert response.status_code == 200
        execution = response.json()["data"]["answer_execution"]

        async def _tamper_terminal_conditions() -> None:
            async with session_factory() as session:
                terminal = (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.execution_id == execution["id"],
                            AnswerExecutionEventModel.to_state == "completed",
                        )
                    )
                ).one()
                payload = json.loads(json.dumps(terminal.payload))
                payload["data"]["query_condition_set"]["conditions"][0]["value"] = "staging"
                await session.execute(
                    AnswerExecutionEventModel.__table__.update()
                    .where(AnswerExecutionEventModel.__table__.c.id == terminal.id)
                    .values(payload=payload)
                )
                await session.commit()

        asyncio.run(_tamper_terminal_conditions())
        history = client.get("/api/v1/sessions/ticket20-tampered-conditions", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_history_rejects_provider_input_that_contradicts_the_frozen_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_type = _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-tampered-provider-input",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert response.status_code == 200
        execution = response.json()["data"]["answer_execution"]

        async def _tamper_terminal_provider_input() -> None:
            async with session_factory() as session:
                terminal = (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.execution_id == execution["id"],
                            AnswerExecutionEventModel.to_state == "completed",
                        )
                    )
                ).one()
                payload = json.loads(json.dumps(terminal.payload))
                payload["data"]["provider_input"]["snapshot_ids"] = ["forged-snapshot"]
                await session.execute(
                    AnswerExecutionEventModel.__table__.update()
                    .where(AnswerExecutionEventModel.__table__.c.id == terminal.id)
                    .values(payload=payload)
                )
                await session.commit()

        asyncio.run(_tamper_terminal_provider_input())
        history = client.get("/api/v1/sessions/ticket20-tampered-provider-input", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_missing_decisive_condition_is_a_structured_insufficiency_not_hidden_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingConditionExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def execute(
            self,
            *,
            request_id: str,
            user_id: str,
            session_id: str,
            question: str,
            query_conditions: QueryConditionSet | None = None,
            progress=None,
        ) -> AnswerExecutionOutcome:
            del user_id, progress
            assert query_conditions is not None
            decision = decide_answer_evidence(
                normalized_question=question,
                query_conditions=query_conditions,
                candidates=[_frozen_candidate(question, query_conditions)],
            )
            assert decision.insufficient_reply is not None
            assert decision.insufficient_reply.reason == "decisive_condition_missing"
            return AnswerExecutionOutcome(
                kind=AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY,
                text="A decisive condition is missing.",
                evidence=(),
                question=question,
                query_conditions=query_conditions,
                request_id=request_id,
                session_id=session_id,
                gate_passed=False,
                gate_reason=decision.insufficient_reply.reason,
                steps=(),
                runtime={"request_id": request_id, "session_id": session_id},
                sufficiency_decision=decision,
            )

    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", _MissingConditionExecutor)
    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": "Which reviewed operating decision applies?",
                "session_id": "ticket20-missing-condition",
            },
        )
        assert response.status_code == 200
        response_data = response.json()["data"]
        execution = response_data["answer_execution"]
        insufficient_reply = {
            "outcome": "insufficient_evidence_reply",
            "reason": "decisive_condition_missing",
            "query_condition_set_identity": execution["query_condition_set"]["identity"],
        }
        assert execution["outcome"] == AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY.value
        assert execution["query_condition_set"]["conditions"] == []
        assert execution["insufficient_evidence_reply"] == insufficient_reply
        assert response_data["insufficient_evidence_reply"] == insufficient_reply
        assert response_data["message"]["insufficient_evidence_reply"] == insufficient_reply
        assert response_data["message"]["evidence_summary"]["sources"] == []

        streamed = client.post(
            "/api/v1/chat/stream",
            headers=headers,
            json={
                "message": "Which reviewed operating decision applies?",
                "session_id": "ticket20-missing-condition-stream",
            },
        )
        assert streamed.status_code == 200
        stream_execution = _sse_event_data(streamed.text, "answer_execution")["answer_execution"]
        stream_reply = _sse_event_data(streamed.text, "insufficient_evidence_reply")["insufficient_evidence_reply"]
        assert stream_execution["insufficient_evidence_reply"] == stream_reply
        assert _sse_event_data(streamed.text, "outcome") == {
            "outcome": AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY.value
        }

        _history, assistant = _assistant_execution(
            client,
            headers,
            "ticket20-missing-condition",
            execution["id"],
        )
        assert assistant["insufficient_evidence_reply"] == insufficient_reply
        assert assistant["answer_execution"]["insufficient_evidence_reply"] == insufficient_reply

        async def _load_result() -> dict:
            async with session_factory() as session:
                loaded = await AnswerExecutionStore(session, ChatRepository(session)).load(
                    execution_id=execution["id"],
                    user_id="ticket20-user",
                )
                assert loaded is not None and loaded.result is not None
                return loaded.result

        result = asyncio.run(_load_result())
        assert result["insufficient_evidence_reply"] == insufficient_reply


def test_user_stop_persists_a_stopped_execution_without_a_completed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        started = asyncio.Event()
        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _blocking_executor(started),
        )

        async with session_factory() as session:
            task = asyncio.create_task(
                ChatService(session).run_chat(
                    user_id="ticket20-user",
                    question="Stop this execution.",
                    session_id="ticket20-user-stop",
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        async with session_factory() as session:
            messages = await ChatService(session).get_session_messages(
                session_id="ticket20-user-stop",
                user_id="ticket20-user",
                role="user",
            )
        assistant = next(message for message in messages if message["type"] == "assistant")
        assert assistant["content"] == ""
        assert assistant["answer_execution"]["state"] == "stopped"
        assert assistant["answer_execution"]["failure_code"] == "ANSWER_EXECUTION_STOPPED"
        assert "outcome" not in assistant
        assert "evidence_summary" not in assistant

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_cancellation_during_completion_persists_a_stopped_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        completion_started = asyncio.Event()
        executor_type = _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY)
        original_complete = AnswerExecutionStore.complete

        async def _blocked_complete(self, *, handle, outcome):
            completion_started.set()
            await asyncio.Future()
            return await original_complete(self, handle=handle, outcome=outcome)

        monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)
        monkeypatch.setattr(AnswerExecutionStore, "complete", _blocked_complete)

        async with session_factory() as session:
            task = asyncio.create_task(
                ChatService(session).run_chat(
                    user_id="ticket20-user",
                    question="hello",
                    session_id="ticket20-completion-cancellation",
                )
            )
            await asyncio.wait_for(completion_started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        async with session_factory() as session:
            messages = await ChatService(session).get_session_messages(
                session_id="ticket20-completion-cancellation",
                user_id="ticket20-user",
                role="user",
            )
        assistant = next(message for message in messages if message["type"] == "assistant")
        assert assistant["answer_execution"]["state"] == "stopped"
        assert assistant["answer_execution"]["failure_code"] == "ANSWER_EXECUTION_STOPPED"
        assert "outcome" not in assistant

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_stream_interruption_waits_for_the_stopped_execution_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        started = asyncio.Event()
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()
        original_stop = AnswerExecutionStore.stop

        async def _delayed_stop(self, *, handle):
            stop_started.set()
            await allow_stop.wait()
            return await original_stop(self, handle=handle)

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _blocking_executor(started),
        )
        monkeypatch.setattr(AnswerExecutionStore, "stop", _delayed_stop)

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(message="Interrupt this stream.", session_id="ticket20-stream-interruption"),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )
            identity_task = asyncio.create_task(anext(response.body_iterator))
            await asyncio.wait_for(started.wait(), timeout=1)
            if not identity_task.done():
                allow_stop.set()
                identity_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await identity_task
                pytest.fail("stream did not emit its reserved answer identity before execution began")
            identity_event = identity_task.result()
            assert identity_event.startswith("event: answer_identity")
            answer_id = _sse_event_data(identity_event, "answer_identity")["answer_id"]
            next_event = asyncio.create_task(anext(response.body_iterator))
            next_event.cancel()
            await asyncio.wait_for(stop_started.wait(), timeout=1)
            next_event.cancel()
            await asyncio.sleep(0)
            assert not next_event.done()
            allow_stop.set()
            with pytest.raises(asyncio.CancelledError):
                await next_event

        async with session_factory() as session:
            messages = await ChatService(session).get_session_messages(
                session_id="ticket20-stream-interruption",
                user_id="ticket20-user",
                role="user",
            )
        assistant = next(message for message in messages if message["type"] == "assistant")
        assert assistant["id"] == answer_id
        assert assistant["answer_execution"]["assistant_message_id"] == answer_id
        assert assistant["answer_execution"]["state"] == "stopped"
        assert "outcome" not in assistant
        assert "evidence_summary" not in assistant

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_completed_stream_interruption_fails_closed_for_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(message="hello", session_id="ticket20-closed-stream-interruption"),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )
            saw_answer_identity = False
            while True:
                event = await anext(response.body_iterator)
                saw_answer_identity = saw_answer_identity or event.startswith("event: answer_identity")
                if event.startswith("event: answer_execution"):
                    break
            assert saw_answer_identity
            await response.body_iterator.aclose()

        async with session_factory() as session:
            events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                        )
                    )
                ).all()
            )
            assert len(events) == 1
            with pytest.raises(ValueError, match="stream delivery was interrupted"):
                await ChatService(session).get_session_messages(
                    session_id="ticket20-closed-stream-interruption",
                    user_id="ticket20-user",
                    role="user",
                )

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_post_completion_stream_projection_exception_marks_history_non_projectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        def _raise_while_chunking(_text: str):
            raise RuntimeError("ticket20 post-completion stream projection failed")

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )
        monkeypatch.setattr(chat_api_module, "_chunk_text", _raise_while_chunking)

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(message="hello", session_id="ticket20-post-completion-projection-failure"),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )
            events = [event async for event in response.body_iterator]
            stream = "".join(events)
            assert "event: answer_execution" in stream
            assert "event: error" in stream
            assert "event: done" in stream

        async with session_factory() as session:
            interruption_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                        )
                    )
                ).all()
            )
            assert len(interruption_events) == 1
            with pytest.raises(ValueError, match="stream delivery was interrupted"):
                await ChatService(session).get_session_messages(
                    session_id="ticket20-post-completion-projection-failure",
                    user_id="ticket20-user",
                    role="user",
                )

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_completed_chat_task_exception_marks_history_non_projectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        original_run_chat = ChatService.run_chat

        async def _raise_after_completion_commit(self, *args, **kwargs):
            await original_run_chat(self, *args, **kwargs)
            raise RuntimeError("ticket20 completed chat task failed after commit")

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )
        monkeypatch.setattr(ChatService, "run_chat", _raise_after_completion_commit)

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(message="hello", session_id="ticket20-post-commit-task-failure"),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )
            stream = "".join([event async for event in response.body_iterator])
            assert "event: error" in stream
            assert "event: done" in stream
            assert "event: outcome" not in stream

        async with session_factory() as session:
            interruption_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                        )
                    )
                ).all()
            )
            assert len(interruption_events) == 1
            with pytest.raises(ValueError, match="stream delivery was interrupted"):
                await ChatService(session).get_session_messages(
                    session_id="ticket20-post-commit-task-failure",
                    user_id="ticket20-user",
                    role="user",
                )

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_cancellation_after_completion_commit_records_stream_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        completion_committed = asyncio.Event()
        original_run_chat = ChatService.run_chat

        async def _hold_after_completion_commit(self, *args, **kwargs):
            result = await original_run_chat(self, *args, **kwargs)
            completion_committed.set()
            await asyncio.Future()
            return result

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )
        monkeypatch.setattr(ChatService, "run_chat", _hold_after_completion_commit)

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(message="hello", session_id="ticket20-cancel-after-completion"),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )

            async def send(_message: dict) -> None:
                return None

            response_task = asyncio.create_task(response.stream_response(send))
            await asyncio.sleep(0)
            await asyncio.wait_for(completion_committed.wait(), timeout=1)
            response_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await response_task

        async with session_factory() as session:
            interruption_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                        )
                    )
                ).all()
            )
            assert len(interruption_events) == 1
            with pytest.raises(ValueError, match="stream delivery was interrupted"):
                await ChatService(session).get_session_messages(
                    session_id="ticket20-cancel-after-completion",
                    user_id="ticket20-user",
                    role="user",
                )

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_response_send_failure_after_a_completed_task_records_stream_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        execution_finished = asyncio.Event()
        original_run_chat = ChatService.run_chat

        class _ProgressThenCompleteExecutor:
            def __init__(self, **_kwargs: object) -> None:
                pass

            async def execute(
                self,
                *,
                request_id: str,
                user_id: str,
                session_id: str,
                question: str,
                query_conditions: QueryConditionSet | None = None,
                progress=None,
            ) -> AnswerExecutionOutcome:
                del user_id
                assert query_conditions is not None
                assert progress is not None
                await progress("retrieve", "the closed result is ready")
                return _outcome(
                    AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY,
                    request_id=request_id,
                    session_id=session_id,
                    question=question,
                    query_conditions=query_conditions,
                )

        async def _signal_completed_run_chat(self, *args, **kwargs):
            result = await original_run_chat(self, *args, **kwargs)
            execution_finished.set()
            return result

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _ProgressThenCompleteExecutor,
        )
        monkeypatch.setattr(ChatService, "run_chat", _signal_completed_run_chat)

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(message="hello", session_id="ticket20-send-failure-after-completion"),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )
            scope = dict(_stream_request().scope)
            scope["asgi"] = {"version": "3.0", "spec_version": "2.4"}

            async def receive() -> dict:
                await asyncio.Future()
                raise AssertionError("the ASGI 2.4 response should not receive")

            async def send(message: dict) -> None:
                body = message.get("body", b"")
                if isinstance(body, memoryview):
                    body = body.tobytes()
                if isinstance(body, bytes) and b"event: stage" in body:
                    await asyncio.wait_for(execution_finished.wait(), timeout=1)
                    raise OSError("client disconnected after the task completed")

            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)

        async with session_factory() as session:
            interruption_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                        )
                    )
                ).all()
            )
            assert len(interruption_events) == 1
            with pytest.raises(ValueError, match="stream delivery was interrupted"):
                await ChatService(session).get_session_messages(
                    session_id="ticket20-send-failure-after-completion",
                    user_id="ticket20-user",
                    role="user",
                )

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_response_send_failure_while_sending_done_records_stream_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(message="hello", session_id="ticket20-send-failure-on-done"),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )
            scope = dict(_stream_request().scope)
            scope["asgi"] = {"version": "3.0", "spec_version": "2.4"}

            async def receive() -> dict:
                await asyncio.Future()
                raise AssertionError("the ASGI 2.4 response should not receive")

            async def send(message: dict) -> None:
                body = message.get("body", b"")
                if isinstance(body, memoryview):
                    body = body.tobytes()
                if isinstance(body, bytes) and b"event: done" in body:
                    raise OSError("client disconnected while done was sent")

            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)

        async with session_factory() as session:
            interruption_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                        )
                    )
                ).all()
            )
            assert len(interruption_events) == 1
            with pytest.raises(ValueError, match="stream delivery was interrupted"):
                await ChatService(session).get_session_messages(
                    session_id="ticket20-send-failure-on-done",
                    user_id="ticket20-user",
                    role="user",
                )

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_outer_asgi_send_failure_after_inner_middleware_delivery_marks_history_non_projectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response-owned send hook must not mistake BaseHTTP buffering for delivery."""

    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async def _override_get_db_session() -> Generator[AsyncSession, None, None]:
            async with session_factory() as session:
                yield session

        async def _override_current_user() -> SimpleNamespace:
            return SimpleNamespace(username="ticket20-user", role="user")

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )
        app.dependency_overrides[get_db_session] = _override_get_db_session
        app.dependency_overrides[get_current_user] = _override_current_user
        request_body = json.dumps(
            {
                "message": "hello",
                "session_id": "ticket20-outer-asgi-send-failure",
            }
        ).encode("utf-8")
        request_received = False
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/chat/stream",
            "raw_path": b"/api/v1/chat/stream",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive() -> dict[str, object]:
            nonlocal request_received
            if not request_received:
                request_received = True
                return {"type": "http.request", "body": request_body, "more_body": False}
            await asyncio.Future()
            raise AssertionError("the ASGI response should not need another request body")

        async def send(message: dict[str, object]) -> None:
            body = message.get("body", b"")
            if isinstance(body, memoryview):
                body = body.tobytes()
            if isinstance(body, bytes) and b"event: done" in body:
                raise OSError("outer ASGI transport rejected the terminal body")

        try:
            with pytest.raises(OSError):
                await app(scope, receive, send)

            async with session_factory() as session:
                interruption_events = list(
                    (
                        await session.scalars(
                            select(AnswerExecutionEventModel).where(
                                AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                            )
                        )
                    ).all()
                )
                assert len(interruption_events) == 1
                with pytest.raises(ValueError, match="stream delivery was interrupted"):
                    await ChatService(session).get_session_messages(
                        session_id="ticket20-outer-asgi-send-failure",
                        user_id="ticket20-user",
                        role="user",
                    )
        finally:
            app.dependency_overrides.clear()

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_final_response_send_failure_after_done_marks_history_non_projectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async def _override_get_db_session() -> Generator[AsyncSession, None, None]:
            async with session_factory() as session:
                yield session

        async def _override_current_user() -> SimpleNamespace:
            return SimpleNamespace(username="ticket20-user", role="user")

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )
        app.dependency_overrides[get_db_session] = _override_get_db_session
        app.dependency_overrides[get_current_user] = _override_current_user
        request_body = json.dumps(
            {
                "message": "hello",
                "session_id": "ticket20-final-response-send-failure",
            }
        ).encode("utf-8")
        request_received = False
        terminal_body_sent = False
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/chat/stream",
            "raw_path": b"/api/v1/chat/stream",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive() -> dict[str, object]:
            nonlocal request_received
            if not request_received:
                request_received = True
                return {"type": "http.request", "body": request_body, "more_body": False}
            await asyncio.Future()
            raise AssertionError("the ASGI response should not need another request body")

        async def send(message: dict[str, object]) -> None:
            nonlocal terminal_body_sent
            body = message.get("body", b"")
            if isinstance(body, memoryview):
                body = body.tobytes()
            if isinstance(body, bytes) and b"event: done" in body:
                terminal_body_sent = True
                return
            if message.get("type") == "http.response.body" and message.get("more_body") is False:
                assert terminal_body_sent
                raise OSError("outer ASGI transport rejected response finalization")

        try:
            with pytest.raises(OSError):
                await app(scope, receive, send)

            async with session_factory() as session:
                interruption_events = list(
                    (
                        await session.scalars(
                            select(AnswerExecutionEventModel).where(
                                AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                            )
                        )
                    ).all()
                )
                assert len(interruption_events) == 1
                with pytest.raises(ValueError, match="stream delivery was interrupted"):
                    await ChatService(session).get_session_messages(
                        session_id="ticket20-final-response-send-failure",
                        user_id="ticket20-user",
                        role="user",
                    )
        finally:
            app.dependency_overrides.clear()

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_completed_stream_is_non_projectable_until_outer_delivery_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async def _override_get_db_session() -> Generator[AsyncSession, None, None]:
            async with session_factory() as session:
                yield session

        async def _override_current_user() -> SimpleNamespace:
            return SimpleNamespace(username="ticket20-user", role="user")

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )
        app.dependency_overrides[get_db_session] = _override_get_db_session
        app.dependency_overrides[get_current_user] = _override_current_user
        request_body = json.dumps(
            {
                "message": "hello",
                "session_id": "ticket20-delivery-pending",
            }
        ).encode("utf-8")
        request_received = False
        terminal_body_sent = asyncio.Event()
        allow_finalization = asyncio.Event()
        response_task: asyncio.Task[None] | None = None
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/chat/stream",
            "raw_path": b"/api/v1/chat/stream",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive() -> dict[str, object]:
            nonlocal request_received
            if not request_received:
                request_received = True
                return {"type": "http.request", "body": request_body, "more_body": False}
            await asyncio.Future()
            raise AssertionError("the ASGI response should not need another request body")

        async def send(message: dict[str, object]) -> None:
            body = message.get("body", b"")
            if isinstance(body, memoryview):
                body = body.tobytes()
            if isinstance(body, bytes) and b"event: done" in body:
                terminal_body_sent.set()
                return
            if message.get("type") == "http.response.body" and message.get("more_body") is False:
                await allow_finalization.wait()

        try:
            response_task = asyncio.create_task(app(scope, receive, send))
            await asyncio.wait_for(terminal_body_sent.wait(), timeout=1)

            async with session_factory() as session:
                with pytest.raises(ValueError, match="stream delivery is pending"):
                    await ChatService(session).get_session_messages(
                        session_id="ticket20-delivery-pending",
                        user_id="ticket20-user",
                        role="user",
                    )

            allow_finalization.set()
            await asyncio.wait_for(response_task, timeout=1)

            async with session_factory() as session:
                messages = await ChatService(session).get_session_messages(
                    session_id="ticket20-delivery-pending",
                    user_id="ticket20-user",
                    role="user",
                )
                assistant = next(message for message in messages if message["type"] == "assistant")
                assert assistant["answer_execution"]["state"] == "completed"
                delivery_events = list(
                    (
                        await session.scalars(
                            select(AnswerExecutionEventModel.event_type)
                            .where(AnswerExecutionEventModel.execution_id == assistant["answer_execution"]["id"])
                            .order_by(AnswerExecutionEventModel.sequence.asc())
                        )
                    ).all()
                )
                pending_index = delivery_events.index("stream_delivery_pending")
                completed_index = delivery_events.index("stream_delivery_completed")
                assert pending_index < completed_index
                assert delivery_events[-1] == "stream_delivery_completed"
        finally:
            allow_finalization.set()
            if response_task is not None and not response_task.done():
                await asyncio.gather(response_task, return_exceptions=True)
            app.dependency_overrides.clear()

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_sse_rejects_withdrawn_evidence_after_identity_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        released = asyncio.Event()

        class DelayedExecutor(_scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER)):
            async def execute(self, **kwargs):
                outcome = await super().execute(**kwargs)
                await released.wait()
                return outcome

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            DelayedExecutor,
        )

        async with session_factory() as session:
            session.add(
                Document(
                    id="ticket20-reviewed-document",
                    filename="ticket20-reviewed-document.md",
                    file_type="md",
                    file_size=42,
                    status="ready",
                    published_generation=1,
                    chunk_count=1,
                )
            )
            await session.commit()

        async with session_factory() as stream_session:
            response = await chat_stream(
                ChatRequest(
                    message=_KNOWLEDGE_QUESTION,
                    session_id="ticket20-stream-withdrawal-refresh",
                    query_conditions=[dict(_PRODUCTION_CONDITION)],
                ),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                stream_session,
            )
            identity_event = await anext(response.body_iterator)
            assert identity_event.startswith("event: answer_identity")

            async with session_factory() as tombstone_session:
                document = await tombstone_session.get(Document, "ticket20-reviewed-document")
                assert document is not None
                document.deleted_at = datetime.now(UTC)
                await tombstone_session.commit()
            released.set()

            terminal_payload = "".join([chunk async for chunk in response.body_iterator])

        execution = _sse_event_data(terminal_payload, "answer_execution")["answer_execution"]
        assert execution["state"] == "failed"
        assert execution.get("outcome") is None
        assert execution.get("evidence_summary") is None
        assert "event: evidence_summary" not in terminal_payload
        assert "content_preview" not in json.dumps(execution)

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_stream_pending_delivery_write_failure_projects_the_admitted_failed_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async def _fail_pending_delivery(self, **_kwargs) -> None:
            raise RuntimeError("ticket20 pending delivery write failed")

        monkeypatch.setattr(
            ChatService,
            "record_stream_delivery_pending",
            _fail_pending_delivery,
        )

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(
                    message=_KNOWLEDGE_QUESTION,
                    session_id="ticket20-pending-delivery-write-failure",
                    query_conditions=[dict(_PRODUCTION_CONDITION)],
                ),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )
            payload = "".join([chunk async for chunk in response.body_iterator])

        answer_identity = _sse_event_data(payload, "answer_identity")["answer_id"]
        execution = _sse_event_data(payload, "answer_execution")["answer_execution"]
        error = _sse_event_data(payload, "error")
        assert execution["state"] == "failed"
        assert execution["question"] == _KNOWLEDGE_QUESTION
        assert execution["query_condition_set"]["conditions"] == [dict(_PRODUCTION_CONDITION)]
        assert execution["assistant_message_id"] == answer_identity
        assert error["code"] == "CHAT_STREAM_FAILED"
        assert "event: outcome" not in payload
        assert "event: evidence_summary" not in payload

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_stream_delivery_interruption_retries_after_a_transaction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )
        original_record = AnswerExecutionStore.record_stream_delivery_interruption
        attempts = 0

        async def _fail_once_after_transaction_error(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await self.session.execute(text("SELECT * FROM ticket20_missing_delivery_retry_table"))
            return await original_record(self, **kwargs)

        monkeypatch.setattr(
            AnswerExecutionStore,
            "record_stream_delivery_interruption",
            _fail_once_after_transaction_error,
        )

        async with session_factory() as session:
            response = await chat_stream(
                ChatRequest(message="hello", session_id="ticket20-delivery-retry"),
                _stream_request(),
                SimpleNamespace(username="ticket20-user", role="user"),
                session,
            )
            scope = dict(_stream_request().scope)
            scope["asgi"] = {"version": "3.0", "spec_version": "2.4"}

            async def receive() -> dict:
                await asyncio.Future()
                raise AssertionError("the ASGI 2.4 response should not receive")

            async def send(message: dict) -> None:
                body = message.get("body", b"")
                if isinstance(body, memoryview):
                    body = body.tobytes()
                if isinstance(body, bytes) and b"event: done" in body:
                    raise OSError("client disconnected while done was sent")

            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)

        async with session_factory() as session:
            assert attempts == 2
            interruption_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.event_type == "stream_delivery_interrupted"
                        )
                    )
                ).all()
            )
            assert len(interruption_events) == 1
            with pytest.raises(ValueError, match="stream delivery was interrupted"):
                await ChatService(session).get_session_messages(
                    session_id="ticket20-delivery-retry",
                    user_id="ticket20-user",
                    role="user",
                )

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_history_uses_immutable_message_binding_when_mutable_index_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        created = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-cleared-message-index",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert created.status_code == 200
        expected_execution = created.json()["data"]["answer_execution"]
        assistant_id = created.json()["data"]["message"]["id"]

        async def _clear_mutable_index() -> None:
            async with session_factory() as session:
                assistant = await session.get(ChatMessage, assistant_id)
                assert assistant is not None
                assistant.answer_execution_id = None
                assistant.rag_trace = {
                    "outcome": AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY.value,
                    "evidence": [],
                }
                await session.commit()

        asyncio.run(_clear_mutable_index())
        history = client.get("/api/v1/sessions/ticket20-cleared-message-index", headers=headers)
        assert history.status_code == 200
        assistant = next(
            message
            for message in history.json()["data"]["messages"]
            if message["id"] == assistant_id
        )
        assert assistant["answer_execution"] == expected_execution
        assert assistant["outcome"] == AnswerOutcomeKind.EVIDENCE_GATED_ANSWER.value
        assert assistant["evidence_summary"]["source_count"] == 1


def test_history_rejects_cross_conversation_message_relocation_even_with_a_cleared_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        created = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-original-private-session",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert created.status_code == 200
        assistant_id = created.json()["data"]["message"]["id"]

        async def _move_assistant_to_another_private_session() -> None:
            async with session_factory() as session:
                repository = ChatRepository(session)
                await repository.get_or_create_session(
                    session_id="ticket20-relocated-private-session",
                    user_id="ticket20-user",
                )
                assistant = await session.get(ChatMessage, assistant_id)
                assert assistant is not None
                assistant.session_id = "ticket20-relocated-private-session"
                assistant.answer_execution_id = None
                assistant.rag_trace = {
                    "outcome": AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY.value,
                    "evidence": [],
                }
                await session.commit()

        asyncio.run(_move_assistant_to_another_private_session())
        original_history = client.get("/api/v1/sessions/ticket20-original-private-session", headers=headers)
        assert original_history.status_code == 500
        assert original_history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in original_history.text
        history = client.get("/api/v1/sessions/ticket20-relocated-private-session", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_history_rejects_original_conversation_after_user_message_relocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        created = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-original-user-binding-session",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert created.status_code == 200
        execution_id = created.json()["data"]["answer_execution"]["id"]

        async def _move_user_to_another_private_session() -> None:
            async with session_factory() as session:
                repository = ChatRepository(session)
                await repository.get_or_create_session(
                    session_id="ticket20-relocated-user-binding-session",
                    user_id="ticket20-user",
                )
                user_message = (
                    await session.scalars(
                        select(ChatMessage).where(
                            ChatMessage.answer_execution_id == execution_id,
                            ChatMessage.type == "user",
                        )
                    )
                ).one()
                user_message.session_id = "ticket20-relocated-user-binding-session"
                user_message.answer_execution_id = None
                await session.commit()

        asyncio.run(_move_user_to_another_private_session())
        history = client.get("/api/v1/sessions/ticket20-original-user-binding-session", headers=headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_direct_load_rejects_non_null_frozen_message_index_for_another_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        source = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-index-source-session",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        foreign = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-index-foreign-session",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert source.status_code == 200
        assert foreign.status_code == 200
        source_execution_id = source.json()["data"]["answer_execution"]["id"]
        foreign_execution_id = foreign.json()["data"]["answer_execution"]["id"]

        async def _run() -> None:
            async with session_factory() as session:
                source_messages = list(
                    (
                        await session.scalars(
                            select(ChatMessage).where(
                                ChatMessage.answer_execution_id == source_execution_id,
                            )
                        )
                    ).all()
                )
                message_ids = {message.type: message.id for message in source_messages}
                assert set(message_ids) == {"user", "assistant"}

            for message_type in ("user", "assistant"):
                async with session_factory() as session:
                    message = await session.get(ChatMessage, message_ids[message_type])
                    assert message is not None
                    message.answer_execution_id = foreign_execution_id
                    await session.commit()

                async with session_factory() as session:
                    store = AnswerExecutionStore(session, ChatRepository(session))
                    with pytest.raises(ValueError, match="contradicts its frozen answer execution binding"):
                        await store.load(
                            execution_id=source_execution_id,
                            user_id="ticket20-user",
                            session_id="ticket20-index-source-session",
                        )

                async with session_factory() as session:
                    message = await session.get(ChatMessage, message_ids[message_type])
                    assert message is not None
                    message.answer_execution_id = source_execution_id
                    await session.commit()

        asyncio.run(_run())


def test_unindexed_assistant_history_does_not_reconstruct_unrelated_execution_trails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        unrelated = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-unindexed-unrelated-session",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        target = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-unindexed-target-session",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert unrelated.status_code == 200
        assert target.status_code == 200
        unrelated_execution_id = unrelated.json()["data"]["answer_execution"]["id"]
        target_assistant_id = target.json()["data"]["message"]["id"]

        async def _clear_target_index() -> None:
            async with session_factory() as session:
                assistant = await session.get(ChatMessage, target_assistant_id)
                assert assistant is not None
                assistant.answer_execution_id = None
                await session.commit()

        asyncio.run(_clear_target_index())
        original_events_for_execution = AnswerExecutionStore._events_for_execution

        async def _reject_unrelated_event_reconstruction(self, *, execution_id: str):
            if execution_id == unrelated_execution_id:
                raise AssertionError("unrelated execution trail must not be reconstructed")
            return await original_events_for_execution(self, execution_id=execution_id)

        monkeypatch.setattr(
            AnswerExecutionStore,
            "_events_for_execution",
            _reject_unrelated_event_reconstruction,
        )
        history = client.get("/api/v1/sessions/ticket20-unindexed-target-session", headers=headers)
        assert history.status_code == 200
        assistant = next(
            message
            for message in history.json()["data"]["messages"]
            if message["id"] == target_assistant_id
        )
        assert assistant["answer_execution"]["state"] == "completed"


def test_history_rejects_cross_owner_message_relocation_even_with_a_cleared_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_service_module,
        "EvidenceGatedAnswerExecutor",
        _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER),
    )

    with _client_context() as (client, session_factory, redis):
        owner_headers = _headers(session_factory, redis)
        relocated_owner = "ticket20-relocated-owner"
        relocated_headers = _headers(session_factory, redis, username=relocated_owner)
        created = client.post(
            "/api/v1/chat",
            headers=owner_headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-original-owner-private-session",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert created.status_code == 200
        assistant_id = created.json()["data"]["message"]["id"]

        async def _move_assistant_to_another_owner() -> None:
            async with session_factory() as session:
                repository = ChatRepository(session)
                await repository.get_or_create_session(
                    session_id="ticket20-relocated-owner-private-session",
                    user_id=relocated_owner,
                )
                assistant = await session.get(ChatMessage, assistant_id)
                assert assistant is not None
                assistant.user_id = relocated_owner
                assistant.session_id = "ticket20-relocated-owner-private-session"
                assistant.answer_execution_id = None
                assistant.rag_trace = {
                    "outcome": AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY.value,
                    "evidence": [],
                }
                await session.commit()

        asyncio.run(_move_assistant_to_another_owner())
        history = client.get("/api/v1/sessions/ticket20-relocated-owner-private-session", headers=relocated_headers)
        assert history.status_code == 500
        assert history.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in history.text


def test_real_retrieval_failure_persists_failed_instead_of_insufficiency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRetriever:
        async def retrieve(self, query: str, top_k: int) -> list[dict]:
            del query, top_k
            raise RuntimeError("ticket20 retriever unavailable")

    monkeypatch.setattr(
        ChatService,
        "_resolve_retriever",
        lambda _self: (_FailingRetriever(), "ticket20-failing-retriever"),
    )

    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        normal = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-real-retrieval-failure",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert normal.status_code == 500
        assert "outcome" not in normal.text
        assert "insufficient_evidence_reply" not in normal.text

        streamed = client.post(
            "/api/v1/chat/stream",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-real-retrieval-stream-failure",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert streamed.status_code == 200
        assert _sse_event_data(streamed.text, "error")["code"] == "CHAT_STREAM_FAILED"
        stream_execution = _sse_event_data(streamed.text, "answer_execution")["answer_execution"]
        assert _sse_event_data(streamed.text, "answer_identity") == {
            "answer_id": stream_execution["assistant_message_id"]
        }
        assert stream_execution == {
            "id": stream_execution["id"],
            "assistant_message_id": stream_execution["assistant_message_id"],
            "state": "failed",
            "question": _KNOWLEDGE_QUESTION,
            "query_condition_set": {
                "identity": stream_execution["query_condition_set"]["identity"],
                "normalized_question": _KNOWLEDGE_QUESTION,
                "conditions": [dict(_PRODUCTION_CONDITION)],
            },
            "condition_provenance": {"mode": "explicit"},
            "failure_code": "ANSWER_EXECUTION_FAILED",
        }
        assert "event: outcome" not in streamed.text
        assert "insufficient_evidence_reply" not in streamed.text

        for session_id in (
            "ticket20-real-retrieval-failure",
            "ticket20-real-retrieval-stream-failure",
        ):
            history = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert history.status_code == 200
            assistant = next(
                message
                for message in history.json()["data"]["messages"]
                if message["type"] == "assistant"
            )
            assert assistant["answer_execution"]["state"] == "failed"
            assert assistant["answer_execution"]["failure_code"] == "ANSWER_EXECUTION_FAILED"
            assert "outcome" not in assistant
            assert "insufficient_evidence_reply" not in assistant


def test_execution_and_persistence_failures_are_not_reclassified_as_insufficiency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def execute(self, **_kwargs: object) -> AnswerExecutionOutcome:
            raise RuntimeError("retrieval failed")

    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", _FailingExecutor)
    with _client_context() as (client, session_factory, redis):
        unauthenticated = client.post(
            "/api/v1/chat",
            json={"message": _KNOWLEDGE_QUESTION, "session_id": "ticket20-auth-failure"},
        )
        assert unauthenticated.status_code == 401
        assert "outcome" not in unauthenticated.text
        assert "insufficient_evidence_reply" not in unauthenticated.text

        headers = _headers(session_factory, redis)
        auth_failure_history = client.get("/api/v1/sessions/ticket20-auth-failure", headers=headers)
        assert auth_failure_history.status_code == 200
        assert auth_failure_history.json()["data"]["messages"] == []

        normal = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": _KNOWLEDGE_QUESTION, "session_id": "ticket20-normal-failure"},
        )
        assert normal.status_code == 500
        assert "outcome" not in normal.text

        streamed = client.post(
            "/api/v1/chat/stream",
            headers=headers,
            json={"message": _KNOWLEDGE_QUESTION, "session_id": "ticket20-stream-failure"},
        )
        assert streamed.status_code == 200
        assert _sse_event_data(streamed.text, "error")["code"] == "CHAT_STREAM_FAILED"
        assert "event: outcome" not in streamed.text
        assert "insufficient_evidence_reply" not in streamed.text

        _history, normal_assistant = _assistant_execution(
            client,
            headers,
            "ticket20-normal-failure",
            next(
                message["answer_execution"]["id"]
                for message in client.get(
                    "/api/v1/sessions/ticket20-normal-failure",
                    headers=headers,
                ).json()["data"]["messages"]
                if message["type"] == "assistant"
            ),
        )
        _history, stream_assistant = _assistant_execution(
            client,
            headers,
            "ticket20-stream-failure",
            next(
                message["answer_execution"]["id"]
                for message in client.get(
                    "/api/v1/sessions/ticket20-stream-failure",
                    headers=headers,
                ).json()["data"]["messages"]
                if message["type"] == "assistant"
            ),
        )
        for assistant in (normal_assistant, stream_assistant):
            assert assistant["answer_execution"]["state"] == "failed"
            assert assistant["answer_execution"]["failure_code"] == "ANSWER_EXECUTION_FAILED"
            assert "outcome" not in assistant
            assert "evidence_summary" not in assistant

    executor_type = _scripted_executor(AnswerOutcomeKind.EVIDENCE_GATED_ANSWER)
    monkeypatch.setattr(chat_service_module, "EvidenceGatedAnswerExecutor", executor_type)
    original_add_message = ChatRepository.add_message

    async def _fail_assistant_persistence(self, *args, **kwargs):
        if kwargs.get("message_type") == "assistant":
            raise OSError("assistant persistence failed")
        return await original_add_message(self, *args, **kwargs)

    monkeypatch.setattr(ChatRepository, "add_message", _fail_assistant_persistence)
    with _client_context() as (client, session_factory, redis):
        headers = _headers(session_factory, redis)
        persistence = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "message": _KNOWLEDGE_QUESTION,
                "session_id": "ticket20-persistence-failure",
                "query_conditions": [dict(_PRODUCTION_CONDITION)],
            },
        )
        assert persistence.status_code == 500
        assert persistence.json()["code"] == "INTERNAL_ERROR"
        assert "outcome" not in persistence.text

        history = client.get("/api/v1/sessions/ticket20-persistence-failure", headers=headers)
        assert history.status_code == 200
        messages = history.json()["data"]["messages"]
        assert len(messages) == 1
        assert messages[0]["type"] == "user"
        assert messages[0]["content"] == _KNOWLEDGE_QUESTION
        assert messages[0]["answer_execution"]["state"] == "failed"
        assert messages[0]["answer_execution"]["failure_code"] == "ANSWER_EXECUTION_PERSISTENCE_FAILED"
        assert "outcome" not in messages[0]
        assert "evidence_summary" not in messages[0]


def test_persistence_recovery_terminal_is_idempotent_under_its_execution_lock() -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            repository = ChatRepository(session)
            chat_session = await repository.get_or_create_session(
                session_id="ticket20-persistence-recovery-lock",
                user_id="ticket20-user",
            )
            store = AnswerExecutionStore(session, repository)
            resolution = await store.resolve_query_conditions(
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                explicit_conditions=[dict(_PRODUCTION_CONDITION)],
                inherit_conditions=False,
            )
            handle = await store.admit(
                request_id="ticket20-persistence-recovery-lock-request",
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                resolution=resolution,
            )
            first = await store.fail_persistence_without_assistant_message(handle=handle)
            second = await store.fail_persistence_without_assistant_message(handle=handle)
            events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel)
                        .where(AnswerExecutionEventModel.execution_id == handle.execution_id)
                        .order_by(AnswerExecutionEventModel.sequence.asc())
                    )
                ).all()
            )
            await session.commit()

        assert first.projection == second.projection
        assert first.projection["state"] == "failed"
        assert first.projection["failure_code"] == "ANSWER_EXECUTION_PERSISTENCE_FAILED"
        assert [event.sequence for event in events] == [1, 2, 3, 4]
        assert [event.to_state for event in events] == ["admitted", "queued", "running", "failed"]

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_failed_stream_completion_check_releases_its_transaction() -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            repository = ChatRepository(session)
            chat_session = await repository.get_or_create_session(
                session_id="ticket20-failed-delivery-transaction",
                user_id="ticket20-user",
            )
            store = AnswerExecutionStore(session, repository)
            resolution = await store.resolve_query_conditions(
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                explicit_conditions=[dict(_PRODUCTION_CONDITION)],
                inherit_conditions=False,
            )
            handle = await store.admit(
                request_id="ticket20-failed-delivery-transaction-request",
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                resolution=resolution,
            )
            await store.fail(handle=handle, failure_code="ANSWER_EXECUTION_FAILED")
            await session.commit()

        async with session_factory() as session:
            recorded = await ChatService(session).record_stream_delivery_completion(
                execution_id=handle.execution_id,
                user_id=handle.user_id,
                session_id=handle.session_id,
            )
            assert recorded is False
            assert not session.in_transaction()

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


@pytest.mark.parametrize("expired", (False, True))
def test_private_execution_retention_cleanup_locks_header_before_event_trail(
    monkeypatch: pytest.MonkeyPatch,
    expired: bool,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    session_id = f"ticket20-retention-lock-{'expired' if expired else 'deleted'}"
    execution_id = f"answer-execution:ticket20-retention-lock:{'expired' if expired else 'deleted'}"
    created_at = datetime.now(UTC) - timedelta(days=31) if expired else datetime.now(UTC)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(
                ChatSession(
                    id=session_id,
                    user_id="ticket20-user",
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
            session.add(
                AnswerExecutionModel(
                    id=execution_id,
                    session_id=session_id,
                    user_id="ticket20-user",
                    initial_state="admitted",
                    request={},
                    created_at=created_at,
                )
            )
            session.add(
                AnswerExecutionEventModel(
                    execution_id=execution_id,
                    sequence=1,
                    event_type="created",
                    from_state=None,
                    to_state="admitted",
                    payload={"schema": "answer_execution_event/v1", "sequence": 1, "data": {}},
                    occurred_at=created_at,
                )
            )
            await session.commit()

        async with session_factory() as session:
            repository = ChatRepository(session)
            original_scalars = session.scalars
            lock_queries = []

            async def _capture_scalars(statement, *args, **kwargs):
                if "answer_executions" in str(statement):
                    lock_queries.append(statement)
                return await original_scalars(statement, *args, **kwargs)

            monkeypatch.setattr(session, "scalars", _capture_scalars)
            if expired:
                assert await repository.purge_expired_sessions(now=datetime.now(UTC)) == 1
            else:
                assert await repository.delete_session(session_id=session_id, user_id="ticket20-user")
            await session.commit()

            assert len(lock_queries) == (4 if expired else 2)
            assert all(query._for_update_arg is not None for query in lock_queries)
            assert await session.get(AnswerExecutionModel, execution_id) is None
            remaining_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.execution_id == execution_id
                        )
                    )
                ).all()
            )
            assert remaining_events == []

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_sqlite_private_retention_fence_blocks_competing_delivery_writer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'ticket20-retention-delivery-writer.db'}",
        connect_args={"timeout": 0.05},
    )
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    session_id = "ticket20-retention-delivery-writer"

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            repository = ChatRepository(session)
            chat_session = await repository.get_or_create_session(
                session_id=session_id,
                user_id="ticket20-user",
            )
            store = AnswerExecutionStore(session, repository)
            resolution = await store.resolve_query_conditions(
                user_id="ticket20-user",
                session_id=chat_session.id,
                question="hello",
                explicit_conditions=[],
                inherit_conditions=False,
            )
            handle = await store.admit(
                request_id="ticket20-retention-delivery-writer-request",
                user_id="ticket20-user",
                session_id=chat_session.id,
                question="hello",
                resolution=resolution,
            )
            assert await store.record_stream_delivery_pending(
                execution_id=handle.execution_id,
                user_id=handle.user_id,
                session_id=handle.session_id,
            )
            _message, completed = await store.complete(
                handle=handle,
                outcome=_outcome(
                    AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY,
                    request_id=handle.request_id,
                    session_id=handle.session_id,
                    question=handle.question,
                    query_conditions=handle.query_conditions,
                ),
            )
            await session.commit()

        writer_read = asyncio.Event()
        release_writer = asyncio.Event()
        original_events_for_execution = AnswerExecutionStore._events_for_execution
        paused = False

        async def _pause_writer_after_read(self, *, execution_id: str):
            nonlocal paused
            events = await original_events_for_execution(self, execution_id=execution_id)
            if execution_id == completed.projection["id"] and not paused:
                paused = True
                writer_read.set()
                await release_writer.wait()
            return events

        monkeypatch.setattr(AnswerExecutionStore, "_events_for_execution", _pause_writer_after_read)

        async with session_factory() as writer_session:
            writer_store = AnswerExecutionStore(writer_session, ChatRepository(writer_session))
            writer_task = asyncio.create_task(
                writer_store.record_stream_delivery_interruption(
                    execution_id=completed.projection["id"],
                    user_id="ticket20-user",
                    session_id=session_id,
                )
            )
            try:
                await asyncio.wait_for(writer_read.wait(), timeout=1)
                cleanup_blocked = False
                async with session_factory() as cleanup_session:
                    try:
                        await ChatRepository(cleanup_session).delete_session(
                            session_id=session_id,
                            user_id="ticket20-user",
                        )
                        await cleanup_session.commit()
                    except OperationalError:
                        cleanup_blocked = True
                        await cleanup_session.rollback()
            finally:
                release_writer.set()
            writer_result = (await asyncio.wait_for(asyncio.gather(writer_task, return_exceptions=True), timeout=1))[0]
            assert cleanup_blocked
            assert writer_result is True
            await writer_session.commit()

        async with session_factory() as cleanup_session:
            assert await ChatRepository(cleanup_session).delete_session(
                session_id=session_id,
                user_id="ticket20-user",
            )
            await cleanup_session.commit()

        async with session_factory() as verification_session:
            assert await verification_session.get(ChatSession, session_id) is None
            assert list(
                (
                    await verification_session.scalars(
                        select(ChatMessage).where(ChatMessage.session_id == session_id)
                    )
                ).all()
            ) == []
            assert list(
                (
                    await verification_session.scalars(
                        select(AnswerExecutionModel).where(AnswerExecutionModel.session_id == session_id)
                    )
                ).all()
            ) == []
            assert list(
                (
                    await verification_session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.execution_id == completed.projection["id"]
                        )
                    )
                ).all()
            ) == []

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_sqlite_admission_fence_blocks_competing_private_session_deletion_after_a_prior_read(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'ticket20-admission-retention-fence.db'}",
        connect_args={"timeout": 0.05},
    )
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    session_id = "ticket20-admission-retention-fence"

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            await ChatRepository(session).get_or_create_session(
                session_id=session_id,
                user_id="ticket20-user",
            )
            await session.commit()

        monkeypatch.setattr(
            chat_service_module,
            "EvidenceGatedAnswerExecutor",
            _scripted_executor(AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY),
        )
        admission_reached = asyncio.Event()
        release_admission = asyncio.Event()
        original_admit = AnswerExecutionStore.admit

        async def _pause_admission(self, **kwargs):
            if kwargs["session_id"] == session_id:
                admission_reached.set()
                await release_admission.wait()
            return await original_admit(self, **kwargs)

        monkeypatch.setattr(AnswerExecutionStore, "admit", _pause_admission)

        async with session_factory() as admission_session:
            # Authentication normally reads through this same request session
            # before ChatService attempts to acquire its SQLite write fence.
            await admission_session.execute(select(ChatSession.id).where(ChatSession.id == session_id))
            assert admission_session.in_transaction()
            admission_task = asyncio.create_task(
                ChatService(admission_session).run_chat(
                    user_id="ticket20-user",
                    question="hello",
                    session_id=session_id,
                    query_conditions=[],
                )
            )
            try:
                await asyncio.wait_for(admission_reached.wait(), timeout=1)
                cleanup_blocked = False
                async with session_factory() as cleanup_session:
                    try:
                        await ChatRepository(cleanup_session).delete_session(
                            session_id=session_id,
                            user_id="ticket20-user",
                        )
                        await cleanup_session.commit()
                    except OperationalError:
                        cleanup_blocked = True
                        await cleanup_session.rollback()
            finally:
                release_admission.set()
            result = (await asyncio.wait_for(asyncio.gather(admission_task, return_exceptions=True), timeout=1))[0]
            assert cleanup_blocked
            assert not isinstance(result, BaseException)
            assert result["session_id"] == session_id

        async with session_factory() as cleanup_session:
            assert await ChatRepository(cleanup_session).delete_session(
                session_id=session_id,
                user_id="ticket20-user",
            )
            await cleanup_session.commit()

        async with session_factory() as verification_session:
            assert await verification_session.get(ChatSession, session_id) is None
            assert list(
                (
                    await verification_session.scalars(
                        select(ChatMessage).where(ChatMessage.session_id == session_id)
                    )
                ).all()
            ) == []
            assert list(
                (
                    await verification_session.scalars(
                        select(AnswerExecutionModel).where(AnswerExecutionModel.session_id == session_id)
                    )
                ).all()
            ) == []
            assert list((await verification_session.scalars(select(AnswerExecutionEventModel))).all()) == []

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_document_tombstone_appends_redaction_without_rewriting_frozen_execution() -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(
                Document(
                    id="ticket20-reviewed-document",
                    filename="ticket20-reviewed-document.md",
                    file_type="md",
                    file_size=42,
                    status="ready",
                    published_generation=1,
                    chunk_count=1,
                )
            )
            repository = ChatRepository(session)
            chat_session = await repository.get_or_create_session(
                session_id="ticket20-withdrawal-history",
                user_id="ticket20-user",
            )
            store = AnswerExecutionStore(session, repository)
            resolution = await store.resolve_query_conditions(
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                explicit_conditions=[dict(_PRODUCTION_CONDITION)],
                inherit_conditions=False,
            )
            handle = await store.admit(
                request_id="ticket20-withdrawal-request",
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                resolution=resolution,
            )
            _message, initial = await store.complete(
                handle=handle,
                outcome=_outcome(
                    AnswerOutcomeKind.EVIDENCE_GATED_ANSWER,
                    request_id=handle.request_id,
                    session_id=handle.session_id,
                    question=handle.question,
                    query_conditions=handle.query_conditions,
                ),
            )
            await session.commit()

        async with session_factory() as session:
            document = await session.get(Document, "ticket20-reviewed-document")
            assert document is not None
            await AnswerExecutionStore(session, ChatRepository(session)).redact_document_evidence(
                document_id=document.id,
            )
            await session.commit()

        async with session_factory() as session:
            store = AnswerExecutionStore(session, ChatRepository(session))
            loaded = await store.load(execution_id=initial.projection["id"], user_id="ticket20-user")
            assert loaded is not None and loaded.result is not None
            redaction_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.execution_id == initial.projection["id"],
                            AnswerExecutionEventModel.event_type == "evidence_redacted",
                        )
                    )
                ).all()
            )
            terminal_event = next(
                event
                for event in (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.execution_id == initial.projection["id"],
                            AnswerExecutionEventModel.to_state == "completed",
                        )
                    )
                ).all()
            )

        assert len(redaction_events) == 1
        assert {
            key: value
            for key, value in loaded.projection.items()
            if key != "evidence_summary"
        } == {
            key: value
            for key, value in initial.projection.items()
            if key != "evidence_summary"
        }
        assert loaded.result["evidence_set_identity"] == initial.result["evidence_set_identity"]
        assert loaded.result["item_identities"] == initial.result["item_identities"]
        assert loaded.result["snapshot_ids"] == initial.result["snapshot_ids"]
        assert loaded.result["knowledge_version_identities"] == initial.result["knowledge_version_identities"]
        frozen_item = loaded.result["evidence_set"]["items"][0]
        assert frozen_item["withdrawn"] is True
        assert frozen_item["evidence"]["withdrawn"] is True
        assert "content_preview" not in frozen_item["evidence"]
        summary = evidence_summary_from_execution(loaded.result)
        assert loaded.projection["evidence_summary"] == summary
        assert loaded.projection["evidence_summary"] != initial.projection["evidence_summary"]
        assert summary["sources"][0]["snapshot_id"] == initial.result["snapshot_ids"][0]
        assert summary["sources"][0]["withdrawal_notice"] == "This source has been withdrawn."
        assert "excerpt" not in summary["sources"][0]

        persisted_terminal_item = terminal_event.payload["data"]["evidence_set"]["items"][0]
        assert persisted_terminal_item["evidence"]["content_preview"]
        assert "withdrawn" not in persisted_terminal_item

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_document_tombstone_after_evidence_freeze_rejects_new_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(
                Document(
                    id="ticket20-reviewed-document",
                    filename="ticket20-reviewed-document.md",
                    file_type="md",
                    file_size=42,
                    status="ready",
                    published_generation=1,
                    chunk_count=1,
                )
            )
            repository = ChatRepository(session)
            chat_session = await repository.get_or_create_session(
                session_id="ticket20-withdrawal-before-completion",
                user_id="ticket20-user",
            )
            store = AnswerExecutionStore(session, repository)
            resolution = await store.resolve_query_conditions(
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                explicit_conditions=[dict(_PRODUCTION_CONDITION)],
                inherit_conditions=False,
            )
            handle = await store.admit(
                request_id="ticket20-withdrawal-before-completion-request",
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                resolution=resolution,
            )
            await session.commit()

        frozen_outcome = _outcome(
            AnswerOutcomeKind.EVIDENCE_GATED_ANSWER,
            request_id=handle.request_id,
            session_id=handle.session_id,
            question=handle.question,
            query_conditions=handle.query_conditions,
        )

        async with session_factory() as session:
            document = await session.get(Document, "ticket20-reviewed-document")
            assert document is not None
            document.deleted_at = datetime.now(UTC)
            await session.commit()

        async def _global_redaction_must_not_run(self, *, document_id: str) -> int:
            del self, document_id
            raise AssertionError("completion-time tombstone redaction must not scan every execution")

        monkeypatch.setattr(
            AnswerExecutionStore,
            "redact_document_evidence",
            _global_redaction_must_not_run,
        )

        async with session_factory() as session:
            store = AnswerExecutionStore(session, ChatRepository(session))
            _message, completed = await store.complete(handle=handle, outcome=frozen_outcome)
            await session.commit()

        async with session_factory() as session:
            store = AnswerExecutionStore(session, ChatRepository(session))
            loaded = await store.load(execution_id=completed.projection["id"], user_id="ticket20-user")
            assert loaded is not None and loaded.result is not None
            redaction_events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel).where(
                            AnswerExecutionEventModel.execution_id == completed.projection["id"],
                            AnswerExecutionEventModel.event_type == "evidence_redacted",
                        )
                    )
                ).all()
            )

        assert redaction_events == []
        assert loaded.projection["state"] == "failed"
        assert loaded.projection.get("outcome") is None
        assert loaded.result.get("evidence_set") is None

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())


def test_redaction_and_delivery_interruption_share_one_ordered_event_chain() -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _run() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(
                Document(
                    id="ticket20-reviewed-document",
                    filename="ticket20-reviewed-document.md",
                    file_type="md",
                    file_size=42,
                    status="ready",
                    published_generation=1,
                    chunk_count=1,
                )
            )
            repository = ChatRepository(session)
            chat_session = await repository.get_or_create_session(
                session_id="ticket20-redaction-delivery-order",
                user_id="ticket20-user",
            )
            store = AnswerExecutionStore(session, repository)
            resolution = await store.resolve_query_conditions(
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                explicit_conditions=[dict(_PRODUCTION_CONDITION)],
                inherit_conditions=False,
            )
            handle = await store.admit(
                request_id="ticket20-redaction-delivery-order-request",
                user_id="ticket20-user",
                session_id=chat_session.id,
                question=_KNOWLEDGE_QUESTION,
                resolution=resolution,
            )
            assert await store.record_stream_delivery_pending(
                execution_id=handle.execution_id,
                user_id=handle.user_id,
                session_id=handle.session_id,
            )
            _message, completed = await store.complete(
                handle=handle,
                outcome=_outcome(
                    AnswerOutcomeKind.EVIDENCE_GATED_ANSWER,
                    request_id=handle.request_id,
                    session_id=handle.session_id,
                    question=handle.question,
                    query_conditions=handle.query_conditions,
                ),
            )
            await session.commit()

        async with session_factory() as session:
            store = AnswerExecutionStore(session, ChatRepository(session))
            assert await store.redact_document_evidence(document_id="ticket20-reviewed-document") == 1
            await session.commit()

        async with session_factory() as session:
            store = AnswerExecutionStore(session, ChatRepository(session))
            assert await store.record_stream_delivery_interruption(
                execution_id=completed.projection["id"],
                user_id="ticket20-user",
                session_id="ticket20-redaction-delivery-order",
            )
            await session.commit()

        async with session_factory() as session:
            events = list(
                (
                    await session.scalars(
                        select(AnswerExecutionEventModel)
                        .where(AnswerExecutionEventModel.execution_id == completed.projection["id"])
                        .order_by(AnswerExecutionEventModel.sequence.asc())
                    )
                ).all()
            )
            assert [event.sequence for event in events] == list(range(1, len(events) + 1))
            assert len({event.sequence for event in events}) == len(events)
            assert [event.event_type for event in events[-2:]] == [
                "evidence_redacted",
                "stream_delivery_interrupted",
            ]
            store = AnswerExecutionStore(session, ChatRepository(session))
            with pytest.raises(ValueError, match="stream delivery was interrupted"):
                await store.load(execution_id=completed.projection["id"], user_id="ticket20-user")

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(db_engine.dispose())
