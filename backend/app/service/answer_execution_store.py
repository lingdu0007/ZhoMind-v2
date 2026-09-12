from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.contracts.canonical import (
    AnswerExecutionState,
    AnswerOutcome,
    StableIdentity,
    StableIdentityKind,
    validate_transition,
)
from app.maintenance.containment import answer_is_blocked
from app.model.answer_execution import AnswerExecutionEventModel, AnswerExecutionModel
from app.model.chat import ChatMessage
from app.model.document import Document
from app.rag.answer_evidence import evidence_summary_from_execution
from app.rag.answer_execution import AnswerExecutionOutcome
from app.rag.evidence_sufficiency import InsufficientEvidenceReply, QueryConditionSet
from app.repository.chat_repository import ChatRepository
from app.reviewed_bundles.withdrawal_facts import read_publication_withdrawals

_REQUEST_SCHEMA = "answer_execution_request/v1"
_EVENT_SCHEMA = "answer_execution_event/v1"
_RESULT_SCHEMA = "answer_execution_result/v1"
_EVIDENCE_REDACTED_EVENT = "evidence_redacted"
_STREAM_DELIVERY_PENDING_EVENT = "stream_delivery_pending"
_STREAM_DELIVERY_COMPLETED_EVENT = "stream_delivery_completed"
_STREAM_DELIVERY_INTERRUPTED_EVENT = "stream_delivery_interrupted"
_TERMINAL_STATES = frozenset(
    {
        AnswerExecutionState.COMPLETED,
        AnswerExecutionState.STOPPED,
        AnswerExecutionState.FAILED,
        AnswerExecutionState.THROTTLED,
        AnswerExecutionState.REJECTED,
    }
)
_OUTCOMES_WITH_FROZEN_EVIDENCE = frozenset(
    {
        AnswerOutcome.EVIDENCE_GATED_ANSWER,
        AnswerOutcome.GENERATION_UNAVAILABLE,
    }
)


@dataclass(frozen=True)
class QueryConditionResolution:
    query_conditions: QueryConditionSet
    provenance: dict[str, str]


@dataclass(frozen=True)
class AnswerExecutionHandle:
    execution_id: str
    request_id: str
    user_id: str
    session_id: str
    assistant_message_id: str | None
    question: str
    query_conditions: QueryConditionSet
    condition_provenance: dict[str, str]


@dataclass(frozen=True)
class LoadedAnswerExecution:
    projection: dict[str, Any]
    result: dict[str, Any] | None


class AnswerExecutionStore:
    """Persists one private answer execution and reconstructs its closed result."""

    def __init__(self, session: AsyncSession, repository: ChatRepository) -> None:
        self.session = session
        self.repository = repository

    async def resolve_query_conditions(
        self,
        *,
        user_id: str,
        session_id: str,
        question: str,
        explicit_conditions: Sequence[object] | None,
        inherit_conditions: bool,
    ) -> QueryConditionResolution:
        normalized_question = question.strip()
        if not normalized_question:
            raise AppError(status_code=422, code="QUERY_REQUIRED", message="a non-empty question is required")
        if inherit_conditions:
            if explicit_conditions is not None:
                raise AppError(
                    status_code=422,
                    code="QUERY_CONDITION_SOURCE_CONFLICT",
                    message="inherited conditions cannot be combined with explicit conditions",
                )
            previous = await self._latest_completed_execution(session_id=session_id, user_id=user_id)
            if previous is None or previous.result is None:
                raise AppError(
                    status_code=422,
                    code="QUERY_CONDITION_INHERITANCE_UNAVAILABLE",
                    message="conditions can only be inherited from a completed turn in this conversation",
                )
            prior_conditions = previous.result.get("query_condition_set")
            if not isinstance(prior_conditions, Mapping):
                raise AppError(
                    status_code=500,
                    code="ANSWER_EXECUTION_CONTRACT_INVALID",
                    message="the retained answer execution cannot provide conditions",
                )
            prior_records = prior_conditions.get("conditions")
            if not isinstance(prior_records, list):
                raise AppError(
                    status_code=500,
                    code="ANSWER_EXECUTION_CONTRACT_INVALID",
                    message="the retained answer execution cannot provide conditions",
                )
            try:
                # Inheritance copies the retained conditions into this turn. Its
                # identity must remain distinct even if the new question is
                # byte-for-byte equal to the originating turn's question.
                query_conditions = QueryConditionSet.from_records(
                    normalized_question=normalized_question,
                    records=prior_records,
                    identity_nonce=uuid.uuid4().hex,
                )
            except ValueError as exc:
                raise AppError(
                    status_code=500,
                    code="ANSWER_EXECUTION_CONTRACT_INVALID",
                    message="the retained answer execution cannot provide conditions",
                ) from exc
            return QueryConditionResolution(
                query_conditions=query_conditions,
                provenance={"mode": "inherited", "source_execution_id": previous.projection["id"]},
            )

        if explicit_conditions is not None:
            try:
                query_conditions = QueryConditionSet.from_records(
                    normalized_question=normalized_question,
                    records=explicit_conditions,
                )
            except ValueError as exc:
                raise AppError(
                    status_code=422,
                    code="QUERY_CONDITION_INVALID",
                    message="query conditions must be explicit, complete, and non-duplicated",
                ) from exc
            return QueryConditionResolution(
                query_conditions=query_conditions,
                provenance={"mode": "explicit"},
            )

        try:
            query_conditions = QueryConditionSet.from_question(normalized_question)
        except ValueError as exc:
            raise AppError(
                status_code=422,
                code="QUERY_CONDITION_INVALID",
                message="question-derived query conditions exceed the accepted retry boundary",
            ) from exc
        return QueryConditionResolution(
            query_conditions=query_conditions,
            provenance={"mode": "question_normalized"},
        )

    async def admit(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        question: str,
        resolution: QueryConditionResolution,
        queued: bool = False,
    ) -> AnswerExecutionHandle:
        await self.repository.acquire_private_conversation_write_fence()
        identity = StableIdentity.new(StableIdentityKind.ANSWER_EXECUTION)
        handle = AnswerExecutionHandle(
            execution_id=identity.stable_id,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            assistant_message_id=str(uuid.uuid4()),
            question=question,
            query_conditions=resolution.query_conditions,
            condition_provenance=dict(resolution.provenance),
        )
        user_message = await self.repository.add_message(
            session_id=session_id,
            user_id=user_id,
            message_type="user",
            content=question,
            answer_execution_id=handle.execution_id,
        )
        self.session.add(
            AnswerExecutionModel(
                id=handle.execution_id,
                session_id=session_id,
                user_id=user_id,
                initial_state=AnswerExecutionState.ADMITTED.value,
                request={
                    "schema": _REQUEST_SCHEMA,
                    "request_id": request_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "user_message_id": user_message.id,
                    "assistant_message_id": handle.assistant_message_id,
                    "question": question,
                    "query_condition_set": resolution.query_conditions.to_record(),
                    "condition_provenance": dict(resolution.provenance),
                },
            )
        )
        self._append_event(
            execution_id=handle.execution_id,
            event_type="created",
            from_state=None,
            to_state=AnswerExecutionState.ADMITTED,
            sequence=1,
            payload={"request_id": request_id},
        )
        self._append_event(
            execution_id=handle.execution_id,
            event_type="state_changed",
            from_state=AnswerExecutionState.ADMITTED,
            to_state=AnswerExecutionState.QUEUED,
            sequence=2,
            payload={"request_id": request_id},
        )
        if not queued:
            self._append_event(
                execution_id=handle.execution_id,
                event_type="state_changed",
                from_state=AnswerExecutionState.QUEUED,
                to_state=AnswerExecutionState.RUNNING,
                sequence=3,
                payload={"request_id": request_id},
            )
        await self.session.flush()
        return handle

    async def start(self, *, handle: AnswerExecutionHandle) -> None:
        await self.repository.acquire_private_conversation_write_fence()
        execution = await self._locked_execution(execution_id=handle.execution_id)
        if execution.user_id != handle.user_id or execution.session_id != handle.session_id:
            raise ValueError("private answer execution does not belong to this conversation")
        events = await self._events_for_execution(execution_id=execution.id)
        state, terminal, *_ = self._reconstruct_state(execution=execution, events=events)
        if state is not AnswerExecutionState.QUEUED or terminal is not None:
            raise ValueError("only queued executions may start")
        self._append_event(
            execution_id=handle.execution_id, event_type="state_changed",
            from_state=AnswerExecutionState.QUEUED, to_state=AnswerExecutionState.RUNNING,
            sequence=self._next_event_sequence(events), payload={"request_id": handle.request_id},
        )
        await self.session.flush()

    async def recover_interrupted(self) -> int:
        """Called before serving requests in the supported single-process runtime."""
        await self.repository.acquire_private_conversation_write_fence()
        terminal = select(AnswerExecutionEventModel.execution_id).where(
            AnswerExecutionEventModel.to_state.in_([state.value for state in _TERMINAL_STATES]),
        )
        executions = list(await self.session.scalars(
            select(AnswerExecutionModel).where(AnswerExecutionModel.id.not_in(terminal)).with_for_update(),
        ))
        recovered = 0
        for execution in executions:
            loaded = await self.load(execution_id=execution.id, user_id=execution.user_id, session_id=execution.session_id)
            if loaded is None or loaded.projection["state"] not in {"queued", "running"}:
                raise ValueError("interrupted execution has no verifiable recovery state")
            handle = self._handle_from_request(execution, execution.request)
            await self.fail(handle=handle, failure_code="ANSWER_EXECUTION_INTERRUPTED")
            recovered += 1
        return recovered

    async def complete(
        self,
        *,
        handle: AnswerExecutionHandle,
        outcome: AnswerExecutionOutcome,
    ) -> tuple[dict[str, Any], LoadedAnswerExecution]:
        await self.repository.acquire_private_conversation_write_fence()
        if not isinstance(handle.assistant_message_id, str) or not handle.assistant_message_id:
            raise ValueError("admitted answer execution has no reserved assistant message identity")
        result = self._completed_result(
            handle=handle,
            outcome=outcome,
            assistant_message_id=handle.assistant_message_id,
        )
        tombstoned_document_ids = await self._tombstoned_evidence_document_ids(result=result)
        if tombstoned_document_ids or await self._publication_redactions(result):
            return await self.fail(handle=handle, failure_code="ANSWER_EVIDENCE_WITHDRAWN")
        if await answer_is_blocked(self.session, result.get("knowledge_version_identities", [])):
            return await self.fail(handle=handle, failure_code="MAINTENANCE_CONTAINMENT_ACTIVE")
        assistant_message = await self.repository.add_message(
            session_id=handle.session_id,
            user_id=handle.user_id,
            message_type="assistant",
            content=outcome.text,
            answer_execution_id=handle.execution_id,
            rag_trace=outcome.to_persisted_rag_trace(),
            message_id=handle.assistant_message_id,
        )
        execution = await self._locked_execution(execution_id=handle.execution_id)
        if execution.user_id != handle.user_id or execution.session_id != handle.session_id:
            raise ValueError("private answer execution does not belong to this conversation")
        events = await self._events_for_execution(execution_id=execution.id)
        state, terminal, _redactions, _delivery_pending, _delivery_completed, _delivery_interrupted = self._reconstruct_state(
            execution=execution,
            events=events,
        )
        if state is not AnswerExecutionState.RUNNING or terminal is not None:
            raise ValueError("answer execution cannot complete from its persisted state")
        next_sequence = self._next_event_sequence(events)
        self._append_event(
            execution_id=handle.execution_id,
            event_type="state_changed",
            from_state=AnswerExecutionState.RUNNING,
            to_state=AnswerExecutionState.COMPLETED,
            sequence=next_sequence,
            payload=result,
        )
        next_sequence += 1
        completion_redactions: list[dict[str, object]] = []
        for document_id in tombstoned_document_ids:
            item_ids = self._item_identities_for_document(terminal=result, document_id=document_id)
            if not item_ids:
                continue
            redaction = {
                "document_id": document_id,
                "item_identities": item_ids,
            }
            self._append_event(
                execution_id=handle.execution_id,
                event_type=_EVIDENCE_REDACTED_EVENT,
                from_state=None,
                to_state=None,
                sequence=next_sequence,
                payload=redaction,
            )
            completion_redactions.append(redaction)
            next_sequence += 1
        await self.session.flush()
        if tombstoned_document_ids:
            redacted_result = self._apply_evidence_redactions(
                terminal=result,
                redactions=completion_redactions,
            )
            loaded = LoadedAnswerExecution(
                projection=self._completed_projection(handle=handle, result=redacted_result),
                result=redacted_result,
            )
        else:
            loaded = LoadedAnswerExecution(
                projection=self._completed_projection(handle=handle, result=result),
                result=result,
            )
        return self._message_record(assistant_message, loaded), loaded

    async def stop(self, *, handle: AnswerExecutionHandle) -> tuple[dict[str, Any], LoadedAnswerExecution]:
        return await self._finish_non_completed_terminal(
            handle=handle,
            state=AnswerExecutionState.STOPPED,
            failure_code="ANSWER_EXECUTION_STOPPED",
        )

    async def fail(
        self,
        *,
        handle: AnswerExecutionHandle,
        failure_code: str,
    ) -> tuple[dict[str, Any], LoadedAnswerExecution]:
        return await self._finish_non_completed_terminal(
            handle=handle,
            state=AnswerExecutionState.FAILED,
            failure_code=failure_code,
        )

    async def _finish_non_completed_terminal(
        self,
        *,
        handle: AnswerExecutionHandle,
        state: AnswerExecutionState,
        failure_code: str,
    ) -> tuple[dict[str, Any], LoadedAnswerExecution]:
        await self.repository.acquire_private_conversation_write_fence()
        if not isinstance(handle.assistant_message_id, str) or not handle.assistant_message_id:
            raise ValueError("admitted answer execution has no reserved assistant message identity")
        assistant_message = await self.repository.add_message(
            session_id=handle.session_id,
            user_id=handle.user_id,
            message_type="assistant",
            content="",
            answer_execution_id=handle.execution_id,
            message_id=handle.assistant_message_id,
        )
        result = self._terminal_result(
            handle=handle,
            state=state,
            detail={"code": failure_code},
            assistant_message_id=handle.assistant_message_id,
        )
        execution = await self._locked_execution(execution_id=handle.execution_id)
        if execution.user_id != handle.user_id or execution.session_id != handle.session_id:
            raise ValueError("private answer execution does not belong to this conversation")
        events = await self._events_for_execution(execution_id=execution.id)
        persisted_state, terminal, _redactions, _delivery_pending, _delivery_completed, _delivery_interrupted = self._reconstruct_state(
            execution=execution,
            events=events,
        )
        if persisted_state not in {AnswerExecutionState.RUNNING, AnswerExecutionState.QUEUED} or terminal is not None:
            raise ValueError("answer execution cannot finish from its persisted state")
        self._append_event(
            execution_id=handle.execution_id,
            event_type="state_changed",
            from_state=persisted_state,
            to_state=state,
            sequence=self._next_event_sequence(events),
            payload=result,
        )
        await self.session.flush()
        loaded = LoadedAnswerExecution(
            projection=self._terminal_projection(handle=handle, result=result),
            result=result,
        )
        return self._message_record(assistant_message, loaded), loaded

    async def load(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str | None = None,
        message_id: str | None = None,
        message_type: str | None = None,
    ) -> LoadedAnswerExecution | None:
        execution = await self.session.get(AnswerExecutionModel, execution_id)
        if execution is None or execution.user_id != user_id:
            return None
        if session_id is not None and execution.session_id != session_id:
            raise ValueError("private answer execution does not belong to this conversation")
        request = execution.request if isinstance(execution.request, Mapping) else {}
        handle = self._handle_from_request(execution, request)
        events = await self._events_for_execution(execution_id=execution.id)
        state, terminal, redactions, delivery_pending, delivery_completed, delivery_interrupted = self._reconstruct_state(
            execution=execution,
            events=events,
        )
        await self._validate_frozen_message_bindings(
            execution=execution,
            request=request,
            terminal=terminal,
        )
        if message_id is not None or message_type is not None:
            if not isinstance(message_id, str) or not message_id or message_type is None:
                raise ValueError("answer execution message binding is incomplete")
            self._validate_message_binding(
                request=request,
                terminal=terminal,
                message_id=message_id,
                message_type=message_type,
            )
        if delivery_interrupted:
            raise ValueError("closed answer execution stream delivery was interrupted")
        if state is AnswerExecutionState.COMPLETED:
            if terminal is None:
                raise ValueError("completed answer execution has no terminal result")
            if delivery_pending and not delivery_completed:
                raise ValueError("closed answer execution stream delivery is pending")
            terminal = self._apply_evidence_redactions(
                terminal=terminal, redactions=(*redactions, *await self._publication_redactions(terminal)),
            )
            return LoadedAnswerExecution(
                projection=self._completed_projection(handle=handle, result=terminal),
                result=terminal,
            )
        if state in _TERMINAL_STATES:
            if terminal is None:
                raise ValueError("terminal answer execution has no terminal record")
            return LoadedAnswerExecution(
                projection=self._terminal_projection(handle=handle, result=terminal),
                result=terminal,
            )
        return LoadedAnswerExecution(
            projection={
                "id": handle.execution_id,
                "state": state.value,
                "question": handle.question,
                "query_condition_set": handle.query_conditions.to_record(),
                "condition_provenance": dict(handle.condition_provenance),
            },
            result=None,
        )

    async def execution_ids_for_messages(
        self,
        *,
        messages: Sequence[ChatMessage],
    ) -> tuple[dict[str, str], frozenset[str]]:
        """Return immutable execution bindings and malformed message identities."""

        message_ids = {message.id for message in messages}
        if not message_ids:
            return {}, frozenset()
        bindings: dict[str, list[str]] = {message_id: [] for message_id in message_ids}
        executions = list(
            (
                await self.session.scalars(
                    select(AnswerExecutionModel)
                    .where(AnswerExecutionModel.request["user_message_id"].as_string().in_(message_ids))
                    .order_by(AnswerExecutionModel.created_at.asc(), AnswerExecutionModel.id.asc())
                )
            ).all()
        )
        executions_by_id = {execution.id: execution for execution in executions}
        for execution in executions:
            request = execution.request if isinstance(execution.request, Mapping) else {}
            message_id = request.get("user_message_id")
            if isinstance(message_id, str) and message_id in bindings:
                bindings[message_id].append(execution.id)

        terminal_events = list(
            (
                await self.session.scalars(
                    select(AnswerExecutionEventModel)
                    .where(
                        AnswerExecutionEventModel.to_state.in_(
                            tuple(state.value for state in _TERMINAL_STATES)
                        ),
                        AnswerExecutionEventModel.payload["data"]["assistant_message_id"]
                        .as_string()
                        .in_(message_ids),
                    )
                    .order_by(
                        AnswerExecutionEventModel.occurred_at.asc(),
                        AnswerExecutionEventModel.id.asc(),
                    )
                )
            ).all()
        )
        for event in terminal_events:
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            data = payload.get("data")
            message_id = data.get("assistant_message_id") if isinstance(data, Mapping) else None
            if isinstance(message_id, str) and message_id in bindings:
                bindings[message_id].append(event.execution_id)

        resolved: dict[str, str] = {}
        invalid_message_ids: set[str] = set()
        for message in messages:
            candidates = list(dict.fromkeys(bindings.get(message.id, [])))
            if len(candidates) > 1:
                invalid_message_ids.add(message.id)
                continue
            if candidates:
                execution = executions_by_id.get(candidates[0])
                if execution is None:
                    execution = await self.session.get(AnswerExecutionModel, candidates[0])
                if execution is None:
                    invalid_message_ids.add(message.id)
                    continue
                if execution.user_id != message.user_id or execution.session_id != message.session_id:
                    invalid_message_ids.add(message.id)
                    continue
            if message.answer_execution_id is not None:
                if candidates != [message.answer_execution_id]:
                    invalid_message_ids.add(message.id)
                    continue
            if candidates:
                resolved[message.id] = candidates[0]
        return resolved, frozenset(invalid_message_ids)

    async def load_for_message(
        self,
        *,
        user_id: str,
        session_id: str,
        message_id: str,
        message_type: str,
        indexed_execution_id: str | None,
    ) -> LoadedAnswerExecution | None:
        """Find a message execution from immutable bindings, not its mutable index."""

        if message_type not in {"user", "assistant"}:
            raise ValueError("answer execution is linked to an invalid conversation message")
        if indexed_execution_id is not None:
            loaded = await self.load(
                execution_id=indexed_execution_id,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                message_type=message_type,
            )
            if loaded is None:
                raise ValueError("conversation message index has no frozen answer execution binding")
            return loaded
        matching_executions: list[AnswerExecutionModel] = []
        if message_type == "user":
            executions = list(
                (
                    await self.session.scalars(
                        select(AnswerExecutionModel).order_by(
                            AnswerExecutionModel.created_at.asc(),
                            AnswerExecutionModel.id.asc(),
                        )
                    )
                ).all()
            )
            for execution in executions:
                request = execution.request if isinstance(execution.request, Mapping) else {}
                if request.get("user_message_id") != message_id:
                    continue
                matching_executions.append(execution)
        else:
            terminal_events = list(
                (
                    await self.session.scalars(
                        select(AnswerExecutionEventModel)
                        .where(
                            AnswerExecutionEventModel.to_state.in_(
                                tuple(state.value for state in _TERMINAL_STATES)
                            )
                        )
                        .order_by(
                            AnswerExecutionEventModel.occurred_at.asc(),
                            AnswerExecutionEventModel.id.asc(),
                        )
                    )
                ).all()
            )
            matching_execution_ids: list[str] = []
            for event in terminal_events:
                payload = event.payload if isinstance(event.payload, Mapping) else {}
                data = payload.get("data")
                if not isinstance(data, Mapping) or data.get("assistant_message_id") != message_id:
                    continue
                if event.execution_id not in matching_execution_ids:
                    matching_execution_ids.append(event.execution_id)
            for execution_id in matching_execution_ids:
                execution = await self.session.get(AnswerExecutionModel, execution_id)
                if execution is None:
                    raise ValueError("conversation message has an orphaned frozen answer execution binding")
                matching_executions.append(execution)

        if len(matching_executions) > 1:
            raise ValueError("conversation message has more than one frozen answer execution binding")
        if not matching_executions:
            return None
        execution = matching_executions[0]
        if execution.user_id != user_id or execution.session_id != session_id:
            raise ValueError("conversation message is outside its frozen private conversation")
        loaded = await self.load(
            execution_id=execution.id,
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            message_type=message_type,
        )
        if loaded is None:
            raise ValueError("private answer execution is missing for its conversation message")
        return loaded

    async def load_pending_stream_result(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
    ) -> LoadedAnswerExecution:
        """Refresh redactions for the one completed result this stream is delivering."""

        execution = await self.session.get(AnswerExecutionModel, execution_id)
        if execution is None or execution.user_id != user_id or execution.session_id != session_id:
            raise ValueError("private answer execution does not belong to this conversation")
        request = execution.request if isinstance(execution.request, Mapping) else {}
        handle = self._handle_from_request(execution, request)
        events = await self._events_for_execution(execution_id=execution.id)
        state, terminal, redactions, delivery_pending, delivery_completed, delivery_interrupted = self._reconstruct_state(
            execution=execution,
            events=events,
        )
        await self._validate_frozen_message_bindings(
            execution=execution,
            request=request,
            terminal=terminal,
        )
        if (
            state is not AnswerExecutionState.COMPLETED
            or terminal is None
            or not delivery_pending
            or delivery_completed
            or delivery_interrupted
        ):
            raise ValueError("answer execution is not a pending completed stream result")
        redacted_terminal = self._apply_evidence_redactions(
            terminal=terminal, redactions=(*redactions, *await self._publication_redactions(terminal)),
        )
        return LoadedAnswerExecution(
            projection=self._completed_projection(handle=handle, result=redacted_terminal),
            result=redacted_terminal,
        )

    async def record_stream_delivery_pending(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
    ) -> bool:
        """Record that a completed SSE projection needs outer delivery confirmation."""

        await self.repository.acquire_private_conversation_write_fence()
        execution = await self._locked_execution(execution_id=execution_id)
        if execution.user_id != user_id or execution.session_id != session_id:
            raise ValueError("private answer execution does not belong to this conversation")
        events = await self._events_for_execution(execution_id=execution.id)
        state, terminal, _redactions, delivery_pending, delivery_completed, delivery_interrupted = self._reconstruct_state(
            execution=execution,
            events=events,
        )
        if delivery_pending:
            return False
        if (
            state not in {AnswerExecutionState.RUNNING, AnswerExecutionState.QUEUED}
            or terminal is not None
            or delivery_completed
            or delivery_interrupted
        ):
            raise ValueError("stream delivery cannot begin from the persisted answer execution state")
        self._append_event(
            execution_id=execution.id,
            event_type=_STREAM_DELIVERY_PENDING_EVENT,
            from_state=None,
            to_state=None,
            sequence=self._next_event_sequence(events),
            payload={"code": "CHAT_STREAM_DELIVERY_PENDING"},
        )
        await self.session.flush()
        return True

    async def record_stream_delivery_completion(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
    ) -> bool:
        """Record the outer transport's completed delivery of a closed SSE result."""

        await self.repository.acquire_private_conversation_write_fence()
        execution = await self._locked_execution(execution_id=execution_id)
        if execution.user_id != user_id or execution.session_id != session_id:
            raise ValueError("private answer execution does not belong to this conversation")
        events = await self._events_for_execution(execution_id=execution.id)
        state, terminal, _redactions, delivery_pending, delivery_completed, delivery_interrupted = self._reconstruct_state(
            execution=execution,
            events=events,
        )
        if delivery_completed or delivery_interrupted:
            return False
        if (
            state is not AnswerExecutionState.COMPLETED
            or terminal is None
            or not delivery_pending
        ):
            return False
        request = execution.request if isinstance(execution.request, Mapping) else {}
        handle = self._handle_from_request(execution, request)
        self._completed_projection(handle=handle, result=terminal)
        self._append_event(
            execution_id=execution.id,
            event_type=_STREAM_DELIVERY_COMPLETED_EVENT,
            from_state=None,
            to_state=None,
            sequence=self._next_event_sequence(events),
            payload={"code": "CHAT_STREAM_DELIVERED"},
        )
        await self.session.flush()
        return True

    async def record_stream_delivery_interruption(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
    ) -> bool:
        """Record a closed SSE result that was not fully projected to its client."""

        await self.repository.acquire_private_conversation_write_fence()
        execution = await self._locked_execution(execution_id=execution_id)
        if (
            execution is None
            or execution.user_id != user_id
            or execution.session_id != session_id
        ):
            raise ValueError("private answer execution does not belong to this conversation")
        request = execution.request if isinstance(execution.request, Mapping) else {}
        handle = self._handle_from_request(execution, request)
        events = await self._events_for_execution(execution_id=execution.id)
        state, terminal, _redactions, delivery_pending, delivery_completed, delivery_interrupted = self._reconstruct_state(
            execution=execution,
            events=events,
        )
        if delivery_interrupted or delivery_completed:
            return False
        if state is not AnswerExecutionState.COMPLETED or terminal is None or not delivery_pending:
            return False
        self._completed_projection(handle=handle, result=terminal)
        self._append_event(
            execution_id=execution.id,
            event_type=_STREAM_DELIVERY_INTERRUPTED_EVENT,
            from_state=None,
            to_state=None,
            sequence=self._next_event_sequence(events),
            payload={"code": "CHAT_STREAM_INTERRUPTED"},
        )
        await self.session.flush()
        return True

    async def _publication_redactions(self, terminal: Mapping[str, object]) -> list[dict]:
        versions = terminal.get("knowledge_version_identities")
        if not isinstance(versions, list) or not versions:
            return []
        events = await read_publication_withdrawals(self.session, versions)
        redactions = []
        for event in events:
            withdrawal = event.payload
            item_ids = self._item_identities_for_document(
                terminal=terminal, document_id=withdrawal["document_identity"], publication_identity=event.aggregate_id,
            )
            if item_ids:
                redactions.append({
                    "document_id": withdrawal["document_identity"], "item_identities": item_ids, "withdrawal": withdrawal,
                })
        return redactions

    async def redact_document_evidence(
        self, *, document_id: str, publication_identity: str | None = None, withdrawal: dict | None = None,
    ) -> int:
        """Append private redaction events without rewriting frozen terminal results."""

        if not document_id.strip():
            raise ValueError("document id is required for answer evidence redaction")
        await self.repository.acquire_private_conversation_write_fence()
        executions = list(
            (
                await self.session.scalars(
                    select(AnswerExecutionModel).order_by(AnswerExecutionModel.id.asc())
                )
            ).all()
        )
        appended = 0
        for listed_execution in executions:
            execution = await self._locked_execution(execution_id=listed_execution.id)
            events = await self._events_for_execution(execution_id=execution.id)
            state, terminal, redactions, _delivery_pending, _delivery_completed, delivery_interrupted = self._reconstruct_state(
                execution=execution,
                events=events,
            )
            if delivery_interrupted or state is not AnswerExecutionState.COMPLETED or terminal is None:
                continue
            item_ids = self._item_identities_for_document(
                terminal=terminal, document_id=document_id, publication_identity=publication_identity,
            )
            if not item_ids:
                continue
            already_redacted = {
                item_id
                for redaction in redactions
                if redaction.get("document_id") == document_id
                for item_id in self._redaction_item_ids(redaction)
            }
            pending = [item_id for item_id in item_ids if item_id not in already_redacted]
            if not pending:
                continue
            self._append_event(
                execution_id=execution.id,
                event_type=_EVIDENCE_REDACTED_EVENT,
                from_state=None,
                to_state=None,
                sequence=self._next_event_sequence(events),
                payload={
                    "document_id": document_id,
                    "item_identities": pending,
                    **({"withdrawal": withdrawal} if withdrawal is not None else {}),
                },
            )
            appended += 1
        if appended:
            await self.session.flush()
        return appended

    async def _tombstoned_evidence_document_ids(
        self,
        *,
        result: Mapping[str, object],
    ) -> tuple[str, ...]:
        document_ids = self._evidence_document_ids(result=result)
        if not document_ids:
            return ()
        documents = list(
            (
                await self.session.scalars(
                    select(Document)
                    .where(Document.id.in_(document_ids))
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            ).all()
        )
        by_id = {document.id: document for document in documents}
        return tuple(
            document_id
            for document_id in document_ids
            if document_id in by_id and by_id[document_id].deleted_at is not None
        )

    async def _latest_completed_execution(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> LoadedAnswerExecution | None:
        executions = list(
            (
                await self.session.scalars(
                    select(AnswerExecutionModel)
                    .where(
                        AnswerExecutionModel.session_id == session_id,
                        AnswerExecutionModel.user_id == user_id,
                    )
                    .order_by(AnswerExecutionModel.created_at.desc())
                )
            ).all()
        )
        for execution in executions:
            loaded = await self.load(execution_id=execution.id, user_id=user_id)
            if loaded is not None and loaded.projection["state"] == AnswerExecutionState.COMPLETED.value:
                return loaded
        return None

    async def _events_for_execution(self, *, execution_id: str) -> list[AnswerExecutionEventModel]:
        return list(
            (
                await self.session.scalars(
                    select(AnswerExecutionEventModel)
                    .where(AnswerExecutionEventModel.execution_id == execution_id)
                    .order_by(
                        AnswerExecutionEventModel.sequence.asc(),
                        AnswerExecutionEventModel.occurred_at.asc(),
                        AnswerExecutionEventModel.id.asc(),
                    )
                )
            ).all()
        )

    async def _locked_execution(self, *, execution_id: str) -> AnswerExecutionModel:
        execution = await self.session.scalar(
            select(AnswerExecutionModel)
            .where(AnswerExecutionModel.id == execution_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if execution is None:
            raise ValueError("private answer execution is missing")
        return execution

    def _append_event(
        self,
        *,
        execution_id: str,
        event_type: str,
        from_state: AnswerExecutionState | None,
        to_state: AnswerExecutionState | None,
        sequence: int,
        payload: Mapping[str, object],
    ) -> None:
        self.session.add(
            AnswerExecutionEventModel(
                execution_id=execution_id,
                sequence=sequence,
                event_type=event_type,
                from_state=from_state.value if from_state is not None else None,
                to_state=to_state.value if to_state is not None else None,
                payload={
                    "schema": _EVENT_SCHEMA,
                    "sequence": sequence,
                    "data": dict(payload),
                },
            )
        )

    @staticmethod
    def _message_record(message, loaded: LoadedAnswerExecution) -> dict[str, Any]:
        return {
            "id": message.id,
            "type": message.type,
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
            "answer_execution_id": message.answer_execution_id,
            "answer_execution": loaded.projection,
            "answer_execution_result": loaded.result,
            "rag_trace": message.rag_trace,
        }

    @staticmethod
    def _query_conditions_from_record(
        value: Mapping[str, object],
        *,
        normalized_question: str,
    ) -> QueryConditionSet:
        conditions = value.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError("query condition set has no condition list")
        identity_nonce = value.get("identity_nonce")
        if identity_nonce is not None and (not isinstance(identity_nonce, str) or not identity_nonce.strip()):
            raise ValueError("query condition set identity nonce is invalid")
        parsed = QueryConditionSet.from_records(
            normalized_question=normalized_question,
            records=conditions,
            identity_nonce=identity_nonce,
        )
        if value.get("identity") != parsed.identity:
            raise ValueError("query condition set identity changed")
        return parsed

    @classmethod
    def _handle_from_request(
        cls,
        execution: AnswerExecutionModel,
        request: Mapping[str, object],
    ) -> AnswerExecutionHandle:
        request_conditions = request.get("query_condition_set")
        request_provenance = request.get("condition_provenance")
        reserved_assistant_message_id = request.get("assistant_message_id")
        if (
            request.get("schema") != _REQUEST_SCHEMA
            or request.get("session_id") != execution.session_id
            or request.get("user_id") != execution.user_id
            or not isinstance(request.get("request_id"), str)
            or not isinstance(request.get("user_message_id"), str)
            or not isinstance(request.get("question"), str)
            or not isinstance(request_conditions, Mapping)
            or not isinstance(request_provenance, Mapping)
        ):
            raise ValueError("answer execution request is malformed")
        if reserved_assistant_message_id is not None and (
            not isinstance(reserved_assistant_message_id, str)
            or not reserved_assistant_message_id
            or reserved_assistant_message_id != reserved_assistant_message_id.strip()
        ):
            raise ValueError("answer execution reserved assistant message identity is invalid")
        question = str(request["question"]).strip()
        query_conditions = cls._query_conditions_from_record(
            request_conditions,
            normalized_question=question,
        )
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in request_provenance.items()):
            raise ValueError("answer execution condition provenance is invalid")
        provenance = dict(request_provenance)
        if provenance.get("mode") not in {"explicit", "inherited", "question_normalized"}:
            raise ValueError("answer execution condition provenance is invalid")
        if provenance["mode"] == "inherited" and (
            set(provenance) != {"mode", "source_execution_id"} or not provenance.get("source_execution_id")
        ):
            raise ValueError("inherited condition provenance is incomplete")
        if provenance["mode"] != "inherited" and set(provenance) != {"mode"}:
            raise ValueError("answer execution condition provenance is invalid")
        return AnswerExecutionHandle(
            execution_id=execution.id,
            request_id=str(request["request_id"]),
            user_id=execution.user_id,
            session_id=execution.session_id,
            assistant_message_id=reserved_assistant_message_id,
            question=question,
            query_conditions=query_conditions,
            condition_provenance=provenance,
        )

    @staticmethod
    def _validate_message_binding(
        *,
        request: Mapping[str, object],
        terminal: Mapping[str, object] | None,
        message_id: str,
        message_type: str,
    ) -> None:
        if message_type == "user":
            expected_message_id = request.get("user_message_id")
        elif message_type == "assistant" and terminal is not None:
            expected_message_id = terminal.get("assistant_message_id")
        else:
            raise ValueError("answer execution is linked to an invalid conversation message")
        if not isinstance(expected_message_id, str) or expected_message_id != message_id:
            raise ValueError("conversation message contradicts its frozen answer execution binding")

    async def _validate_frozen_message_bindings(
        self,
        *,
        execution: AnswerExecutionModel,
        request: Mapping[str, object],
        terminal: Mapping[str, object] | None,
    ) -> None:
        user_message_id = request.get("user_message_id")
        await self._validate_frozen_message_binding(
            execution=execution,
            message_id=user_message_id,
            expected_type="user",
        )
        if terminal is None:
            return
        assistant_message_id = terminal.get("assistant_message_id")
        if assistant_message_id is None:
            return
        reserved_assistant_message_id = request.get("assistant_message_id")
        if (
            reserved_assistant_message_id is not None
            and assistant_message_id != reserved_assistant_message_id
        ):
            raise ValueError("terminal answer execution contradicts its reserved assistant message identity")
        await self._validate_frozen_message_binding(
            execution=execution,
            message_id=assistant_message_id,
            expected_type="assistant",
        )

    async def _validate_frozen_message_binding(
        self,
        *,
        execution: AnswerExecutionModel,
        message_id: object,
        expected_type: str,
    ) -> None:
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("frozen answer execution message binding is malformed")
        message = await self.session.get(ChatMessage, message_id)
        if (
            message is None
            or message.type != expected_type
            or message.user_id != execution.user_id
            or message.session_id != execution.session_id
            or message.answer_execution_id not in {None, execution.id}
        ):
            raise ValueError("conversation message contradicts its frozen answer execution binding")

    @staticmethod
    def _reconstruct_state(
        *,
        execution: AnswerExecutionModel,
        events: Sequence[AnswerExecutionEventModel],
    ) -> tuple[
        AnswerExecutionState,
        dict[str, Any] | None,
        tuple[dict[str, Any], ...],
        bool,
        bool,
        bool,
    ]:
        state = AnswerExecutionState(execution.initial_state)
        terminal: dict[str, Any] | None = None
        redactions: list[dict[str, Any]] = []
        delivery_pending = False
        delivery_completed = False
        delivery_interrupted = False
        created_seen = False
        expected_sequence = 1
        transitions = sorted(
            events,
            key=lambda event: (
                int(event.payload.get("sequence", 0)) if isinstance(event.payload, Mapping) else 0,
                event.occurred_at,
                event.id,
            ),
        )
        for event in transitions:
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            if payload.get("schema") != _EVENT_SCHEMA:
                raise ValueError("answer execution event schema is invalid")
            sequence = payload.get("sequence")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != event.sequence
                or sequence != expected_sequence
            ):
                raise ValueError("answer execution event sequence is invalid")
            expected_sequence += 1
            event_data = payload.get("data")
            if not isinstance(event_data, Mapping):
                raise ValueError("answer execution event data is invalid")
            if event.event_type == "created":
                if (
                    created_seen
                    or event.from_state is not None
                    or event.to_state != AnswerExecutionState.ADMITTED.value
                    or state is not AnswerExecutionState.ADMITTED
                ):
                    raise ValueError("answer execution admission event is invalid")
                created_seen = True
                continue
            if event.event_type == _EVIDENCE_REDACTED_EVENT:
                if (
                    state is not AnswerExecutionState.COMPLETED
                    or terminal is None
                    or event.from_state is not None
                    or event.to_state is not None
                ):
                    raise ValueError("answer evidence redaction is not attached to a completed execution")
                redactions.append(dict(event_data))
                continue
            if event.event_type == _STREAM_DELIVERY_PENDING_EVENT:
                if (
                    state not in {AnswerExecutionState.RUNNING, AnswerExecutionState.QUEUED}
                    or terminal is not None
                    or delivery_pending
                    or delivery_completed
                    or delivery_interrupted
                    or event.from_state is not None
                    or event.to_state is not None
                    or event_data.get("code") != "CHAT_STREAM_DELIVERY_PENDING"
                ):
                    raise ValueError("answer execution stream delivery pending record is malformed")
                delivery_pending = True
                continue
            if event.event_type == _STREAM_DELIVERY_COMPLETED_EVENT:
                if (
                    state is not AnswerExecutionState.COMPLETED
                    or terminal is None
                    or not delivery_pending
                    or delivery_completed
                    or delivery_interrupted
                    or event.from_state is not None
                    or event.to_state is not None
                    or event_data.get("code") != "CHAT_STREAM_DELIVERED"
                ):
                    raise ValueError("answer execution stream delivery completion is malformed")
                delivery_completed = True
                continue
            if event.event_type == _STREAM_DELIVERY_INTERRUPTED_EVENT:
                if (
                    state is not AnswerExecutionState.COMPLETED
                    or terminal is None
                    or not delivery_pending
                    or delivery_completed
                    or delivery_interrupted
                    or event.from_state is not None
                    or event.to_state is not None
                    or event_data.get("code") != "CHAT_STREAM_INTERRUPTED"
                ):
                    raise ValueError("answer execution stream interruption is malformed")
                delivery_interrupted = True
                continue
            if event.event_type != "state_changed" or event.to_state is None:
                raise ValueError("answer execution event type is invalid")
            if event.from_state != state.value:
                raise ValueError("answer execution event has a contradictory source state")
            state = validate_transition(AnswerExecutionState, state, event.to_state)
            if state in _TERMINAL_STATES:
                if terminal is not None:
                    raise ValueError("answer execution has more than one terminal result")
                if event_data.get("state") != state.value:
                    raise ValueError("answer execution terminal result contradicts its event state")
                terminal = dict(event_data)
        if not created_seen:
            raise ValueError("answer execution has no admission event")
        return (
            state,
            terminal,
            tuple(redactions),
            delivery_pending,
            delivery_completed,
            delivery_interrupted,
        )

    @staticmethod
    def _next_event_sequence(events: Sequence[AnswerExecutionEventModel]) -> int:
        sequences: list[int] = []
        for event in events:
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            sequence = payload.get("sequence")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != event.sequence
                or sequence < 1
            ):
                raise ValueError("answer execution event sequence is invalid")
            sequences.append(sequence)
        return max(sequences, default=0) + 1

    @staticmethod
    def _item_identities_for_document(
        *, terminal: Mapping[str, object], document_id: str, publication_identity: str | None = None,
    ) -> list[str]:
        evidence_set = terminal.get("evidence_set")
        if not isinstance(evidence_set, Mapping):
            return []
        items = evidence_set.get("items")
        if not isinstance(items, list):
            raise ValueError("completed evidence-bound execution has malformed frozen evidence")
        item_ids: list[str] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("frozen evidence item is malformed")
            evidence = item.get("evidence")
            item_identity = item.get("item_identity")
            if not isinstance(evidence, Mapping) or not isinstance(item_identity, str):
                raise ValueError("frozen evidence item identity is malformed")
            binding = item.get("identity_binding")
            if evidence.get("document_id") == document_id and (
                publication_identity is None
                or isinstance(binding, Mapping) and binding.get("publication_identity") == publication_identity
            ):
                item_ids.append(item_identity)
        return item_ids

    @staticmethod
    def _evidence_document_ids(*, result: Mapping[str, object]) -> tuple[str, ...]:
        evidence_set = result.get("evidence_set")
        if evidence_set is None:
            return ()
        if not isinstance(evidence_set, Mapping):
            raise ValueError("completed evidence-bound execution has malformed frozen evidence")
        items = evidence_set.get("items")
        if not isinstance(items, list):
            raise ValueError("completed evidence-bound execution has malformed frozen evidence")
        document_ids: list[str] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("frozen evidence item is malformed")
            evidence = item.get("evidence")
            document_id = evidence.get("document_id") if isinstance(evidence, Mapping) else None
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("frozen evidence item has no document identity")
            if document_id not in document_ids:
                document_ids.append(document_id)
        return tuple(document_ids)

    @staticmethod
    def _redaction_item_ids(redaction: Mapping[str, object]) -> list[str]:
        document_id = redaction.get("document_id")
        item_ids = redaction.get("item_identities")
        if (
            not isinstance(document_id, str)
            or not document_id.strip()
            or not isinstance(item_ids, list)
            or not item_ids
            or any(not isinstance(item_id, str) or not item_id for item_id in item_ids)
        ):
            raise ValueError("answer evidence redaction is malformed")
        return list(item_ids)

    @classmethod
    def _apply_evidence_redactions(
        cls,
        *,
        terminal: Mapping[str, object],
        redactions: Sequence[Mapping[str, object]],
    ) -> dict[str, Any]:
        result = deepcopy(dict(terminal))
        if not redactions:
            return result
        evidence_set = result.get("evidence_set")
        if not isinstance(evidence_set, dict):
            raise ValueError("answer evidence redaction needs a frozen Answer Evidence Set")
        items = evidence_set.get("items")
        if not isinstance(items, list):
            raise ValueError("frozen Answer Evidence Set has no item list")
        by_identity = {
            item.get("item_identity"): item
            for item in items
            if isinstance(item, dict) and isinstance(item.get("item_identity"), str)
        }
        if len(by_identity) != len(items):
            raise ValueError("frozen Answer Evidence Set has malformed item identities")
        for redaction in redactions:
            document_id = redaction.get("document_id")
            if not isinstance(document_id, str):
                raise ValueError("answer evidence redaction is malformed")
            for item_identity in cls._redaction_item_ids(redaction):
                item = by_identity.get(item_identity)
                if item is None:
                    raise ValueError("answer evidence redaction targets an unknown item")
                evidence = item.get("evidence")
                if not isinstance(evidence, dict) or evidence.get("document_id") != document_id:
                    raise ValueError("answer evidence redaction targets a different document")
                evidence.pop("content_preview", None)
                evidence.pop("content", None)
                evidence["withdrawn"] = True
                item["withdrawn"] = True
                withdrawal = redaction.get("withdrawal")
                if isinstance(withdrawal, Mapping):
                    binding = item.get("identity_binding")
                    if (
                        not isinstance(binding, Mapping)
                        or binding.get("publication_identity") != withdrawal.get("publication_identity")
                        or evidence.get("generation") != withdrawal.get("generation")
                        or document_id != withdrawal.get("document_identity")
                    ):
                        raise ValueError("withdrawal targets a different publication")
                    item["withdrawal"] = dict(withdrawal)
        return result

    @staticmethod
    def _frozen_evidence_identity_lists(
        evidence_set: Mapping[str, object],
    ) -> tuple[list[str], list[str], list[str]]:
        items = evidence_set.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("frozen evidence set has no items")
        item_identities: list[str] = []
        snapshot_ids: list[str] = []
        knowledge_version_identities: list[str] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("frozen answer evidence item is malformed")
            item_identity = item.get("item_identity")
            snapshot_id = item.get("snapshot_id")
            binding = item.get("identity_binding")
            publication_identity = binding.get("publication_identity") if isinstance(binding, Mapping) else None
            if (
                not isinstance(item_identity, str)
                or not item_identity
                or not isinstance(snapshot_id, str)
                or not snapshot_id
                or not isinstance(publication_identity, str)
                or not publication_identity
            ):
                raise ValueError("frozen answer evidence identity is malformed")
            item_identities.append(item_identity)
            snapshot_ids.append(snapshot_id)
            knowledge_version_identities.append(publication_identity)
        return item_identities, snapshot_ids, knowledge_version_identities

    @staticmethod
    def _completed_result(
        *,
        handle: AnswerExecutionHandle,
        outcome: AnswerExecutionOutcome,
        assistant_message_id: str,
    ) -> dict[str, Any]:
        if outcome.question != handle.question or outcome.query_conditions != handle.query_conditions:
            raise ValueError("answer outcome does not match the admitted question and conditions")
        if not assistant_message_id:
            raise ValueError("completed answer execution requires an assistant message binding")
        if (
            handle.assistant_message_id is not None
            and assistant_message_id != handle.assistant_message_id
        ):
            raise ValueError("completed answer execution contradicts its reserved assistant message identity")
        kind = AnswerOutcome(outcome.kind.value)
        evidence_set = outcome.evidence_set.to_record() if outcome.evidence_set is not None else None
        if kind in _OUTCOMES_WITH_FROZEN_EVIDENCE and evidence_set is None:
            raise ValueError("supported or unavailable generation outcome must retain frozen evidence")
        if kind not in _OUTCOMES_WITH_FROZEN_EVIDENCE and evidence_set is not None:
            raise ValueError("non-evidence outcome cannot retain an Answer Evidence Set")

        item_identities: list[str] = []
        snapshot_ids: list[str] = []
        knowledge_version_identities: list[str] = []
        if evidence_set is not None:
            (
                item_identities,
                snapshot_ids,
                knowledge_version_identities,
            ) = AnswerExecutionStore._frozen_evidence_identity_lists(evidence_set)

        distinct_versions = list(dict.fromkeys(knowledge_version_identities))
        provider_input = None
        if kind in _OUTCOMES_WITH_FROZEN_EVIDENCE:
            evidence_set_identity = evidence_set["identity"] if evidence_set is not None else None
            if not isinstance(evidence_set_identity, str):
                raise ValueError("frozen evidence set has no identity")
            provider_input = AnswerExecutionStore._provider_input_record(
                handle=handle,
                evidence_set_identity=evidence_set_identity,
                item_identities=item_identities,
                snapshot_ids=snapshot_ids,
                knowledge_version_identities=distinct_versions,
            )
        insufficient_reply = None
        if kind is AnswerOutcome.INSUFFICIENT_EVIDENCE_REPLY:
            decision = outcome.sufficiency_decision
            if decision is None or decision.insufficient_reply is None:
                raise ValueError("insufficient outcome must retain its structured insufficiency reply")
            insufficient_reply = decision.insufficient_reply.to_record()

        return {
            "schema": _RESULT_SCHEMA,
            "state": AnswerExecutionState.COMPLETED.value,
            "assistant_message_id": assistant_message_id,
            "question": handle.question,
            "query_condition_set": handle.query_conditions.to_record(),
            "condition_provenance": dict(handle.condition_provenance),
            "outcome": kind.value,
            "text": outcome.text,
            "evidence_set": evidence_set,
            "evidence_set_identity": evidence_set["identity"] if evidence_set is not None else None,
            "item_identities": item_identities,
            "snapshot_ids": snapshot_ids,
            "knowledge_version_identities": distinct_versions,
            "insufficient_evidence_reply": insufficient_reply,
            "provider_input": provider_input,
        }

    @staticmethod
    def _terminal_result(
        *,
        handle: AnswerExecutionHandle,
        state: AnswerExecutionState,
        detail: Mapping[str, object],
        assistant_message_id: str | None,
    ) -> dict[str, Any]:
        if state not in _TERMINAL_STATES or state is AnswerExecutionState.COMPLETED:
            raise ValueError("terminal result must be a non-completed execution state")
        if assistant_message_id is not None and (
            not isinstance(assistant_message_id, str) or not assistant_message_id
        ):
            raise ValueError("terminal answer execution requires an assistant message binding")
        if (
            handle.assistant_message_id is not None
            and assistant_message_id is not None
            and assistant_message_id != handle.assistant_message_id
        ):
            raise ValueError("terminal answer execution contradicts its reserved assistant message identity")
        return {
            "schema": _RESULT_SCHEMA,
            "state": state.value,
            "assistant_message_id": assistant_message_id,
            "question": handle.question,
            "query_condition_set": handle.query_conditions.to_record(),
            "condition_provenance": dict(handle.condition_provenance),
            "failure": dict(detail),
        }

    @classmethod
    def _completed_projection(
        cls,
        *,
        handle: AnswerExecutionHandle,
        result: Mapping[str, object],
    ) -> dict[str, Any]:
        if (
            result.get("schema") != _RESULT_SCHEMA
            or result.get("state") != AnswerExecutionState.COMPLETED.value
            or result.get("question") != handle.question
            or not isinstance(result.get("assistant_message_id"), str)
            or not result.get("assistant_message_id")
            or not isinstance(result.get("outcome"), str)
        ):
            raise ValueError("completed answer execution result is malformed")
        if (
            handle.assistant_message_id is not None
            and result.get("assistant_message_id") != handle.assistant_message_id
        ):
            raise ValueError("completed answer execution result contradicts its reserved assistant message identity")
        outcome = AnswerOutcome(str(result["outcome"]))
        if not isinstance(result.get("text"), str):
            raise ValueError("completed answer execution text is malformed")
        query_conditions = cls._validated_result_context(handle=handle, result=result)
        item_identities = cls._identity_list(
            result.get("item_identities"),
            label="item",
        )
        snapshot_ids = cls._identity_list(
            result.get("snapshot_ids"),
            label="snapshot",
        )
        knowledge_version_identities = cls._identity_list(
            result.get("knowledge_version_identities"),
            label="knowledge version",
        )
        insufficient_reply = cls._validate_completed_outcome(
            handle=handle,
            result=result,
            outcome=outcome,
            query_conditions=query_conditions,
            item_identities=item_identities,
            snapshot_ids=snapshot_ids,
            knowledge_version_identities=knowledge_version_identities,
        )
        projection: dict[str, Any] = {
            "id": handle.execution_id,
            "assistant_message_id": result["assistant_message_id"],
            "state": AnswerExecutionState.COMPLETED.value,
            "question": handle.question,
            "query_condition_set": query_conditions,
            "condition_provenance": dict(handle.condition_provenance),
            "outcome": outcome.value,
            "answer_text": result["text"],
            "evidence_set_identity": result.get("evidence_set_identity"),
            "item_identities": item_identities,
            "snapshot_ids": snapshot_ids,
            "knowledge_version_identities": knowledge_version_identities,
            "evidence_summary": evidence_summary_from_execution(result),
        }
        if insufficient_reply is not None:
            projection["insufficient_evidence_reply"] = insufficient_reply
        return projection

    @staticmethod
    def _provider_input_record(
        *,
        handle: AnswerExecutionHandle,
        evidence_set_identity: str,
        item_identities: list[str],
        snapshot_ids: list[str],
        knowledge_version_identities: list[str],
    ) -> dict[str, object]:
        return {
            "question": handle.question,
            "query_condition_set_identity": handle.query_conditions.identity,
            "evidence_set_identity": evidence_set_identity,
            "item_identities": item_identities,
            "snapshot_ids": snapshot_ids,
            "knowledge_version_identities": knowledge_version_identities,
        }

    @classmethod
    def _validated_result_context(
        cls,
        *,
        handle: AnswerExecutionHandle,
        result: Mapping[str, object],
    ) -> dict[str, object]:
        query_conditions = result.get("query_condition_set")
        condition_provenance = result.get("condition_provenance")
        if not isinstance(query_conditions, Mapping) or not isinstance(condition_provenance, Mapping):
            raise ValueError("answer execution result has no frozen query conditions")
        persisted_conditions = cls._query_conditions_from_record(
            query_conditions,
            normalized_question=handle.question,
        )
        if (
            persisted_conditions != handle.query_conditions
            or dict(query_conditions) != handle.query_conditions.to_record()
            or dict(condition_provenance) != handle.condition_provenance
        ):
            raise ValueError("answer execution result changed its admitted query conditions")
        return handle.query_conditions.to_record()

    @classmethod
    def _validate_completed_outcome(
        cls,
        *,
        handle: AnswerExecutionHandle,
        result: Mapping[str, object],
        outcome: AnswerOutcome,
        query_conditions: Mapping[str, object],
        item_identities: list[str],
        snapshot_ids: list[str],
        knowledge_version_identities: list[str],
    ) -> dict[str, str] | None:
        if outcome in _OUTCOMES_WITH_FROZEN_EVIDENCE:
            evidence_set = result.get("evidence_set")
            evidence_set_identity = result.get("evidence_set_identity")
            if (
                not isinstance(evidence_set, Mapping)
                or not isinstance(evidence_set_identity, str)
                or evidence_set.get("identity") != evidence_set_identity
                or evidence_set.get("query_condition_set_identity") != handle.query_conditions.identity
                or evidence_set.get("query_conditions") != query_conditions
                or result.get("insufficient_evidence_reply") is not None
            ):
                raise ValueError("evidence-bound answer execution has malformed frozen evidence")
            # This validates the frozen item bindings, snapshots, citations, and
            # evidence-set identity without retrieving or reselecting evidence.
            evidence_summary_from_execution(result)
            (
                frozen_item_identities,
                frozen_snapshot_ids,
                frozen_knowledge_versions,
            ) = cls._frozen_evidence_identity_lists(evidence_set)
            if (
                item_identities != frozen_item_identities
                or snapshot_ids != frozen_snapshot_ids
                or knowledge_version_identities != list(dict.fromkeys(frozen_knowledge_versions))
            ):
                raise ValueError("completed answer execution identities contradict frozen evidence")
            expected_provider_input = cls._provider_input_record(
                handle=handle,
                evidence_set_identity=evidence_set_identity,
                item_identities=item_identities,
                snapshot_ids=snapshot_ids,
                knowledge_version_identities=knowledge_version_identities,
            )
            provider_input = result.get("provider_input")
            if not isinstance(provider_input, Mapping) or dict(provider_input) != expected_provider_input:
                raise ValueError("completed answer execution provider input changed")
            return None

        cls._validate_no_evidence_fields(
            result=result,
            item_identities=item_identities,
            snapshot_ids=snapshot_ids,
            knowledge_version_identities=knowledge_version_identities,
        )
        if outcome is AnswerOutcome.INSUFFICIENT_EVIDENCE_REPLY:
            reply = result.get("insufficient_evidence_reply")
            if not isinstance(reply, Mapping):
                raise ValueError("insufficient answer execution has no structured insufficiency reply")
            reason = reply.get("reason")
            if not isinstance(reason, str):
                raise ValueError("insufficient answer execution reason is malformed")
            expected_reply = InsufficientEvidenceReply(
                reason=reason,
                query_condition_set_identity=handle.query_conditions.identity,
            ).to_record()
            if dict(reply) != expected_reply:
                raise ValueError("insufficient answer execution reply changed")
            return expected_reply

        if outcome is AnswerOutcome.NON_KNOWLEDGE_BASE_REPLY:
            if result.get("insufficient_evidence_reply") is not None:
                raise ValueError("non-knowledge-base answer execution has an insufficiency reply")
            return None

        raise ValueError("completed answer execution has an unsupported outcome")

    @staticmethod
    def _validate_no_evidence_fields(
        *,
        result: Mapping[str, object],
        item_identities: list[str],
        snapshot_ids: list[str],
        knowledge_version_identities: list[str],
    ) -> None:
        if (
            result.get("evidence_set") is not None
            or result.get("evidence_set_identity") is not None
            or item_identities
            or snapshot_ids
            or knowledge_version_identities
            or result.get("provider_input") is not None
        ):
            raise ValueError("non-evidence answer execution retained frozen evidence fields")

    @staticmethod
    def _identity_list(value: object, *, label: str) -> list[str]:
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ValueError(f"completed answer execution has malformed {label} identities")
        return list(value)

    @classmethod
    def _terminal_projection(
        cls,
        *,
        handle: AnswerExecutionHandle,
        result: Mapping[str, object],
    ) -> dict[str, Any]:
        state = AnswerExecutionState(str(result.get("state") or ""))
        if state is AnswerExecutionState.COMPLETED or state not in _TERMINAL_STATES:
            raise ValueError("answer execution terminal state is malformed")
        if (
            result.get("schema") != _RESULT_SCHEMA
            or result.get("question") != handle.question
            or result.get("outcome") is not None
        ):
            raise ValueError("non-completed execution has a malformed terminal result")
        assistant_message_id = result.get("assistant_message_id")
        if assistant_message_id is not None and (
            not isinstance(assistant_message_id, str) or not assistant_message_id
        ):
            raise ValueError("non-completed execution has a malformed assistant message binding")
        if (
            handle.assistant_message_id is not None
            and assistant_message_id is not None
            and assistant_message_id != handle.assistant_message_id
        ):
            raise ValueError("non-completed execution result contradicts its reserved assistant message identity")
        query_conditions = cls._validated_result_context(handle=handle, result=result)
        if any(
            result.get(field) is not None
            for field in (
                "evidence_set",
                "evidence_set_identity",
                "insufficient_evidence_reply",
                "provider_input",
            )
        ) or any(
            result.get(field) not in (None, [])
            for field in (
                "item_identities",
                "snapshot_ids",
                "knowledge_version_identities",
            )
        ):
            raise ValueError("non-completed execution retained a completed answer outcome")
        projection: dict[str, Any] = {
            "id": handle.execution_id,
            "state": state.value,
            "question": handle.question,
            "query_condition_set": query_conditions,
            "condition_provenance": dict(handle.condition_provenance),
        }
        if assistant_message_id is not None:
            projection["assistant_message_id"] = assistant_message_id
        failure = result.get("failure")
        if not isinstance(failure, Mapping) or not isinstance(failure.get("code"), str) or not failure["code"]:
            raise ValueError("non-completed execution has no failure code")
        if assistant_message_id is None and (
            state is not AnswerExecutionState.FAILED
            or failure["code"] != "ANSWER_EXECUTION_PERSISTENCE_FAILED"
        ):
            raise ValueError("unbound terminal answer execution is not a persistence failure")
        projection["failure_code"] = failure["code"]
        return projection

    async def fail_persistence_without_assistant_message(
        self,
        *,
        handle: AnswerExecutionHandle,
    ) -> LoadedAnswerExecution:
        """Retain a terminal failure when assistant-message persistence is unavailable."""

        await self.repository.acquire_private_conversation_write_fence()
        result = self._terminal_result(
            handle=handle,
            state=AnswerExecutionState.FAILED,
            detail={"code": "ANSWER_EXECUTION_PERSISTENCE_FAILED"},
            assistant_message_id=None,
        )
        execution = await self._locked_execution(execution_id=handle.execution_id)
        if execution.user_id != handle.user_id or execution.session_id != handle.session_id:
            raise ValueError("private answer execution does not belong to this conversation")
        events = await self._events_for_execution(execution_id=execution.id)
        persisted_state, terminal, _redactions, _delivery_pending, _delivery_completed, _delivery_interrupted = self._reconstruct_state(
            execution=execution,
            events=events,
        )
        if terminal is not None:
            if (
                persisted_state is AnswerExecutionState.FAILED
                and terminal.get("failure") == {"code": "ANSWER_EXECUTION_PERSISTENCE_FAILED"}
                and terminal.get("assistant_message_id") is None
            ):
                return LoadedAnswerExecution(
                    projection=self._terminal_projection(handle=handle, result=terminal),
                    result=terminal,
                )
            raise ValueError("answer execution already has a contradictory terminal result")
        if persisted_state not in {AnswerExecutionState.RUNNING, AnswerExecutionState.QUEUED}:
            raise ValueError("answer execution cannot recover persistence from its persisted state")
        self._append_event(
            execution_id=handle.execution_id,
            event_type="state_changed",
            from_state=persisted_state,
            to_state=AnswerExecutionState.FAILED,
            sequence=self._next_event_sequence(events),
            payload=result,
        )
        await self.session.flush()
        return LoadedAnswerExecution(
            projection=self._terminal_projection(handle=handle, result=result),
            result=result,
        )
