from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.knowledge_feedback.schemas import KnowledgeFeedbackCreate, ReviewWorkItemUpdate
from app.model.chat import ChatMessage
from app.model.document import Document, DocumentChunk
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, MaintenanceSignalLink, ReviewWorkItem
from app.rag.answer_evidence import evidence_summary_from_execution
from app.repository.chat_repository import ChatRepository
from app.retention.policy import lock_registry, read_policy
from app.service.answer_execution_store import AnswerExecutionStore

FEEDBACK_RETENTION_DAYS = 180
REVIEW_AGE_DAYS = 90
_SAFE_CODE = re.compile(r"^[A-Za-z0-9._:+-]{1,96}$")
_FEEDBACK_SCOPE_UNIQUE_CONSTRAINT = "uq_feedback_user_answer_scope"


def _is_feedback_scope_unique_violation(error: IntegrityError) -> bool:
    original = error.orig
    if getattr(original, "constraint_name", None) == _FEEDBACK_SCOPE_UNIQUE_CONSTRAINT:
        return True
    diagnostics = getattr(original, "diag", None)
    if getattr(diagnostics, "constraint_name", None) == _FEEDBACK_SCOPE_UNIQUE_CONSTRAINT:
        return True

    detail = str(original).lower()
    if _FEEDBACK_SCOPE_UNIQUE_CONSTRAINT in detail:
        return True
    return (
        "unique constraint failed" in detail
        and "knowledge_feedback_signals.user_id" in detail
        and "knowledge_feedback_signals.answer_id" in detail
        and "knowledge_feedback_signals.scope_key" in detail
    )


def _source_entry_id(source: object) -> str | None:
    if not isinstance(source, dict):
        return None
    entry_id = source.get("entry_id")
    if not isinstance(entry_id, str):
        metadata = source.get("metadata")
        entry_id = metadata.get("entry_id") if isinstance(metadata, dict) else None
    return entry_id.strip() if isinstance(entry_id, str) and entry_id.strip() else None


def _source_publication_version(source: object) -> str | None:
    if not isinstance(source, dict):
        return None
    value = source.get("publication_version")
    if not isinstance(value, str):
        metadata = source.get("metadata")
        value = metadata.get("publication_version") if isinstance(metadata, dict) else None
    if not isinstance(value, str):
        return None
    version = value.strip()
    return version if _SAFE_CODE.fullmatch(version) else None


class KnowledgeFeedbackService:
    """Own feedback lifecycle and expose a conversation-free editorial queue."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def submit(self, *, user_id: str, payload: KnowledgeFeedbackCreate) -> dict[str, Any]:
        await self._purge_expired()
        result = await self._session.execute(
            select(ChatMessage).where(
                ChatMessage.id == payload.answer_id,
                ChatMessage.user_id == user_id,
                ChatMessage.type == "assistant",
            )
        )
        answer = result.scalar_one_or_none()
        if answer is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="answer not found")

        loaded = await AnswerExecutionStore(
            self._session,
            ChatRepository(self._session),
        ).load_for_message(
            user_id=user_id,
            session_id=answer.session_id,
            message_id=answer.id,
            message_type=answer.type,
            indexed_execution_id=answer.answer_execution_id,
        )
        if payload.entry_id is None:
            return await self._submit_gap_feedback(
                user_id=user_id,
                payload=payload,
                answer=answer,
                loaded=loaded,
            )

        projection, closed_result = self._closed_supported_execution(loaded)
        try:
            summary = evidence_summary_from_execution(closed_result)
        except ValueError as exc:
            raise AppError(
                status_code=422,
                code="KNOWLEDGE_FEEDBACK_CLOSED_ANSWER_REQUIRED",
                message="feedback requires a closed evidence-gated answer",
            ) from exc
        sources = summary.get("sources") if isinstance(summary, dict) else []
        sources = sources if isinstance(sources, list) else []
        entry_id = payload.entry_id
        entry_sources = [source for source in sources if _source_entry_id(source) == entry_id]
        if not entry_sources:
            raise AppError(
                status_code=422,
                code="KNOWLEDGE_FEEDBACK_ENTRY_MISMATCH",
                message="entry is not part of the answer evidence",
            )

        edition = self._knowledge_edition(closed_result, entry_sources)
        scope_key = f"entry:{entry_id}"
        existing = await self._existing_submission(
            user_id=user_id,
            answer_id=payload.answer_id,
            scope_key=scope_key,
            label=payload.label,
            note=payload.note,
            conflict_message="feedback already exists for this answer and entry",
        )
        if existing is not None:
            return existing
        metadata = {
            **self._execution_binding(projection),
            "outcome": projection["outcome"],
            "evidence_coverage": summary.get("coverage", "unavailable"),
            "source_count": len(entry_sources),
        }
        return await self._create_submission(
            answer_id=payload.answer_id,
            user_id=user_id,
            entry_id=entry_id,
            scope_key=scope_key,
            knowledge_edition=edition,
            label=payload.label,
            note=payload.note,
            signal_metadata=metadata,
            subject_id=entry_id,
            conflict_message="feedback already exists for this answer and entry",
        )

    @staticmethod
    def _closed_supported_execution(loaded: object) -> tuple[dict[str, Any], dict[str, Any]]:
        projection = getattr(loaded, "projection", None)
        result = getattr(loaded, "result", None)
        if (
            not isinstance(projection, dict)
            or not isinstance(result, dict)
            or projection.get("state") != "completed"
            or projection.get("outcome") != "evidence_gated_answer"
            or result.get("state") != "completed"
            or result.get("outcome") != "evidence_gated_answer"
        ):
            raise AppError(
                status_code=422,
                code="KNOWLEDGE_FEEDBACK_CLOSED_ANSWER_REQUIRED",
                message="feedback requires a closed evidence-gated answer",
            )
        return projection, result

    async def list_for_user(self, *, user_id: str, answer_id: str | None = None) -> dict[str, Any]:
        normalized_answer_id = answer_id.strip() if isinstance(answer_id, str) else ""
        if answer_id is not None and not normalized_answer_id:
            raise AppError(
                status_code=422,
                code="KNOWLEDGE_FEEDBACK_ANSWER_INVALID",
                message="answer identity is required",
            )
        await self._purge_expired()
        statement = select(KnowledgeFeedbackSignal).where(KnowledgeFeedbackSignal.user_id == user_id)
        if normalized_answer_id:
            statement = statement.where(KnowledgeFeedbackSignal.answer_id == normalized_answer_id)
        result = await self._session.execute(
            statement.order_by(KnowledgeFeedbackSignal.created_at.desc(), KnowledgeFeedbackSignal.id.desc())
        )
        items = [self._project_signal(signal, duplicate=False) for signal in result.scalars().all()]
        await self._session.commit()
        return {"items": items}

    async def _submit_gap_feedback(
        self,
        *,
        user_id: str,
        payload: KnowledgeFeedbackCreate,
        answer: ChatMessage,
        loaded: object,
    ) -> dict[str, Any]:
        projection = getattr(loaded, "projection", None)
        result = getattr(loaded, "result", None)
        if (
            not isinstance(projection, dict)
            or not isinstance(result, dict)
            or projection.get("state") != "completed"
            or projection.get("outcome") != "insufficient_evidence_reply"
        ):
            raise AppError(
                status_code=422,
                code="KNOWLEDGE_GAP_FEEDBACK_MISMATCH",
                message="gap feedback requires a closed insufficient-evidence answer",
            )
        reply = projection.get("insufficient_evidence_reply")
        query_conditions = projection.get("query_condition_set")
        if (
            not isinstance(reply, dict)
            or reply.get("outcome") != "insufficient_evidence_reply"
            or not isinstance(reply.get("reason"), str)
            or not reply["reason"]
            or not isinstance(reply.get("query_condition_set_identity"), str)
            or not reply["query_condition_set_identity"]
            or not isinstance(query_conditions, dict)
            or query_conditions.get("identity") != reply["query_condition_set_identity"]
        ):
            raise AppError(
                status_code=422,
                code="KNOWLEDGE_GAP_FEEDBACK_MISMATCH",
                message="gap feedback requires a structured insufficient-evidence record",
            )

        gap_context = {
            "outcome": "insufficient_evidence_reply",
            "reason": reply["reason"],
            "query_condition_set_identity": reply["query_condition_set_identity"],
        }
        scope_key = f"gap:{gap_context['reason']}"
        existing = await self._existing_submission(
            user_id=user_id,
            answer_id=payload.answer_id,
            scope_key=scope_key,
            label=payload.label,
            note=payload.note,
            conflict_message="feedback already exists for this answer and gap context",
        )
        if existing is not None:
            return existing
        metadata = {
            **self._execution_binding(projection),
            "answer_id": answer.id,
            "outcome": gap_context["outcome"],
            "gap_context": gap_context,
            "label": payload.label,
            "note": payload.note,
        }
        return await self._create_submission(
            answer_id=answer.id,
            user_id=user_id,
            entry_id=None,
            scope_key=scope_key,
            knowledge_edition=None,
            label=payload.label,
            note=payload.note,
            signal_metadata=metadata,
            subject_id=scope_key,
            conflict_message="feedback already exists for this answer and gap context",
        )

    @staticmethod
    def _execution_binding(projection: dict[str, Any]) -> dict[str, Any]:
        conditions = projection.get("query_condition_set")
        if not isinstance(projection.get("id"), str) or not isinstance(conditions, dict):
            return {}
        return {
            "answer_execution_id": projection["id"],
            "query_condition_set_identity": conditions.get("identity"),
            "query_conditions_sha256": canonical_json_sha256(conditions["conditions"]),
            "knowledge_version_identities": projection.get("knowledge_version_identities", []),
        }

    async def _existing_submission(
        self,
        *,
        user_id: str,
        answer_id: str,
        scope_key: str,
        label: str,
        note: str | None,
        conflict_message: str,
    ) -> dict[str, Any] | None:
        result = await self._session.execute(
            select(KnowledgeFeedbackSignal).where(
                KnowledgeFeedbackSignal.user_id == user_id,
                KnowledgeFeedbackSignal.answer_id == answer_id,
                KnowledgeFeedbackSignal.scope_key == scope_key,
            )
        )
        existing = result.scalar_one_or_none()
        if existing is None:
            return None
        if existing.label == label and existing.note == note:
            await self._session.commit()
            return self._project_signal(existing, duplicate=True)
        raise AppError(
            status_code=409,
            code="KNOWLEDGE_FEEDBACK_ALREADY_SUBMITTED",
            message=conflict_message,
        )

    async def _create_submission(
        self,
        *,
        answer_id: str,
        user_id: str,
        entry_id: str | None,
        scope_key: str,
        knowledge_edition: str | None,
        label: str,
        note: str | None,
        signal_metadata: dict[str, Any],
        subject_id: str,
        conflict_message: str,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        retention_days = (await read_policy(self._session))["days"]["feedback_signals"]
        signal = KnowledgeFeedbackSignal(
            answer_id=answer_id,
            user_id=user_id,
            entry_id=entry_id,
            scope_key=scope_key,
            knowledge_edition=knowledge_edition,
            label=label,
            note=note,
            normalized_metadata=signal_metadata,
            created_at=now,
            expires_at=now + timedelta(days=retention_days),
        )
        try:
            self._session.add(signal)
            await self._session.flush()
            if label != "helpful":
                self._session.add(
                    ReviewWorkItem(
                        kind="feedback_signal",
                        dedupe_key=f"feedback:{signal.id}",
                        subject_id=subject_id,
                        signal_id=signal.id,
                        normalized_metadata={},
                        created_at=now,
                        updated_at=now,
                    )
                )
            await self._session.commit()
        except IntegrityError as error:
            await self._session.rollback()
            if not _is_feedback_scope_unique_violation(error):
                raise
            existing = await self._existing_submission(
                user_id=user_id,
                answer_id=answer_id,
                scope_key=scope_key,
                label=label,
                note=note,
                conflict_message=conflict_message,
            )
            if existing is not None:
                return existing
            raise
        return self._project_signal(signal, duplicate=False)

    async def delete(self, *, user_id: str, signal_id: str) -> dict[str, Any]:
        await lock_registry(self._session)
        result = await self._session.execute(
            select(KnowledgeFeedbackSignal).where(
                KnowledgeFeedbackSignal.id == signal_id,
                KnowledgeFeedbackSignal.user_id == user_id,
            ).with_for_update()
        )
        signal = result.scalar_one_or_none()
        if signal is None:
            return {"id": signal_id, "deleted": False}
        await self.detach_signals([signal.id], now=datetime.now(UTC))
        await self._session.delete(signal)
        await self._session.flush()
        surviving_signal = await self._session.scalar(select(KnowledgeFeedbackSignal.id).where(KnowledgeFeedbackSignal.id == signal_id))
        surviving_reference = await self._session.scalar(select(ReviewWorkItem.id).where(ReviewWorkItem.signal_id == signal_id))
        surviving_maintenance_link = await self._session.scalar(select(MaintenanceSignalLink.item_id).where(
            MaintenanceSignalLink.signal_id == signal_id,
        ))
        if surviving_signal is not None or surviving_reference is not None or surviving_maintenance_link is not None:
            await self._session.rollback()
            raise AppError(
                status_code=503, code="PRIVACY_DELETE_UNVERIFIED",
                message="feedback deletion could not be verified; retry is required",
            )
        await self._session.commit()
        return {"id": signal_id, "deleted": True}

    async def list_review_queue(self) -> dict[str, Any]:
        await self._purge_expired()
        await self._sync_published_triggers()
        result = await self._session.execute(
            select(ReviewWorkItem).order_by(
                ReviewWorkItem.status.asc(), ReviewWorkItem.created_at.asc(), ReviewWorkItem.id.asc()
            )
        )
        items = [await self._project_work_item(item) for item in result.scalars().all()]
        await self._session.commit()
        return {"items": items, "pending_count": sum(item["status"] == "pending" for item in items)}

    async def classify(self, *, item_id: str, payload: ReviewWorkItemUpdate) -> dict[str, Any]:
        await self._purge_expired()
        result = await self._session.execute(select(ReviewWorkItem).where(ReviewWorkItem.id == item_id).with_for_update())
        item = result.scalar_one_or_none()
        if item is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="review work item not found")
        item.classification = payload.classification
        item.status = payload.status
        item.updated_at = datetime.now(UTC)
        await self._session.commit()
        return await self._project_work_item(item)

    @staticmethod
    def _knowledge_edition(rag_trace: object, sources: list[object]) -> str:
        if isinstance(rag_trace, dict):
            edition = rag_trace.get("knowledge_edition")
            if isinstance(edition, str) and _SAFE_CODE.fullmatch(edition.strip()):
                return edition.strip()
        versions = sorted(
            {version for source in sources if (version := _source_publication_version(source)) is not None}
        )
        if not versions:
            raise AppError(
                status_code=422,
                code="KNOWLEDGE_FEEDBACK_EDITION_UNAVAILABLE",
                message="answer knowledge edition is unavailable",
            )
        return f"publication:{'+'.join(versions)}"[:128]

    async def _purge_expired(self) -> None:
        await self.purge_expired()

    async def purge_expired(self, *, now: datetime | None = None, retention_days: int | None = None) -> None:
        await lock_registry(self._session)
        now = now or datetime.now(UTC)
        days = retention_days if retention_days is not None else (await read_policy(self._session))["days"]["feedback_signals"]
        expired = or_(
            KnowledgeFeedbackSignal.expires_at <= now,
            KnowledgeFeedbackSignal.created_at <= now - timedelta(days=days),
        )
        expired_ids = list((await self._session.scalars(select(KnowledgeFeedbackSignal.id).where(expired).with_for_update())).all())
        await self.detach_signals(expired_ids, now=now)
        await self._session.execute(delete(KnowledgeFeedbackSignal).where(KnowledgeFeedbackSignal.id.in_(expired_ids)))
        await self._detach_items(await self.unresolved_feedback_references(), now=now)
        await self._session.execute(delete(MaintenanceSignalLink).where(
            ~MaintenanceSignalLink.signal_id.in_(select(KnowledgeFeedbackSignal.id)),
        ))

    async def unresolved_feedback_references(self) -> list[ReviewWorkItem]:
        items = (await self._session.scalars(select(ReviewWorkItem).where(
            ReviewWorkItem.kind == "feedback_signal",
            or_(
                ReviewWorkItem.signal_id.is_(None),
                ~ReviewWorkItem.signal_id.in_(select(KnowledgeFeedbackSignal.id)),
            ),
        ).execution_options(populate_existing=True).with_for_update())).all()
        return [
            item for item in items if (
                item.signal_id is not None or item.classification is None or item.normalized_metadata
                or item.dedupe_key != f"detached:{item.id}" or item.subject_id.startswith("gap:")
            )
        ]

    async def detach_signals(self, signal_ids: list[str], *, now: datetime) -> None:
        await self._session.execute(delete(MaintenanceSignalLink).where(
            MaintenanceSignalLink.signal_id.in_(signal_ids),
        ))
        items = (await self._session.scalars(select(ReviewWorkItem).where(
            ReviewWorkItem.signal_id.in_(signal_ids),
        ).with_for_update())).all()
        await self._detach_items(list(items), now=now)

    async def _detach_items(self, items: list[ReviewWorkItem], *, now: datetime) -> None:
        for item in items:
            if item.classification is None:
                await self._session.delete(item)
                continue
            item.signal_id = None
            item.dedupe_key = f"detached:{item.id}"
            item.normalized_metadata = {}
            if item.subject_id.startswith("gap:"):
                item.subject_id = "detached-feedback"
            item.created_at = now
            item.updated_at = now
        await self._session.flush()

    async def _sync_published_triggers(self) -> None:
        result = await self._session.execute(
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.deleted_at.is_(None),
                Document.published_generation > 0,
                Document.chunk_strategy == "agent",
                DocumentChunk.generation == Document.published_generation,
            )
            .order_by(Document.id.asc(), DocumentChunk.chunk_index.asc())
        )
        existing_keys = set((await self._session.scalars(select(ReviewWorkItem.dedupe_key))).all())
        today = datetime.now(UTC).date()
        seen_entries: set[str] = set()
        for chunk, document in result.all():
            metadata = chunk.chunk_metadata if isinstance(chunk.chunk_metadata, dict) else {}
            entry_id = metadata.get("entry_id")
            if not isinstance(entry_id, str) or not entry_id.strip() or entry_id in seen_entries:
                continue
            entry_id = entry_id.strip()
            seen_entries.add(entry_id)

            review_date = metadata.get("review_date")
            if isinstance(review_date, str):
                try:
                    stale = today - date.fromisoformat(review_date) > timedelta(days=REVIEW_AGE_DAYS)
                except ValueError:
                    stale = True
                if stale:
                    self._add_trigger(
                        existing_keys,
                        kind="review_age",
                        dedupe_key=f"review-age:{entry_id}:{review_date}",
                        entry_id=entry_id,
                        metadata={"entry_id": entry_id, "review_date": review_date, "threshold_days": REVIEW_AGE_DAYS},
                    )

            failed_source_count = self._failed_source_count(metadata)
            if failed_source_count:
                self._add_trigger(
                    existing_keys,
                    kind="source_link_failure",
                    dedupe_key=f"source-link:{entry_id}:{document.published_generation}",
                    entry_id=entry_id,
                    metadata={"entry_id": entry_id, "failed_source_count": failed_source_count},
                )

            if (
                document.candidate_generation is not None
                and document.candidate_generation > document.published_generation
                and document.candidate_chunk_strategy == "agent"
            ):
                self._add_trigger(
                    existing_keys,
                    kind="release_change",
                    dedupe_key=f"release-change:{entry_id}:{document.candidate_generation}",
                    entry_id=entry_id,
                    metadata={
                        "entry_id": entry_id,
                        "published_version": f"v{document.published_generation}",
                        "candidate_version": f"v{document.candidate_generation}",
                    },
                )

    @staticmethod
    def _failed_source_count(metadata: dict[str, object]) -> int:
        count = int(metadata.get("source_availability") not in (None, "verified"))
        sources = metadata.get("sources")
        if isinstance(sources, list):
            count += sum(
                1
                for source in sources
                if not isinstance(source, dict) or source.get("availability") != "verified"
            )
        return count

    def _add_trigger(
        self,
        existing_keys: set[str],
        *,
        kind: str,
        dedupe_key: str,
        entry_id: str,
        metadata: dict[str, object],
    ) -> None:
        if dedupe_key in existing_keys:
            return
        existing_keys.add(dedupe_key)
        self._session.add(
            ReviewWorkItem(
                kind=kind,
                dedupe_key=dedupe_key,
                subject_id=entry_id,
                normalized_metadata=metadata,
            )
        )

    @staticmethod
    def _project_signal(signal: KnowledgeFeedbackSignal, *, duplicate: bool) -> dict[str, Any]:
        projection = {
            "id": signal.id,
            "answer_id": signal.answer_id,
            "entry_id": signal.entry_id,
            "knowledge_edition": signal.knowledge_edition,
            "label": signal.label,
            "created_at": signal.created_at.isoformat(),
            "expires_at": signal.expires_at.isoformat(),
            "retention_days": (signal.expires_at - signal.created_at).days,
            "duplicate": duplicate,
        }
        outcome = signal.normalized_metadata.get("outcome")
        if outcome in {"evidence_gated_answer", "insufficient_evidence_reply"}:
            projection["outcome"] = outcome
        for key in ("answer_execution_id", "query_condition_set_identity", "knowledge_version_identities"):
            if key in signal.normalized_metadata:
                projection[key] = signal.normalized_metadata[key]
        gap_context = signal.normalized_metadata.get("gap_context")
        if isinstance(gap_context, dict):
            projection["gap_context"] = gap_context
        return projection

    async def _project_work_item(self, item: ReviewWorkItem) -> dict[str, Any]:
        metadata = item.normalized_metadata if isinstance(item.normalized_metadata, dict) else {}
        if item.kind == "feedback_signal":
            metadata = {}
            signal = await self._session.get(KnowledgeFeedbackSignal, item.signal_id) if item.signal_id else None
            if signal is not None:
                expires_at = signal.expires_at.replace(tzinfo=UTC) if signal.expires_at.tzinfo is None else signal.expires_at
                if expires_at > datetime.now(UTC):
                    raw = signal.normalized_metadata
                    metadata = {key: raw[key] for key in ("outcome", "gap_context", "evidence_coverage", "source_count") if key in raw}
                    metadata.update({"answer_id": signal.answer_id, "label": signal.label, "note": signal.note})
                    if signal.entry_id is not None:
                        metadata.update({"entry_id": signal.entry_id, "knowledge_edition": signal.knowledge_edition})
        return {
            "id": item.id,
            "kind": item.kind,
            "subject_id": item.subject_id,
            "status": item.status,
            "classification": item.classification,
            "created_at": item.created_at.isoformat(),
            "metadata": metadata,
        }
