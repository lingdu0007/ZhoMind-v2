from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.knowledge_feedback.schemas import KnowledgeFeedbackCreate, ReviewWorkItemUpdate
from app.model.chat import ChatMessage
from app.model.document import Document, DocumentChunk
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, ReviewWorkItem
from app.rag.answer_evidence import evidence_summary_from_trace

FEEDBACK_RETENTION_DAYS = 180
REVIEW_AGE_DAYS = 90
_SAFE_CODE = re.compile(r"^[A-Za-z0-9._:+-]{1,96}$")


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

        summary = evidence_summary_from_trace(answer.rag_trace)
        sources = summary.get("sources") if isinstance(summary, dict) else []
        sources = sources if isinstance(sources, list) else []
        entry_sources = [source for source in sources if _source_entry_id(source) == payload.entry_id]
        if not entry_sources:
            raise AppError(
                status_code=422,
                code="KNOWLEDGE_FEEDBACK_ENTRY_MISMATCH",
                message="entry is not part of the answer evidence",
            )

        edition = self._knowledge_edition(answer.rag_trace, entry_sources)
        existing_result = await self._session.execute(
            select(KnowledgeFeedbackSignal).where(
                KnowledgeFeedbackSignal.user_id == user_id,
                KnowledgeFeedbackSignal.answer_id == payload.answer_id,
                KnowledgeFeedbackSignal.entry_id == payload.entry_id,
            )
        )
        existing = existing_result.scalar_one_or_none()
        if existing is not None:
            if existing.label == payload.label and existing.note == payload.note:
                await self._session.commit()
                return self._project_signal(existing, duplicate=True)
            raise AppError(
                status_code=409,
                code="KNOWLEDGE_FEEDBACK_ALREADY_SUBMITTED",
                message="feedback already exists for this answer and entry",
            )

        now = datetime.now(UTC)
        metadata = {
            "evidence_coverage": summary.get("coverage", "unavailable"),
            "source_count": len(entry_sources),
        }
        signal = KnowledgeFeedbackSignal(
            answer_id=payload.answer_id,
            user_id=user_id,
            entry_id=payload.entry_id,
            knowledge_edition=edition,
            label=payload.label,
            note=payload.note,
            normalized_metadata=metadata,
            created_at=now,
            expires_at=now + timedelta(days=FEEDBACK_RETENTION_DAYS),
        )
        self._session.add(signal)
        await self._session.flush()
        self._session.add(
            ReviewWorkItem(
                kind="feedback_signal",
                dedupe_key=f"feedback:{signal.id}",
                subject_id=signal.entry_id,
                signal_id=signal.id,
                normalized_metadata={
                    "answer_id": signal.answer_id,
                    "entry_id": signal.entry_id,
                    "knowledge_edition": signal.knowledge_edition,
                    "label": signal.label,
                    "note": signal.note,
                    **metadata,
                },
                created_at=now,
                updated_at=now,
            )
        )
        await self._session.commit()
        return self._project_signal(signal, duplicate=False)

    async def delete(self, *, user_id: str, signal_id: str) -> dict[str, Any]:
        result = await self._session.execute(
            select(KnowledgeFeedbackSignal).where(
                KnowledgeFeedbackSignal.id == signal_id,
                KnowledgeFeedbackSignal.user_id == user_id,
            )
        )
        signal = result.scalar_one_or_none()
        if signal is None:
            return {"id": signal_id, "deleted": False}
        await self._session.execute(delete(ReviewWorkItem).where(ReviewWorkItem.signal_id == signal.id))
        await self._session.delete(signal)
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
        items = [self._project_work_item(item) for item in result.scalars().all()]
        await self._session.commit()
        return {"items": items, "pending_count": sum(item["status"] == "pending" for item in items)}

    async def classify(self, *, item_id: str, payload: ReviewWorkItemUpdate) -> dict[str, Any]:
        result = await self._session.execute(select(ReviewWorkItem).where(ReviewWorkItem.id == item_id))
        item = result.scalar_one_or_none()
        if item is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="review work item not found")
        item.classification = payload.classification
        item.status = payload.status
        item.updated_at = datetime.now(UTC)
        await self._session.commit()
        return self._project_work_item(item)

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
        now = datetime.now(UTC)
        expired_ids = select(KnowledgeFeedbackSignal.id).where(KnowledgeFeedbackSignal.expires_at <= now)
        await self._session.execute(delete(ReviewWorkItem).where(ReviewWorkItem.signal_id.in_(expired_ids)))
        await self._session.execute(delete(KnowledgeFeedbackSignal).where(KnowledgeFeedbackSignal.expires_at <= now))

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
        return {
            "id": signal.id,
            "answer_id": signal.answer_id,
            "entry_id": signal.entry_id,
            "knowledge_edition": signal.knowledge_edition,
            "label": signal.label,
            "created_at": signal.created_at.isoformat(),
            "expires_at": signal.expires_at.isoformat(),
            "retention_days": FEEDBACK_RETENTION_DAYS,
            "duplicate": duplicate,
        }

    @staticmethod
    def _project_work_item(item: ReviewWorkItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "kind": item.kind,
            "subject_id": item.subject_id,
            "status": item.status,
            "classification": item.classification,
            "created_at": item.created_at.isoformat(),
            "metadata": item.normalized_metadata if isinstance(item.normalized_metadata, dict) else {},
        }
