from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.model.answer_execution import AnswerExecutionEventModel, AnswerExecutionModel
from app.model.chat import ChatMessage, ChatSession
from app.retention.policy import read_policy


def orphaned_private_record_predicates():
    return (
        (ChatMessage, ~select(ChatSession.id).where(
            ChatSession.id == ChatMessage.session_id, ChatSession.user_id == ChatMessage.user_id,
        ).exists()),
        (AnswerExecutionModel, ~select(ChatSession.id).where(
            ChatSession.id == AnswerExecutionModel.session_id, ChatSession.user_id == AnswerExecutionModel.user_id,
        ).exists()),
        (AnswerExecutionEventModel, ~AnswerExecutionEventModel.execution_id.in_(select(AnswerExecutionModel.id))),
    )


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_session(self, session_id: str, user_id: str) -> ChatSession | None:
        result = await self.session.execute(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def list_sessions(self, user_id: str, limit: int = 20) -> list[ChatSession]:
        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def create_session(self, session_id: str, user_id: str) -> ChatSession:
        session = ChatSession(id=session_id, user_id=user_id)
        self.session.add(session)
        await self.session.flush()
        return session

    async def get_or_create_session(self, session_id: str, user_id: str) -> ChatSession:
        existing = await self.get_session(session_id=session_id, user_id=user_id)
        if existing is not None:
            return existing
        return await self.create_session(session_id=session_id, user_id=user_id)

    async def get_or_create_session_for_admission(self, session_id: str, user_id: str) -> ChatSession:
        """Serialize a private execution admission with verified retention."""

        await self.acquire_private_conversation_write_fence()
        existing = await self._locked_session(session_id=session_id, user_id=user_id)
        if existing is not None:
            return existing
        return await self.create_session(session_id=session_id, user_id=user_id)

    async def acquire_private_conversation_write_fence(self) -> None:
        """Use SQLite's transaction-wide writer fence where row locks are unavailable."""

        bind = getattr(self.session, "bind", None)
        in_transaction = getattr(self.session, "in_transaction", None)
        if bind is None or bind.dialect.name != "sqlite":
            return
        if callable(in_transaction) and in_transaction():
            # Authentication can leave SQLite in a deferred read transaction.
            # This no-op write upgrades that transaction to the same writer
            # fence without discarding an already-pending caller mutation.
            await self.session.execute(text("UPDATE chat_sessions SET id = id WHERE 0"))
            return
        await self.session.execute(text("BEGIN IMMEDIATE"))

    async def list_messages(self, session_id: str, user_id: str) -> list[ChatMessage]:
        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.asc())
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        session_id: str,
        user_id: str,
        message_type: str,
        content: str,
        answer_execution_id: str | None = None,
        rag_trace: dict | None = None,
        message_id: str | None = None,
    ) -> ChatMessage:
        if message_id is not None and (not isinstance(message_id, str) or not message_id):
            raise ValueError("conversation message identity is malformed")
        values = {
            "session_id": session_id,
            "user_id": user_id,
            "type": message_type,
            "content": content,
            "answer_execution_id": answer_execution_id,
            "rag_trace": rag_trace,
        }
        if message_id is not None:
            values["id"] = message_id
        message = ChatMessage(**values)
        self.session.add(message)
        await self.session.flush()
        return message

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        await self.acquire_private_conversation_write_fence()
        # Existing execution writers lock a header before their event trail.
        # Match that ordering, then fence the conversation and rescan so an
        # admission that held the session lock just before us is included.
        await self._locked_execution_ids(session_id=session_id, user_id=user_id)
        session = await self._locked_session(session_id=session_id, user_id=user_id)
        if session is None:
            return False

        execution_ids = await self._locked_execution_ids(session_id=session_id, user_id=user_id)
        await self._delete_execution_ids(execution_ids)
        await self.session.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id))
        await self.session.delete(session)
        await self.session.flush()
        for model, predicate in (
            (ChatSession, ChatSession.id == session_id),
            (ChatMessage, ChatMessage.session_id == session_id),
            (AnswerExecutionModel, AnswerExecutionModel.session_id == session_id),
            (AnswerExecutionEventModel, AnswerExecutionEventModel.execution_id.in_(execution_ids)),
        ):
            if await self.session.scalar(select(model.id).where(predicate).limit(1)) is not None:
                await self.session.rollback()
                raise AppError(
                    status_code=503, code="PRIVACY_DELETE_UNVERIFIED",
                    message="conversation deletion could not be verified; retry is required",
                )
        return True

    async def purge_expired_sessions(self, *, now: datetime | None = None, retention_days: int | None = None) -> int:
        await self.acquire_private_conversation_write_fence()
        days = retention_days if retention_days is not None else (await read_policy(self.session))["days"]["conversations"]
        cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
        candidate_session_ids = list(
            (
                await self.session.scalars(
                    select(ChatSession.id)
                    .where(ChatSession.created_at <= cutoff)
                    .order_by(ChatSession.id.asc())
                )
            ).all()
        )
        if not candidate_session_ids:
            await self._purge_orphaned_private_records()
            return 0
        await self._locked_execution_ids_for_sessions(candidate_session_ids)
        expired_session_ids = list(
            (
                await self.session.scalars(
                    select(ChatSession.id)
                    .where(ChatSession.id.in_(candidate_session_ids), ChatSession.created_at <= cutoff)
                    .order_by(ChatSession.id.asc())
                    .with_for_update()
                )
            ).all()
        )
        if not expired_session_ids:
            await self._purge_orphaned_private_records()
            return 0
        execution_ids = await self._locked_execution_ids_for_sessions(expired_session_ids)
        await self._delete_execution_ids(execution_ids)
        await self.session.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(expired_session_ids)))
        await self.session.execute(delete(ChatSession).where(ChatSession.id.in_(expired_session_ids)))
        await self.session.flush()
        await self._purge_orphaned_private_records()
        return len(expired_session_ids)

    async def _purge_orphaned_private_records(self) -> None:
        predicates = dict(orphaned_private_record_predicates())
        execution_ids = list((await self.session.scalars(
            select(AnswerExecutionModel.id).where(predicates[AnswerExecutionModel])
            .order_by(AnswerExecutionModel.id).with_for_update(),
        )).all())
        await self._delete_execution_ids(execution_ids)
        await self.session.execute(delete(ChatMessage).where(predicates[ChatMessage]))
        orphan_event_execution_ids = list((await self.session.scalars(
            select(AnswerExecutionEventModel.execution_id).where(predicates[AnswerExecutionEventModel])
            .order_by(AnswerExecutionEventModel.execution_id, AnswerExecutionEventModel.id).with_for_update(),
        )).all())
        await self._delete_execution_ids(list(dict.fromkeys(orphan_event_execution_ids)))
        await self.session.flush()

    async def _locked_session(self, *, session_id: str, user_id: str) -> ChatSession | None:
        return await self.session.scalar(
            select(ChatSession)
            .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )

    async def _locked_execution_ids(self, *, session_id: str, user_id: str) -> list[str]:
        return list(
            (
                await self.session.scalars(
                    select(AnswerExecutionModel.id)
                    .where(
                        AnswerExecutionModel.session_id == session_id,
                        AnswerExecutionModel.user_id == user_id,
                    )
                    .order_by(AnswerExecutionModel.id.asc())
                    .with_for_update()
                )
            ).all()
        )

    async def _locked_execution_ids_for_sessions(self, session_ids: list[str]) -> list[str]:
        if not session_ids:
            return []
        return list(
            (
                await self.session.scalars(
                    select(AnswerExecutionModel.id)
                    .where(AnswerExecutionModel.session_id.in_(session_ids))
                    .order_by(AnswerExecutionModel.id.asc())
                    .with_for_update()
                )
            ).all()
        )

    async def _delete_execution_ids(self, execution_ids: list[str]) -> None:
        if execution_ids:
            # Immutable bindings still identify private messages whose mutable
            # session or execution index was corrupted. Remove them before
            # deleting the binding facts that prevent cross-member disclosure.
            bound_messages = or_(
                ChatMessage.id.in_(select(AnswerExecutionModel.request["user_message_id"].as_string()).where(
                    AnswerExecutionModel.id.in_(execution_ids),
                )),
                ChatMessage.id.in_(select(
                    AnswerExecutionEventModel.payload["data"]["assistant_message_id"].as_string(),
                ).where(AnswerExecutionEventModel.execution_id.in_(execution_ids))),
            )
            await self.session.execute(delete(ChatMessage).where(bound_messages))
            if await self.session.scalar(select(ChatMessage.id).where(bound_messages).limit(1)) is not None:
                raise AppError(
                    status_code=503, code="PRIVACY_DELETE_UNVERIFIED",
                    message="conversation deletion could not be verified; retry is required",
                )
            await self.session.execute(
                delete(AnswerExecutionEventModel).where(AnswerExecutionEventModel.execution_id.in_(execution_ids))
            )
            await self.session.execute(delete(AnswerExecutionModel).where(AnswerExecutionModel.id.in_(execution_ids)))
