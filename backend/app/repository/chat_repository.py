from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.model.chat import ChatMessage, ChatSession


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
        rag_trace: dict | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            user_id=user_id,
            type=message_type,
            content=content,
            rag_trace=rag_trace,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        session = await self.get_session(session_id=session_id, user_id=user_id)
        if session is None:
            return False

        await self.session.execute(
            delete(ChatMessage).where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
        )
        await self.session.delete(session)
        await self.session.flush()
        return True
