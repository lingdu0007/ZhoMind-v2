import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.model.team_invitation import TeamInvitation


class TeamInvitationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        code_hash: str,
        created_by_user_id: uuid.UUID,
        expires_at,
    ) -> TeamInvitation:
        invitation = TeamInvitation(
            code_hash=code_hash,
            created_by_user_id=created_by_user_id,
            expires_at=expires_at,
        )
        self.session.add(invitation)
        await self.session.flush()
        return invitation

    async def get_by_code_hash(self, code_hash: str) -> TeamInvitation | None:
        result = await self.session.execute(select(TeamInvitation).where(TeamInvitation.code_hash == code_hash))
        return result.scalar_one_or_none()

    async def get_by_code_hash_for_update(self, code_hash: str) -> TeamInvitation | None:
        result = await self.session.execute(
            select(TeamInvitation).where(TeamInvitation.code_hash == code_hash).with_for_update()
        )
        return result.scalar_one_or_none()

    async def consume_once(
        self,
        *,
        invitation_id: uuid.UUID,
        user_id: uuid.UUID,
        consumed_at: datetime,
    ) -> bool:
        result = await self.session.execute(
            update(TeamInvitation)
            .where(
                TeamInvitation.id == invitation_id,
                TeamInvitation.revoked_at.is_(None),
                TeamInvitation.consumed_at.is_(None),
                TeamInvitation.expires_at > consumed_at,
            )
            .values(consumed_at=consumed_at, consumed_by_user_id=user_id)
            .execution_options(synchronize_session="fetch")
        )
        return result.rowcount == 1

    async def get_by_id(self, invitation_id: uuid.UUID) -> TeamInvitation | None:
        return await self.session.get(TeamInvitation, invitation_id)

    async def list_all(self) -> list[TeamInvitation]:
        result = await self.session.execute(select(TeamInvitation).order_by(TeamInvitation.created_at.desc()))
        return list(result.scalars())
