import uuid

from sqlalchemy import select
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

    async def get_by_id(self, invitation_id: uuid.UUID) -> TeamInvitation | None:
        return await self.session.get(TeamInvitation, invitation_id)

    async def list_all(self) -> list[TeamInvitation]:
        result = await self.session.execute(select(TeamInvitation).order_by(TeamInvitation.created_at.desc()))
        return list(result.scalars())
