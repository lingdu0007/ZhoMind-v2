
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.model.user import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_username(self, username: str) -> User | None:
        result = await self.session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def create_user(self, username: str, password_hash: str, role: str) -> User:
        user = User(username=username, password_hash=password_hash, role=role)
        self.session.add(user)
        await self.session.flush()
        return user

    async def has_bootstrap_administrator(self) -> bool:
        result = await self.session.execute(select(User.id).where(User.is_bootstrap_administrator.is_(True)).limit(1))
        return result.scalar_one_or_none() is not None

    async def get_bootstrap_administrator(self) -> User | None:
        result = await self.session.execute(
            select(User).where(User.is_bootstrap_administrator.is_(True)).order_by(User.created_at.asc()).limit(1)
        )
        return result.scalar_one_or_none()

    async def list_members(self) -> list[User]:
        result = await self.session.execute(select(User).order_by(User.created_at.asc(), User.username.asc()))
        return list(result.scalars())

    async def count_active_members(self) -> int:
        result = await self.session.scalar(
            select(func.count()).select_from(User).where(User.is_active.is_(True))
        )
        return int(result or 0)

    async def lock_active_members(self) -> None:
        await self.session.execute(select(User.id).where(User.is_active.is_(True)).with_for_update())

    async def promote(self, user: User) -> User:
        user.role = "admin"
        await self.session.flush()
        return user

    async def deactivate(self, user: User) -> User:
        user.is_active = False
        await self.session.flush()
        return user
