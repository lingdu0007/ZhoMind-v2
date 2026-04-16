import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.user_repository import UserRepository


@pytest.mark.asyncio
async def test_user_repository_create_and_get(db_session: AsyncSession) -> None:
    repo = UserRepository(db_session)
    created = await repo.create_user(username="alice", password_hash="hash", role="user")
    fetched = await repo.get_by_username("alice")
    assert created.id == fetched.id
    assert fetched.role == "user"
