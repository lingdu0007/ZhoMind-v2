from app.common.security import build_auth_session_key, create_access_token, decode_access_token, hash_password
from app.repository.user_repository import UserRepository


async def create_authenticated_test_token(session_factory, redis, *, username: str, role: str = "user") -> str:
    async with session_factory() as session:
        users = UserRepository(session)
        user = await users.get_by_username(username)
        if user is None:
            user = await users.create_user(username=username, password_hash=hash_password("test-password"), role=role)
            await session.commit()
        elif user.role != role:
            user.role = role
            await session.commit()

    token = create_access_token(subject=user.username, role=user.role)
    payload = decode_access_token(token)
    key = build_auth_session_key(subject=user.username, jti=payload["jti"])
    await redis.hset(key, mapping={"username": user.username, "role": user.role, "issued_at": str(payload["iat"])})
    await redis.expire(key, max(int(payload["exp"]) - int(payload["iat"]), 1))
    return token
