class InMemorySessionStore:
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    async def read(self, session_id: str) -> dict:
        return self._store.get(session_id, {})

    async def write(self, session_id: str, payload: dict) -> None:
        self._store[session_id] = payload


class InMemoryUserStore:
    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {}

    async def read(self, user_id: str) -> list[dict]:
        return list(self._store.get(user_id, []))

    async def append(self, user_id: str, fact: dict) -> None:
        self._store.setdefault(user_id, []).append(fact)
