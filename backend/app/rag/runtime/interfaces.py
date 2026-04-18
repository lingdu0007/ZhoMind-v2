from typing import Protocol

from app.rag.runtime.state import RagStateDict


class RuntimeNode(Protocol):
    async def run(self, state: RagStateDict) -> RagStateDict: ...


class VerifyPolicy(Protocol):
    async def evaluate(self, state: RagStateDict) -> dict: ...


class MemoryWritePolicy(Protocol):
    async def should_write(self, fact: dict, existing: list[dict]) -> dict: ...
