from typing import Protocol


class TaskBackend(Protocol):
    async def enqueue(self, name: str, payload: dict) -> str: ...

    async def get_status(self, task_id: str) -> dict: ...

    async def cancel(self, task_id: str) -> None: ...


class InMemoryTaskBackend:
    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._next_id = 1

    async def enqueue(self, name: str, payload: dict) -> str:
        payload_task_id = payload.get("job_id") if isinstance(payload, dict) else None
        task_id = payload_task_id if isinstance(payload_task_id, str) and payload_task_id else f"task_{self._next_id}"
        if task_id.startswith("task_"):
            self._next_id += 1
        self._tasks[task_id] = {
            "task_id": task_id,
            "name": name,
            "payload": payload,
            "status": "queued",
        }
        return task_id

    async def get_status(self, task_id: str) -> dict:
        item = self._tasks.get(task_id)
        if item is None:
            return {"task_id": task_id, "status": "missing"}
        return {
            "task_id": item["task_id"],
            "status": item["status"],
            "name": item["name"],
        }

    async def cancel(self, task_id: str) -> None:
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = "canceled"


def create_inmemory_task_backend() -> InMemoryTaskBackend:
    return InMemoryTaskBackend()
