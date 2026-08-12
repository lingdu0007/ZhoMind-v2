import asyncio
import json
import os
import tempfile
import time
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.documents import parsers
from app.extensions.registry import get_extension_registry
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.service.chat_service import CHAT_JUDGE_PROVIDER, CHAT_RERANK_PROVIDER, CHAT_RETRIEVER_PROVIDER
from tests.support.auth import create_authenticated_test_token


class _InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._store[key] = dict(mapping)
        return len(mapping)

    async def expire(self, key: str, _ttl: int) -> bool:
        return key in self._store

    async def exists(self, key: str) -> int:
        return int(key in self._store)

    async def get(self, key: str) -> str | None:
        value = self._store.get(key)
        return value if isinstance(value, str) else None

    async def set(self, key: str, value: str) -> bool:
        self._store[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        return sum(self._store.pop(key, None) is not None for key in keys)


class _RecordingLlmProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        return """## 建议
在执行路径已知时优先使用 deterministic workflow。[S1]

## 适用边界
仅适用于步骤和正常分支可预先定义的任务。[S1]

## 备选方案
必须依据运行时 tool observation 选择下一步时，使用 bounded Agent。[S1]

## 最小实现或验收检查
固定输入可重放，并为动态 loop 设置 step 与 tool budget。[S1]"""


_PILOT_ENTRY = """---
entry_id: pae-workflow-001
title: Prefer deterministic workflows when the path is known
domain: workflow-vs-agent
review_status: approved
applicable_versions:
  - framework-neutral
review_date: 2026-08-12
sources:
  - title: Building Effective Agents
    authority: Anthropic
    url: https://www.anthropic.com/research/building-effective-agents
    version: 2024-12-19
    availability: verified
---
# Decision Question

什么时候应该优先使用 deterministic workflow 而不是 Agent？

## Recommendation

当执行路径已知且步骤可以预先定义时，优先使用 deterministic workflow；只在需要模型动态决定过程时使用 Agent。

## Validation

验证同一输入的步骤顺序、tool 参数和结果可以重放。
""".encode()


def _extract_data(payload: dict) -> dict:
    return payload.get("data") or payload


def _poll_job(client: TestClient, headers: dict[str, str], job_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/documents/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        job = _extract_data(response.json())
        if job["status"] in {"succeeded", "failed", "canceled"}:
            return job
        time.sleep(0.02)
    raise AssertionError("Agent entry build did not complete")


def test_agent_entry_is_retrievable_through_authenticated_chat_only_after_publish(monkeypatch) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="agent-entry-publication-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def init_db() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def override_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.state.test_auth_session_factory = session_factory
    app.state.test_auth_redis = fake_redis
    monkeypatch.setattr(parsers, "_probe_public_source", lambda _url: None, raising=False)
    asyncio.run(init_db())

    registry = get_extension_registry()
    settings = get_settings()
    llm = _RecordingLlmProvider()
    previous_llm = registry.get_llm(settings.rag_primary_llm_provider)
    registry.register_llm(settings.rag_primary_llm_provider, llm)
    registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
    registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
    registry.judges.pop(CHAT_JUDGE_PROVIDER, None)

    try:
        with TestClient(app) as client:
            admin_token = asyncio.run(
                create_authenticated_test_token(session_factory, fake_redis, username="entry-admin", role="admin")
            )
            user_token = asyncio.run(
                create_authenticated_test_token(session_factory, fake_redis, username="knowledge-user", role="user")
            )
            admin_headers = {"Authorization": f"Bearer {admin_token}"}
            user_headers = {"Authorization": f"Bearer {user_token}"}

            upload = client.post(
                "/api/v1/documents/upload",
                headers=admin_headers,
                data={"chunk_strategy": "agent"},
                files={"file": ("pae-workflow-001.md", _PILOT_ENTRY, "text/markdown")},
            )
            assert upload.status_code == 200
            uploaded = _extract_data(upload.json())
            job = _poll_job(client, admin_headers, uploaded["job_id"])
            assert job["status"] == "succeeded", job
            assert "awaiting publication" in job["message"]

            candidate_chat = client.post(
                "/api/v1/chat",
                headers=user_headers,
                json={"message": "什么时候使用 deterministic workflow？", "session_id": "candidate-hidden"},
            )
            assert candidate_chat.status_code == 200
            assert _extract_data(candidate_chat.json())["outcome"] == "insufficient_evidence_reply"
            assert llm.prompts == []

            chunks = client.get(
                f"/api/v1/documents/{uploaded['document_id']}/chunks?page=1&page_size=20",
                headers=admin_headers,
            )
            assert chunks.status_code == 200
            chunk_items = _extract_data(chunks.json())["items"]
            assert {item["metadata"]["section_id"] for item in chunk_items} == {
                "decision-question",
                "recommendation",
                "validation",
            }
            assert all(item["metadata"]["entry_id"] == "pae-workflow-001" for item in chunk_items)

            publish = client.post(f"/api/v1/documents/{uploaded['document_id']}/publish", headers=admin_headers)
            assert publish.status_code == 200

            published_chat = client.post(
                "/api/v1/chat",
                headers=user_headers,
                json={"message": "什么时候使用 deterministic workflow？", "session_id": "published-visible"},
            )
            assert published_chat.status_code == 200
            answer = _extract_data(published_chat.json())
            assert answer["outcome"] == "evidence_gated_answer"
            source = answer["message"]["evidence_summary"]["sources"][0]
            assert source == {
                "citation_id": "S1",
                "entry_id": "pae-workflow-001",
                "entry_title": "Prefer deterministic workflows when the path is known",
                "domain": "workflow-vs-agent",
                "section_id": "decision-question",
                "source_title": "Building Effective Agents",
                "source_authority": "Anthropic",
                "source_url": "https://www.anthropic.com/research/building-effective-agents",
                "source_version": "2024-12-19",
                "publication_version": "v1",
                "review_date": "2026-08-12",
                "excerpt": "# Decision Question 什么时候应该优先使用 deterministic workflow 而不是 Agent？",
                "snapshot_id": source["snapshot_id"],
            }
            assert "score" not in source
            assert "retrieval_source" not in source
            assert "source_id" not in source
            assert "retrieval_diagnostics" not in answer
            assert llm.prompts
            prompt_source = json.loads(llm.prompts[-1])["evidence_sources"][0]
            assert prompt_source["excerpt"] == source["excerpt"]
            assert prompt_source["snapshot_id"] == source["snapshot_id"]

            streamed_chat = client.post(
                "/api/v1/chat/stream",
                headers=user_headers,
                json={"message": "什么时候使用 deterministic workflow？", "session_id": "published-stream"},
            )
            assert streamed_chat.status_code == 200
            stream_summary = None
            current_event = None
            for line in streamed_chat.text.splitlines():
                if line.startswith("event: "):
                    current_event = line.removeprefix("event: ")
                elif line.startswith("data: ") and current_event == "evidence_summary":
                    stream_summary = json.loads(line.removeprefix("data: "))["evidence_summary"]
            assert stream_summary == answer["message"]["evidence_summary"]

            history = client.get("/api/v1/sessions/published-stream", headers=user_headers)
            assert history.status_code == 200
            history_messages = _extract_data(history.json())["messages"]
            history_assistant = next(item for item in history_messages if item["type"] == "assistant")
            assert history_assistant["evidence_summary"] == stream_summary
            stream_snapshot_ids = stream_summary["provider_prompt_snapshot_ids"]
            assert stream_snapshot_ids
            assert all(len(snapshot_id) == 64 for snapshot_id in stream_snapshot_ids)
            assert stream_snapshot_ids == [
                item["snapshot_id"] for item in stream_summary["sources"]
            ]

            invalid_upload = client.post(
                "/api/v1/documents/upload",
                headers=admin_headers,
                data={"chunk_strategy": "agent"},
                files={"file": ("invalid-entry.md", b"---\ntitle: Missing identity\n---\n# Decision\n", "text/markdown")},
            )
            invalid_job = _poll_job(client, admin_headers, _extract_data(invalid_upload.json())["job_id"])
            assert invalid_job["status"] == "failed"
            assert "AGENT_ENTRY_METADATA_INVALID" in invalid_job["message"]
            assert "entry_id" in invalid_job["message"]

            duplicate_upload = client.post(
                "/api/v1/documents/upload",
                headers=admin_headers,
                data={"chunk_strategy": "agent"},
                files={"file": ("duplicate-entry.md", _PILOT_ENTRY, "text/markdown")},
            )
            duplicate_job = _poll_job(client, admin_headers, _extract_data(duplicate_upload.json())["job_id"])
            assert duplicate_job["status"] == "failed"
            assert "AGENT_ENTRY_ID_CONFLICT" in duplicate_job["message"]
    finally:
        if previous_llm is None:
            registry.llm_providers.pop(settings.rag_primary_llm_provider, None)
        else:
            registry.register_llm(settings.rag_primary_llm_provider, previous_llm)
        app.dependency_overrides.clear()
        get_extension_registry.cache_clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)
