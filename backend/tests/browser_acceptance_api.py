from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime

import uvicorn
from sqlalchemy.ext.asyncio import AsyncSession

from app.extensions.registry import get_extension_registry
from app.infra.db import SessionLocal, engine
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.chat import ChatMessage, ChatSession
from app.model.document import Document, DocumentChunk, DocumentJob
from app.model.system_settings import SystemSettingsState
from app.settings import runtime as settings_runtime
from app.settings.runtime import SystemSettingsRuntime
from app.settings.service import SystemSettingsDraftService


class _InMemoryRedis:
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}
        self._values: dict[str, str] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._hashes[key] = {str(name): str(value) for name, value in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return seconds > 0 and (key in self._hashes or key in self._values)

    async def exists(self, key: str) -> int:
        return int(key in self._hashes or key in self._values)

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def set(self, key: str, value: str) -> None:
        self._values[key] = value

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            deleted += int(key in self._values or key in self._hashes)
            self._values.pop(key, None)
            self._hashes.pop(key, None)
        return deleted


class _DeterministicLlm:
    """Deterministic generation adapter for the disposable acceptance API.

    Environment-controlled behaviors keep browser acceptance journeys
    deterministic without mocking network traffic:
    - BROWSER_ACCEPTANCE_LLM_DELAY_MS: sleep before answering so in-flight
      streaming UI states stay observable.
    - BROWSER_ACCEPTANCE_FAIL_FIRST=1: raise on the first call so the
      fail-closed Generation Unavailable path is observable, then succeed on
      subsequent calls.
    """

    def __init__(self) -> None:
        self._delay_ms = int(os.getenv("BROWSER_ACCEPTANCE_LLM_DELAY_MS", "0") or "0")
        self._fail_first = os.getenv("BROWSER_ACCEPTANCE_FAIL_FIRST") == "1"
        self._calls = 0

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        del system_prompt
        self._calls += 1
        if self._fail_first and self._calls == 1:
            raise RuntimeError("browser acceptance first-call failure")
        if self._delay_ms > 0:
            await asyncio.sleep(self._delay_ms / 1000)
        envelope = json.loads(prompt)
        contract = envelope.get("response_contract")
        if isinstance(contract, dict):
            marker = f"[{contract['citation_markers'][0]}]"
            label = contract.get("required_label")
            heading = f"【{label}】\n\n" if label else ""
            return heading + "\n\n".join(
                f"## {section}\n已知路径应由 deterministic workflow 控制。{marker}"
                for section in contract["required_sections"]
            )
        return "部署前需要完成变更审批。"


class _DeterministicSettingsRuntime(SystemSettingsRuntime):
    async def validate(self, *, settings: dict, provider_api_key: str | None) -> None:
        self._candidate_settings(settings=settings, provider_api_key=provider_api_key)

    async def apply(self, *, version: int, settings: dict, provider_api_key: str | None) -> None:
        await super().apply(version=version, settings=settings, provider_api_key=provider_api_key)


async def _create_schema() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def _seed_chat_history(session: AsyncSession) -> None:
    """Seed deterministic historical conversations used by browser journeys.

    - `lin` (a Knowledge User created at runtime) gets one session whose
      assistant messages carry sufficient / unavailable / insufficient evidence
      summaries, exercising historical read projections.
    - `operator` (the Bootstrap Administrator) gets one session with a partial
      RAG trace so unavailable retrieval-diagnostics fields stay observable.
    """
    now = datetime.now(UTC)
    session.add(ChatSession(id="session-evidence-history", user_id="lin", created_at=now, updated_at=now))
    session.add(
        ChatMessage(
            id="msg-history-user",
            session_id="session-evidence-history",
            user_id="lin",
            type="user",
            content="历史的部署问题",
            created_at=now,
        )
    )
    session.add(
        ChatMessage(
            id="msg-history-sufficient",
            session_id="session-evidence-history",
            user_id="lin",
            type="assistant",
            content="历史回答有可核对来源。",
            rag_trace={
                "outcome": "evidence_gated_answer",
                "gate": {"passed": True, "reason": "sufficient_evidence"},
                "evidence": [
                    {
                        "chunk_id": "chunk-history-7",
                        "metadata": {"filename": "deploy-runbook.md"},
                        "content_preview": "历史来源摘录。",
                    }
                ],
            },
            created_at=now,
        )
    )
    session.add(
        ChatMessage(
            id="msg-history-unavailable",
            session_id="session-evidence-history",
            user_id="lin",
            type="assistant",
            content="历史回答没有可用来源。",
            rag_trace=None,
            created_at=now,
        )
    )
    session.add(
        ChatMessage(
            id="msg-history-insufficient",
            session_id="session-evidence-history",
            user_id="lin",
            type="assistant",
            content="历史回答证据不足。",
            rag_trace={
                "outcome": "insufficient_evidence_reply",
                "gate": {"passed": False, "reason": "reject_insufficient_evidence"},
                "evidence": [],
            },
            created_at=now,
        )
    )
    session.add(ChatSession(id="session-admin-diagnostics", user_id="operator", created_at=now, updated_at=now))
    session.add(
        ChatMessage(
            id="msg-admin-diagnostics",
            session_id="session-admin-diagnostics",
            user_id="operator",
            type="assistant",
            content="这是一条缺少部分诊断字段的历史回答。",
            rag_trace={},
            created_at=now,
        )
    )


async def _seed_test_data() -> None:
    ready_documents = [
        ("browser-evidence", "browser-evidence.md", "部署前需要完成变更审批。"),
        ("browser-inspection", "browser-inspection.md", "已发布分块可用于检查部署审批记录。"),
        ("browser-single-delete", "browser-single-delete.md", "单个删除验收文档。"),
        ("browser-batch-first", "browser-batch-first.md", "第一份批量构建文档。"),
        ("browser-batch-second", "browser-batch-second.md", "第二份批量构建文档。"),
        ("browser-batch-partial-first", "browser-batch-partial-first.md", "批量删除部分失败的并发文档。"),
        ("browser-batch-partial-second", "browser-batch-partial-second.md", "批量删除成功文档。"),
    ]
    async with SessionLocal() as session:
        if os.getenv("BROWSER_ACCEPTANCE_SEED") != "minimal":
            for document_id, filename, content in ready_documents:
                session.add(
                    Document(
                        id=document_id,
                        filename=filename,
                        file_type="md",
                        file_size=len(content.encode("utf-8")),
                        source_content=content.encode("utf-8"),
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        next_generation=2,
                        latest_requested_generation=1,
                    )
                )
                session.add(
                    DocumentChunk(
                        id=f"chunk-{document_id}",
                        document_id=document_id,
                        generation=1,
                        chunk_index=0,
                        content=content,
                        keywords=["部署", "审批"],
                        generated_questions=[],
                        chunk_metadata={"filename": filename},
                    )
                )
                session.add(
                    DocumentJob(
                        id=f"job-{document_id}",
                        document_id=document_id,
                        build_generation=1,
                        requested_chunk_strategy="general",
                        status="succeeded",
                        stage="completed",
                        progress=100,
                        message="document build completed",
                    )
                )

            session.add(
                Document(
                    id="browser-agent-entry",
                    filename="synthetic-agent-entry.md",
                    file_type="md",
                    file_size=len("已知路径应由 deterministic workflow 控制。".encode()),
                    source_content=b"synthetic browser acceptance entry",
                    status="ready",
                    chunk_strategy="agent",
                    chunk_count=1,
                    published_generation=1,
                    next_generation=2,
                    latest_requested_generation=1,
                )
            )
            session.add(
                DocumentChunk(
                    id="chunk-browser-agent-entry",
                    document_id="browser-agent-entry",
                    generation=1,
                    chunk_index=0,
                    content="已知路径应由 deterministic workflow 控制。",
                    keywords=["deterministic", "workflow"],
                    generated_questions=[],
                    chunk_metadata={
                        "strategy": "agent",
                        "entry_id": "synthetic-workflow-001",
                        "entry_title": "Prefer deterministic workflows",
                        "domain": "workflow-vs-agent",
                        "section_id": "stable-principle",
                        "review_status": "approved",
                        "review_date": "2026-08-12",
                        "applicable_versions": ["framework-neutral", "Anthropic 2024-12-19"],
                        "approved_summary": "已知路径应由 deterministic workflow 控制。",
                        "suggested_query": "什么时候使用 deterministic workflow？",
                        "sources": [
                            {
                                "title": "Building effective agents",
                                "authority": "Anthropic",
                                "url": "https://www.anthropic.com/engineering/building-effective-agents",
                                "version": "2024-12-19",
                                "availability": "verified",
                            }
                        ],
                        "source_title": "Building effective agents",
                        "source_authority": "Anthropic",
                        "source_url": "https://www.anthropic.com/engineering/building-effective-agents",
                        "source_version": "2024-12-19",
                        "source_availability": "verified",
                    },
                )
            )
            session.add(
                DocumentJob(
                    id="job-browser-agent-entry",
                    document_id="browser-agent-entry",
                    build_generation=1,
                    requested_chunk_strategy="agent",
                    status="succeeded",
                    stage="completed",
                    progress=100,
                    message="synthetic Agent entry published",
                )
            )
            session.add(
                Document(
                    id="browser-cancelable",
                    filename="browser-cancelable.md",
                    file_type="md",
                    file_size=0,
                    status="pending",
                    chunk_strategy="general",
                    next_generation=2,
                    latest_requested_generation=1,
                )
            )
            session.add(
                DocumentJob(
                    id="job-browser-cancelable",
                    document_id="browser-cancelable",
                    build_generation=1,
                    requested_chunk_strategy="general",
                    status="queued",
                    stage="queued",
                    progress=0,
                    message="queued for browser cancellation",
                )
            )

            session.add(
                Document(
                    id="browser-running",
                    filename="browser-running.md",
                    file_type="md",
                    file_size=len("正在生成可检索分块。".encode()),
                    source_content="正在生成可检索分块。".encode(),
                    status="processing",
                    chunk_strategy="general",
                    chunk_count=0,
                    next_generation=3,
                    latest_requested_generation=2,
                    active_build_generation=2,
                    active_build_job_id="job-browser-running",
                    active_build_heartbeat_at=datetime.now(UTC),
                )
            )
            session.add(
                DocumentJob(
                    id="job-browser-running",
                    document_id="browser-running",
                    build_generation=2,
                    requested_chunk_strategy="general",
                    status="running",
                    stage="chunking",
                    progress=56,
                    message="正在生成可检索分块",
                )
            )

            session.add(
                Document(
                    id="browser-candidate",
                    filename="browser-candidate.md",
                    file_type="md",
                    file_size=len("候选构建等待管理员发布。".encode()),
                    source_content="候选构建等待管理员发布。".encode(),
                    status="candidate",
                    chunk_strategy="general",
                    chunk_count=0,
                    published_generation=0,
                    candidate_generation=2,
                    candidate_chunk_strategy="general",
                    candidate_chunk_count=1,
                    next_generation=3,
                    latest_requested_generation=2,
                )
            )
            session.add(
                DocumentChunk(
                    id="chunk-browser-candidate",
                    document_id="browser-candidate",
                    generation=2,
                    chunk_index=0,
                    content="候选构建等待管理员发布。",
                    keywords=["候选", "发布"],
                    generated_questions=[],
                    chunk_metadata={"filename": "browser-candidate.md"},
                )
            )
            session.add(
                DocumentJob(
                    id="job-browser-candidate",
                    document_id="browser-candidate",
                    build_generation=2,
                    requested_chunk_strategy="general",
                    status="succeeded",
                    stage="completed",
                    progress=100,
                    message="candidate build completed; awaiting publication",
                )
            )

            await _seed_chat_history(session)
        await session.commit()

        await SystemSettingsDraftService(session).save(
            actor="browser-bootstrap",
            payload={
                "provider_type": "ark",
                "model": "Qwen/Qwen3-32B",
                "service_url": "https://provider.example.test/v1",
                "provider_api_key": "browser-acceptance-placeholder",
            },
        )
        if os.getenv("BROWSER_ACCEPTANCE_SETTINGS_FAILED") == "1":
            await session.merge(
                SystemSettingsState(
                    id=1,
                    latest_saved_version=1,
                    active_version=None,
                    application_state="failed",
                    application_version=None,
                    application_actor=None,
                    application_at=None,
                    application_message="runtime did not accept the saved configuration",
                )
            )
        await session.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated browser acceptance API environment.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings_runtime._runtime = _DeterministicSettingsRuntime()
    asyncio.run(_create_schema())
    asyncio.run(_seed_test_data())
    redis = _InMemoryRedis()
    app.dependency_overrides[get_redis_client] = lambda: redis
    registry = get_extension_registry()
    registry.register_llm("browser-acceptance", _DeterministicLlm())
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
