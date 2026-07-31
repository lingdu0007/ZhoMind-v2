from __future__ import annotations

import argparse
import asyncio

import uvicorn

from app.extensions.registry import get_extension_registry
from app.infra.db import SessionLocal, engine
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.document import Document, DocumentChunk, DocumentJob
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
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        del prompt, system_prompt
        return "部署前需要完成变更审批。"


async def _create_schema() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def _seed_test_data() -> None:
    ready_documents = [
        ("browser-evidence", "browser-evidence.md", "部署前需要完成变更审批。"),
        ("browser-inspection", "browser-inspection.md", "已发布分块可用于检查部署审批记录。"),
        ("browser-batch-first", "browser-batch-first.md", "第一份批量构建文档。"),
        ("browser-batch-second", "browser-batch-second.md", "第二份批量构建文档。"),
    ]
    async with SessionLocal() as session:
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
        await session.commit()

        await SystemSettingsDraftService(session).save(
            actor="browser-bootstrap",
            payload={
                "model_provider": "ark",
                "llm_model": "Qwen/Qwen3-32B",
                "embedding_model": "BAAI/bge-m3",
                "retrieval_strategy": "migration",
                "retrieval_top_k": 8,
                "score_threshold": 0.3,
                "milvus_uri": "http://milvus.internal:19530",
                "index_name": "zhomind_docs",
                "runtime_timeout_ms": 8000,
                "provider_api_key": None,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated browser acceptance API environment.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(_create_schema())
    asyncio.run(_seed_test_data())
    redis = _InMemoryRedis()
    app.dependency_overrides[get_redis_client] = lambda: redis
    registry = get_extension_registry()
    registry.register_llm("browser-acceptance", _DeterministicLlm())
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
