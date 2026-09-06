from app.common.config import Settings
from app.extensions.registry import ExtensionRegistry
from app.service import chat_service
from app.service.document_retrieval_service import MixedModeDocumentRetrieverService


class _Reranker:
    async def rerank(self, query: str, items: list[dict]) -> list[dict]:
        return items


class _Judge:
    async def judge(self, query: str, context: list[dict]) -> bool:
        return True


class _UntrustedRetriever:
    async def retrieve(self, query: str, top_k: int):
        raise AssertionError("Pilot retrieval must not use an extension retriever")


def test_pilot_profile_forces_the_authorized_retrieval_pool(monkeypatch) -> None:
    registry = ExtensionRegistry()
    registry.register_retriever(chat_service.CHAT_RETRIEVER_PROVIDER, _UntrustedRetriever())
    monkeypatch.setattr(chat_service, "get_extension_registry", lambda: registry)
    monkeypatch.setattr(chat_service, "get_runtime_settings", lambda: Settings())

    service = chat_service.ChatService(session=None)  # type: ignore[arg-type]
    retriever, retriever_name = service._resolve_retriever()

    assert isinstance(retriever, MixedModeDocumentRetrieverService)
    assert retriever_name == MixedModeDocumentRetrieverService.name


def test_pilot_profile_disables_registered_reranker_and_online_judge(monkeypatch) -> None:
    registry = ExtensionRegistry()
    registry.register_rerank(chat_service.CHAT_RERANK_PROVIDER, _Reranker())
    registry.register_judge(chat_service.CHAT_JUDGE_PROVIDER, _Judge())
    monkeypatch.setattr(chat_service, "get_extension_registry", lambda: registry)
    monkeypatch.setattr(chat_service, "get_runtime_settings", lambda: Settings())

    service = chat_service.ChatService(session=None)  # type: ignore[arg-type]
    reranker, reranker_name = service._resolve_reranker()
    judge, judge_name = service._resolve_judge()

    assert reranker is None
    assert reranker_name == "disabled-by-retrieval-answer-policy/pilot-v1"
    assert judge is None
    assert judge_name == "disabled-by-retrieval-answer-policy/pilot-v1"
