from app.service.document_retrieval_service import MixedModeDocumentRetrieverService


def test_complete_chinese_term_can_anchor_answer_evidence() -> None:
    retriever = object.__new__(MixedModeDocumentRetrieverService)

    assert retriever._has_lexical_anchor(
        "请说明星河验收的作用",
        "已发布资料记录了星河验收的作用。",
    )
