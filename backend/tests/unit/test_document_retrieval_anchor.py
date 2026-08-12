from app.service.document_retrieval_service import MixedModeDocumentRetrieverService


def test_complete_chinese_term_can_anchor_answer_evidence() -> None:
    retriever = object.__new__(MixedModeDocumentRetrieverService)

    assert retriever._has_lexical_anchor(
        "请说明星河验收的作用",
        "已发布资料记录了星河验收的作用。",
    )


def test_multiple_specific_terms_can_anchor_a_semantic_decision_query() -> None:
    retriever = object.__new__(MixedModeDocumentRetrieverService)

    assert retriever._has_lexical_anchor(
        "已知所有执行分支时应该用 workflow 还是 Agent？",
        "若所有正常分支能在设计时枚举，使用 workflow；否则保留最小 Agent loop。",
    )


def test_generic_shared_tokens_cannot_anchor_answer_evidence() -> None:
    retriever = object.__new__(MixedModeDocumentRetrieverService)

    assert not retriever._has_lexical_anchor(
        "what evidence supports the warranty period",
        "alpha evidence from the dense corpus",
    )
    assert not retriever._has_lexical_anchor(
        "请说明系统中的星河验收作用",
        "系统会定期备份日志。",
    )
