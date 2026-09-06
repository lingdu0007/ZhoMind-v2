import pytest
from pydantic import ValidationError

from app.common.config import Settings
from app.retrieval.policy import (
    LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID,
    PILOT_RETRIEVAL_PROFILE_ID,
    get_retrieval_policy,
)
from app.retrieval.sparse_bm25 import LiteralPreservingTokenizer


def test_pilot_v1_is_the_default_active_retrieval_policy() -> None:
    policy = get_retrieval_policy(Settings())

    assert policy.identity == PILOT_RETRIEVAL_PROFILE_ID
    assert policy.engine == "sparse_bm25"
    assert policy.tokenization == "literal_preserving/v1"
    assert policy.bm25_k1 == 1.5
    assert policy.bm25_b == 0.75
    assert policy.candidate_depth == 20
    assert policy.tie_breaker == "retrieval-candidate-tie-breaker/v1"
    assert policy.evidence_max_items == 3
    assert policy.evidence_max_chars_per_snapshot == 1200
    assert policy.evidence_max_total_chars == 3000
    assert policy.exact_deduplication is True
    assert policy.entry_section_deduplication is True
    assert policy.reranker_enabled is False
    assert policy.lexical_anchor_hard_gate_enabled is False
    assert policy.semantic_near_deduplication_enabled is False
    assert policy.query_expansion_enabled is False
    assert policy.online_llm_sufficiency_judge_enabled is False
    assert policy.field_boosts == ()


def test_lexical_heuristic_is_explicitly_a_migration_profile_not_sparse_bm25() -> None:
    policy = get_retrieval_policy(
        Settings(RUNTIME_RETRIEVAL_PROFILE=LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID)
    )

    assert policy.identity == LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID
    assert policy.engine == "lexical_heuristic"
    assert policy.strategy == "lexical_heuristic_migration"
    assert policy.diagnostic_or_migration_only is True
    assert "bm25" not in policy.identity


def test_retrieval_policy_rejects_an_unaccepted_profile_identity() -> None:
    with pytest.raises(ValidationError):
        Settings(RUNTIME_RETRIEVAL_PROFILE="retrieval-answer-policy/unrecorded-field-boost-v1")


def test_retrieval_policy_rejects_unrecorded_field_boosts() -> None:
    with pytest.raises(ValidationError):
        Settings(RUNTIME_RETRIEVAL_FIELD_BOOSTS="title:2")


def test_literal_preserving_tokenization_keeps_literals_adjacent_to_chinese_prose() -> None:
    tokens = LiteralPreservingTokenizer().tokenize(
        "配置RUNTIME_RETRIEVAL_PROFILE后，检查/api/v1/retrieval和gpt-4.1-mini，再处理CANDIDATE_PREVIEW_NOT_READY。"
    )

    assert "RUNTIME_RETRIEVAL_PROFILE" in tokens
    assert "/api/v1/retrieval" in tokens
    assert "gpt-4.1-mini" in tokens
    assert "CANDIDATE_PREVIEW_NOT_READY" in tokens
