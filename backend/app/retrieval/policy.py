from __future__ import annotations

from dataclasses import dataclass

from app.common.config import Settings

PILOT_RETRIEVAL_PROFILE_ID = "retrieval-answer-policy/pilot-v1"
LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID = "retrieval-answer-policy/lexical-heuristic-migration-v1"


@dataclass(frozen=True)
class RetrievalPolicy:
    identity: str
    strategy: str
    engine: str
    tokenization: str
    bm25_k1: float | None
    bm25_b: float | None
    candidate_depth: int
    tie_breaker: str
    evidence_max_items: int
    evidence_max_chars_per_snapshot: int
    evidence_max_total_chars: int
    exact_deduplication: bool
    entry_section_deduplication: bool
    reranker_enabled: bool
    lexical_anchor_hard_gate_enabled: bool
    semantic_near_deduplication_enabled: bool
    query_expansion_enabled: bool
    online_llm_sufficiency_judge_enabled: bool
    diagnostic_or_migration_only: bool
    field_boosts: tuple[str, ...] = ()


_PILOT_V1 = RetrievalPolicy(
    identity=PILOT_RETRIEVAL_PROFILE_ID,
    strategy="sparse_bm25",
    engine="sparse_bm25",
    tokenization="literal_preserving/v1",
    bm25_k1=1.5,
    bm25_b=0.75,
    candidate_depth=20,
    tie_breaker="retrieval-candidate-tie-breaker/v1",
    evidence_max_items=3,
    evidence_max_chars_per_snapshot=1200,
    evidence_max_total_chars=3000,
    exact_deduplication=True,
    entry_section_deduplication=True,
    reranker_enabled=False,
    lexical_anchor_hard_gate_enabled=False,
    semantic_near_deduplication_enabled=False,
    query_expansion_enabled=False,
    online_llm_sufficiency_judge_enabled=False,
    diagnostic_or_migration_only=False,
)

_LEXICAL_HEURISTIC_MIGRATION_V1 = RetrievalPolicy(
    identity=LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID,
    strategy="lexical_heuristic_migration",
    engine="lexical_heuristic",
    tokenization="legacy_overlap/v1",
    bm25_k1=None,
    bm25_b=None,
    candidate_depth=0,
    tie_breaker="legacy-retrieval-order/v1",
    evidence_max_items=0,
    evidence_max_chars_per_snapshot=0,
    evidence_max_total_chars=0,
    exact_deduplication=False,
    entry_section_deduplication=False,
    reranker_enabled=True,
    lexical_anchor_hard_gate_enabled=True,
    semantic_near_deduplication_enabled=False,
    query_expansion_enabled=False,
    online_llm_sufficiency_judge_enabled=True,
    diagnostic_or_migration_only=True,
)


def get_retrieval_policy(settings: Settings) -> RetrievalPolicy:
    if settings.runtime_retrieval_profile == PILOT_RETRIEVAL_PROFILE_ID:
        return _PILOT_V1
    if settings.runtime_retrieval_profile == LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID:
        return _LEXICAL_HEURISTIC_MIGRATION_V1
    raise ValueError("runtime retrieval profile is not accepted")
