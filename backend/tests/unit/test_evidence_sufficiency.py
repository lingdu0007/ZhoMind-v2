import hashlib
import json

import pytest

from app.rag.evidence_sufficiency import QueryConditionSet, decide_answer_evidence


def _candidate(
    *,
    entry_id: str,
    section_id: str,
    chunk_id: str,
    content: str,
    score: float,
    evidence_conflict: str = "none",
    assurance_level: str = "source_grounded",
    decision_query: str = "Which reviewed operating decision applies for environment=production?",
    source_id: str | None = None,
    content_length: int | None = None,
) -> dict:
    source_id = source_id or f"{entry_id}-source"
    content_sha256 = hashlib.sha256(f"{chunk_id}-content".encode()).hexdigest()
    return {
        "chunk_id": chunk_id,
        "document_id": f"{entry_id}-document",
        "generation": 1,
        "chunk_index": 0,
        "content_preview": content,
        "content_length": len(content) if content_length is None else content_length,
        "content_sha256": content_sha256,
        "score": score,
        "answer_evidence_eligible": True,
        "entry_id": entry_id,
        "entry_identity": f"entry:{entry_id}",
        "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
        "publication_identity": f"published_knowledge_version:{entry_id}.v1",
        "publication_version": "v1",
        "section_id": section_id,
        "section_identity": f"entry:{entry_id}#{section_id}",
        "decision_query": decision_query,
        "chunk_identity": {
            "document_id": f"{entry_id}-document",
            "generation": 1,
            "chunk_index": 0,
            "content_sha256": content_sha256,
        },
        "source_relationships": [
            {
                "source_identity": f"source:{source_id}",
                "availability": "verified_usable",
                "access_scope": "controlled_internal",
            }
        ],
        "assurance_level": assurance_level,
        "applicability_conditions": [
            {
                "condition_id": "environment-production",
                "field": "environment",
                "operator": "equals",
                "value": "production",
            }
        ],
        "freshness_triggers": [{"trigger_id": "source-change"}],
        "lifecycle_state": "published",
        "metadata": {
            "title": "Use deterministic evidence decisions",
            "publication_version": "v1",
            "entry_id": entry_id,
            "entry_identity": f"entry:{entry_id}",
            "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
            "entry_title": "Use deterministic evidence decisions",
            "domain": "evidence-sufficiency",
            "section_id": section_id,
            "source_id": source_id,
            "source_title": "Reviewed controlled source",
            "source_authority": "ZhoMind editorial authority",
            "source_url": f"controlled://knowledge/{source_id}",
            "source_version": "2026-09-06",
            "review_date": "2026-09-06",
            "review_status": "approved",
            "source_availability": "verified",
            "source_access_scope": "controlled_internal",
            "source_tier": "primary_evidence_source",
            "evidence_conflict": evidence_conflict,
            "decision_query": decision_query,
        },
    }


def test_sufficiency_selects_the_applicable_governing_section_not_the_first_candidate() -> None:
    decision = decide_answer_evidence(
        normalized_question="What is the reviewed default for environment=production?",
        query_conditions=QueryConditionSet.from_records(
            normalized_question="What is the reviewed default for environment=production?",
            records=[
                {
                    "condition_id": "environment-production",
                    "field": "environment",
                    "operator": "equals",
                    "value": "production",
                }
            ],
        ),
        candidates=[
            _candidate(
                entry_id="decision-001",
                section_id="alternatives",
                chunk_id="alternative-first-by-score",
                content="Alternatives remain available when the reviewed default does not apply.",
                score=99.0,
            ),
            _candidate(
                entry_id="decision-001",
                section_id="recommendation_or_reviewed_branches",
                chunk_id="governing-second-by-score",
                content="For production, use the deterministic reviewed default.",
                score=0.1,
            ),
        ],
    )

    assert decision.is_sufficient is True
    assert decision.insufficient_reply is None
    assert decision.evidence_set is not None
    assert [item.section_id for item in decision.evidence_set.items] == ["recommendation_or_reviewed_branches"]
    assert decision.evidence_set.governing_citation.section_id == "recommendation_or_reviewed_branches"
    assert decision.evidence_set.items[0].excerpt == "For production, use the deterministic reviewed default."


def test_sufficiency_requires_each_decisive_reviewed_condition_to_be_visible() -> None:
    decision = decide_answer_evidence(
        normalized_question="What is the reviewed default?",
        query_conditions=QueryConditionSet.from_records(
            normalized_question="What is the reviewed default?",
            records=[],
        ),
        candidates=[
            _candidate(
                entry_id="decision-001",
                section_id="recommendation_or_reviewed_branches",
                chunk_id="governing-with-missing-condition",
                content="For production, use the deterministic reviewed default.",
                score=99.0,
            )
        ],
    )

    assert decision.is_sufficient is False
    assert decision.evidence_set is None
    assert decision.insufficient_reply is not None
    assert decision.insufficient_reply.reason == "decisive_condition_missing"


def test_sufficiency_refuses_an_unresolved_material_conflict_before_selection() -> None:
    question = "What is the reviewed default for environment=production?"
    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_records(
            normalized_question=question,
            records=[
                {
                    "condition_id": "environment-production",
                    "field": "environment",
                    "operator": "equals",
                    "value": "production",
                }
            ],
        ),
        candidates=[
            _candidate(
                entry_id="decision-001",
                section_id="recommendation_or_reviewed_branches",
                chunk_id="conflicted-governing-candidate",
                content="Conflicting evidence remains unresolved.",
                score=99.0,
                evidence_conflict="unresolved",
            )
        ],
    )

    assert decision.is_sufficient is False
    assert decision.evidence_set is None
    assert decision.insufficient_reply is not None
    assert decision.insufficient_reply.reason == "material_evidence_conflict"


def test_claim_linked_governing_evidence_requires_a_matching_reviewed_contract() -> None:
    question = "What is the reviewed default for environment=production?"
    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_records(
            normalized_question=question,
            records=[
                {
                    "condition_id": "environment-production",
                    "field": "environment",
                    "operator": "equals",
                    "value": "production",
                }
            ],
        ),
        candidates=[
            _candidate(
                entry_id="decision-claim-linked-001",
                section_id="recommendation_or_reviewed_branches",
                chunk_id="claim-linked-without-contract",
                content="A high-impact reviewed recommendation needs its Claim-Evidence Link.",
                score=4.0,
                assurance_level="claim_linked",
            )
        ],
    )

    assert decision.is_sufficient is False
    assert decision.evidence_set is None
    assert decision.insufficient_reply is not None
    assert decision.insufficient_reply.reason == "assurance_support_missing"


def test_comparison_selects_only_the_governing_section_and_required_complements() -> None:
    question = "Compare the reviewed options for environment=production."
    conditions = QueryConditionSet.from_records(
        normalized_question=question,
        records=[
            {
                "condition_id": "environment-production",
                "field": "environment",
                "operator": "equals",
                "value": "production",
            }
        ],
    )
    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=conditions,
        candidates=[
            _candidate(
                entry_id="decision-001",
                section_id="minimum_validation_guidance",
                chunk_id="unneeded-first",
                content="This validation section is not required for a comparison.",
                score=100.0,
            ),
            _candidate(
                entry_id="decision-001",
                section_id="alternatives",
                chunk_id="alternative-second",
                content="The alternative uses a narrower but slower operating model.",
                score=3.0,
            ),
            _candidate(
                entry_id="decision-001",
                section_id="trade_offs",
                chunk_id="trade-off-third",
                content="The trade-off is predictable control versus local flexibility.",
                score=2.0,
            ),
            _candidate(
                entry_id="decision-001",
                section_id="recommendation_or_reviewed_branches",
                chunk_id="governing-last",
                content="For production, use the deterministic reviewed default.",
                score=0.1,
            ),
        ],
    )

    assert decision.is_sufficient is True
    assert decision.evidence_set is not None
    assert [item.section_id for item in decision.evidence_set.items] == [
        "recommendation_or_reviewed_branches",
        "alternatives",
        "trade_offs",
    ]
    assert [citation.marker for citation in decision.evidence_set.citations] == ["S1", "S2", "S3"]


def test_frozen_item_and_citation_identities_exclude_raw_scores() -> None:
    question = "What is the reviewed default for environment=production?"
    conditions = QueryConditionSet.from_records(
        normalized_question=question,
        records=[
            {
                "condition_id": "environment-production",
                "field": "environment",
                "operator": "equals",
                "value": "production",
            }
        ],
    )
    candidate = _candidate(
        entry_id="decision-identity-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="stable-governing-item",
        content="Freeze this exact normalized Evidence Excerpt Snapshot.",
        score=0.1,
    )
    first = decide_answer_evidence(
        normalized_question=question,
        query_conditions=conditions,
        candidates=[candidate],
    )
    candidate["score"] = 999.0
    second = decide_answer_evidence(
        normalized_question=question,
        query_conditions=conditions,
        candidates=[candidate],
    )

    assert first.evidence_set is not None
    assert second.evidence_set is not None
    assert first.evidence_set.identity == second.evidence_set.identity
    assert first.evidence_set.citations[0].item_identity == second.evidence_set.citations[0].item_identity
    record = first.evidence_set.to_record()
    assert record["items"][0]["item_identity"] == first.evidence_set.citations[0].item_identity
    assert "score" not in str(record)


def test_sufficiency_returns_no_eligible_published_evidence_when_the_pool_has_no_authorized_candidate() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-no-eligible-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="not-authorized",
        content="This candidate did not pass the authorized pool boundary.",
        score=100.0,
    )
    candidate["answer_evidence_eligible"] = False

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.reason == "no_eligible_published_evidence"
    assert decision.evidence_set is None


def test_sufficiency_distinguishes_an_explicit_nonmatching_condition_from_a_missing_condition() -> None:
    question = "What is the reviewed default for environment=staging?"
    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[
            _candidate(
                entry_id="decision-not-covered-001",
                section_id="recommendation_or_reviewed_branches",
                chunk_id="production-only",
                content="For production, use the deterministic reviewed default.",
                score=1.0,
            )
        ],
    )

    assert decision.reason == "decision_not_covered"
    assert decision.evidence_set is None


def test_sufficiency_returns_knowledge_needs_review_without_selecting_a_snapshot() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-needs-review-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="needs-review",
        content="This entry must be reviewed again before it can support an answer.",
        score=1.0,
    )
    candidate["lifecycle_state"] = "needs_re_review"

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.reason == "knowledge_needs_review"
    assert decision.evidence_set is None


def test_sufficiency_never_silently_drops_required_support_that_exceeds_the_budget() -> None:
    question = "Compare the reviewed options for environment=production."
    content = "x" * 1001
    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[
            _candidate(
                entry_id="decision-budget-001",
                section_id="recommendation_or_reviewed_branches",
                chunk_id="governing",
                content=content,
                score=100.0,
            ),
            _candidate(
                entry_id="decision-budget-001",
                section_id="alternatives",
                chunk_id="alternatives",
                content=content,
                score=99.0,
            ),
            _candidate(
                entry_id="decision-budget-001",
                section_id="trade_offs",
                chunk_id="trade-offs",
                content=content,
                score=98.0,
            ),
        ],
    )

    assert decision.reason == "evidence_budget_exceeded"
    assert decision.evidence_set is None


@pytest.mark.parametrize(
    ("question", "required_sections"),
    [
        (
            "Diagnose the reviewed failure for environment=production.",
            ("failure_modes", "minimum_diagnosis_guidance"),
        ),
        (
            "Perform an acceptance review for environment=production.",
            ("minimum_acceptance_guidance",),
        ),
    ],
)
def test_sufficiency_adds_only_the_required_complements_for_the_question_intent(
    question: str,
    required_sections: tuple[str, ...],
) -> None:
    candidates = [
        _candidate(
            entry_id="decision-intent-001",
            section_id="recommendation_or_reviewed_branches",
            chunk_id="governing",
            content="Use the reviewed governing branch for this condition.",
            score=1.0,
        )
    ]
    candidates.extend(
        _candidate(
            entry_id="decision-intent-001",
            section_id=section_id,
            chunk_id=section_id,
            content=f"Reviewed support for {section_id}.",
            score=100.0,
        )
        for section_id in required_sections
    )
    candidates.append(
        _candidate(
            entry_id="decision-intent-001",
            section_id="minimum_validation_guidance",
            chunk_id="unneeded",
            content="This section is not selected for this intent.",
            score=999.0,
        )
    )

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=candidates,
    )

    assert decision.is_sufficient is True
    assert decision.evidence_set is not None
    assert [item.section_id for item in decision.evidence_set.items] == [
        "recommendation_or_reviewed_branches",
        *required_sections,
    ]


def test_frozen_item_identity_binds_the_selected_editorial_and_chunk_identities() -> None:
    question = "What is the reviewed default for environment=production?"
    original = _candidate(
        entry_id="decision-identity-material-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="identity-material",
        content="Freeze the exact source and editorial identities.",
        score=1.0,
    )
    revised = json.loads(json.dumps(original))
    revised["editorial_revision_identity"] = "editorial_revision:decision-identity-material-001.r2"
    revised["metadata"]["editorial_revision_identity"] = revised["editorial_revision_identity"]
    revised_hash = hashlib.sha256(b"changed-content").hexdigest()
    revised["content_sha256"] = revised_hash
    revised["chunk_identity"]["content_sha256"] = revised_hash

    first = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[original],
    )
    second = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[revised],
    )

    assert first.evidence_set is not None
    assert second.evidence_set is not None
    assert first.evidence_set.citations[0].item_identity != second.evidence_set.citations[0].item_identity
    assert first.evidence_set.identity != second.evidence_set.identity


def test_sufficiency_rejects_a_forged_or_mismatched_content_hash() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-forged-hash-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="forged-hash",
        content="The content hash must be exact before it can freeze.",
        score=1.0,
    )
    candidate["content_sha256"] = hashlib.sha256(b"different-content").hexdigest()

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is False
    assert decision.reason == "no_eligible_published_evidence"


def test_sufficiency_rejects_citation_metadata_from_a_different_entry() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-citation-binding-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="citation-binding",
        content="The frozen item and citation must name the same entry.",
        score=1.0,
    )
    candidate["metadata"]["entry_id"] = "forged-citation-entry"

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is False
    assert decision.reason == "no_eligible_published_evidence"


def test_sufficiency_rejects_an_unknown_source_tier() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-source-tier-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="unknown-source-tier",
        content="The evidence source tier must remain a closed authority value.",
        score=1.0,
    )
    candidate["metadata"]["source_tier"] = "unclassified_external_material"

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is False
    assert decision.reason == "no_eligible_published_evidence"


def test_sufficiency_rejects_a_malformed_source_identity() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-source-identity-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="malformed-source-identity",
        content="The source identity must remain canonical and immutable.",
        score=1.0,
    )
    candidate["source_relationships"][0]["source_identity"] = "source:"

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is False
    assert decision.reason == "no_eligible_published_evidence"


@pytest.mark.parametrize(
    "event_id,expected_sufficient",
    [
        ("event:ticket19-acceptance-active-v1", True),
        ("0123456789abcdef0123456789abcdef", True),
        ("unqualified-event", False),
        ("member:0123456789abcdef0123456789abcdef", False),
        ("g" * 32, False),
    ],
)
def test_release_assured_evidence_requires_and_accepts_its_frozen_assurance_snapshot(
    event_id: str, expected_sufficient: bool,
) -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-release-assured-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="release-assured",
        content="This recommendation is backed by its frozen release assurance.",
        score=1.0,
        assurance_level="release_assured",
    )
    candidate["metadata"]["release_assurance_snapshot"] = {
        "schema": "editorial_release_assurance_snapshot/v1",
        "entry_identity": candidate["entry_identity"],
        "records": [
            {
                "field": "contract_identity",
                "identity": "product_path:ticket19-contract-v1",
                "record_class": "authoritative",
                "payload_sha256": "a" * 64,
            },
            {
                "field": "calibration_identity",
                "identity": "configuration:ticket19-calibration-v1",
                "record_class": "immutable",
                "payload_sha256": "b" * 64,
            },
            {
                "field": "frozen_acceptance_identity",
                "identity": "delivery_acceptance_record:ticket19-acceptance-v1",
                "record_class": "immutable",
                "payload_sha256": "c" * 64,
            },
            {
                "field": "named_gate",
                "identity": "capability:ticket19-gate-v1",
                "record_class": "authoritative",
                "payload_sha256": "d" * 64,
            },
        ],
        "frozen_acceptance_status": {
            "event_id": event_id,
            "event_sha256": "e" * 64,
            "to_state": "active",
        },
    }

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is expected_sufficient
    if expected_sufficient:
        assert decision.evidence_set is not None
        assert len(decision.evidence_set.items[0].excerpt) <= 1200
    else:
        assert decision.reason == "assurance_support_missing"
        assert decision.evidence_set is None


def test_claim_linked_evidence_accepts_a_matching_reviewed_contract() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-claim-contract-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="claim-contract",
        content="A reviewed claim link supports this recommendation.",
        score=1.0,
        assurance_level="claim_linked",
    )
    contract = {
        "schema_version": 1,
        "review_id": "ticket19-claim-review",
        "review_revision": "2026-09-06.1",
        "conflict_state": "none",
        "unknown_state": "none",
        "resolver": {
            "resolver_id": "ticket19-claim-resolver-v1",
            "calibration_id": "ticket19-calibration-v1",
            "calibration_version": "2026-09-06",
            "minimum_confidence": 0.8,
        },
        "claims": [
            {
                "claim_id": "ticket19-governing-claim",
                "scope": "The reviewed governing recommendation for the explicit production condition.",
                "evidence": [
                    {
                        "section_id": "recommendation_or_reviewed_branches",
                        "source_id": candidate["metadata"]["source_id"],
                    }
                ],
            }
        ],
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    candidate["metadata"]["claim_evidence_contract"] = canonical
    candidate["metadata"]["claim_evidence_contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is True
    assert decision.evidence_set is not None


def test_candidate_claim_linked_evidence_requires_every_material_claim_link_in_the_same_section() -> None:
    question = "What is the reviewed default for environment=production?"
    first = _candidate(
        entry_id="decision-candidate-contract-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="candidate-claim-source-a",
        content="The governing recommendation requires independent reviewed support.",
        score=2.0,
        assurance_level="claim_linked",
        source_id="source-claim-a",
    )
    second = _candidate(
        entry_id="decision-candidate-contract-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="candidate-claim-source-b",
        content="The governing recommendation requires independent reviewed support.",
        score=1.0,
        assurance_level="claim_linked",
        source_id="source-claim-b",
    )
    contract = {
        "schema": "candidate_claim_evidence_contract/v1",
        "entry_identity": first["entry_identity"],
        "editorial_revision_identity": first["editorial_revision_identity"],
        "claims": [
            {
                "claim_id": "claim-source-a",
                "section_id": "recommendation_or_reviewed_branches",
                "source_ids": ["source-claim-a"],
            },
            {
                "claim_id": "claim-source-b",
                "section_id": "recommendation_or_reviewed_branches",
                "source_ids": ["source-claim-b"],
            },
        ],
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for candidate in (first, second):
        candidate["metadata"]["claim_evidence_contract"] = canonical
        candidate["metadata"]["claim_evidence_contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[first, second],
    )

    assert decision.is_sufficient is True
    assert decision.evidence_set is not None
    assert {
        dict(item.metadata_items)["source_id"]
        for item in decision.evidence_set.items
    } == {"source-claim-a", "source-claim-b"}


def test_claim_linked_evidence_rejects_malformed_contract_json_as_insufficient_support() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-malformed-claim-contract-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="malformed-claim-contract",
        content="A malformed Claim-Evidence contract must fail closed.",
        score=1.0,
        assurance_level="claim_linked",
    )
    candidate["metadata"]["claim_evidence_contract"] = "{not-json"
    candidate["metadata"]["claim_evidence_contract_sha256"] = "a" * 64

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is False
    assert decision.reason == "assurance_support_missing"


def test_sufficiency_refuses_a_governing_section_when_its_decision_query_does_not_cover_the_question() -> None:
    question = "Which vector database governs production deployment for environment=production?"
    candidate = _candidate(
        entry_id="decision-coverage-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="payroll-not-vector-search",
        content="Use the reviewed payroll approval branch.",
        score=1.0,
        decision_query="Which payroll approval route governs production payroll?",
    )

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is False
    assert decision.reason == "decision_not_covered"


def test_claim_linked_governing_evidence_requires_every_link_for_its_selected_claim() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-claim-completeness-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="claim-governing-only",
        content="The governing recommendation needs its reviewed validation support.",
        score=1.0,
        assurance_level="claim_linked",
    )
    source_id = candidate["metadata"]["source_id"]
    contract = {
        "schema_version": 1,
        "review_id": "ticket19-claim-completeness-review",
        "review_revision": "2026-09-06.2",
        "conflict_state": "none",
        "unknown_state": "none",
        "resolver": {
            "resolver_id": "ticket19-claim-resolver-v1",
            "calibration_id": "ticket19-calibration-v1",
            "calibration_version": "2026-09-06",
            "minimum_confidence": 0.8,
        },
        "claims": [
            {
                "claim_id": "ticket19-governing-and-validation",
                "scope": "The governing recommendation requires its validation evidence.",
                "evidence": [
                    {
                        "section_id": "recommendation_or_reviewed_branches",
                        "source_id": source_id,
                    },
                    {
                        "section_id": "minimum_validation_guidance",
                        "source_id": source_id,
                    },
                ],
            }
        ],
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    candidate["metadata"]["claim_evidence_contract"] = canonical
    candidate["metadata"]["claim_evidence_contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is False
    assert decision.reason == "assurance_support_missing"


def test_sufficiency_selects_the_same_smallest_viable_set_regardless_of_retrieval_order() -> None:
    question = "Compare the reviewed options for environment=production."
    candidates = [
        _candidate(
            entry_id="decision-canonical-selection-001",
            section_id="recommendation_or_reviewed_branches",
            chunk_id="governing-too-large-in-total",
            content="g" * 1200,
            score=999.0,
        ),
        _candidate(
            entry_id="decision-canonical-selection-001",
            section_id="recommendation_or_reviewed_branches",
            chunk_id="governing-smallest-viable",
            content="Use the deterministic reviewed default.",
            score=0.1,
        ),
        _candidate(
            entry_id="decision-canonical-selection-001",
            section_id="alternatives",
            chunk_id="alternatives-support",
            content="a" * 1200,
            score=100.0,
        ),
        _candidate(
            entry_id="decision-canonical-selection-001",
            section_id="trade_offs",
            chunk_id="tradeoffs-support",
            content="t" * 1200,
            score=99.0,
        ),
    ]

    first = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=candidates,
    )
    second = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=list(reversed(candidates)),
    )

    assert first.is_sufficient is True
    assert second.is_sufficient is True
    assert first.evidence_set is not None
    assert second.evidence_set is not None
    assert first.evidence_set.identity == second.evidence_set.identity
    assert [item.source_id for item in first.evidence_set.items] == [
        "governing-smallest-viable",
        "alternatives-support",
        "tradeoffs-support",
    ]


def test_sufficiency_rejects_an_upstream_truncated_candidate_instead_of_freezing_its_preview() -> None:
    question = "What is the reviewed default for environment=production?"
    candidate = _candidate(
        entry_id="decision-truncated-candidate-001",
        section_id="recommendation_or_reviewed_branches",
        chunk_id="truncated-upstream",
        content="x" * 1200,
        content_length=1201,
        score=1.0,
    )

    decision = decide_answer_evidence(
        normalized_question=question,
        query_conditions=QueryConditionSet.from_question(question),
        candidates=[candidate],
    )

    assert decision.is_sufficient is False
    assert decision.reason == "evidence_budget_exceeded"
