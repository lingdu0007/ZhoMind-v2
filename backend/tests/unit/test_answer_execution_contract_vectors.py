import json
from pathlib import Path

import pytest

from app.contracts.canonical import AnswerExecutionState, AnswerOutcome
from app.rag.evidence_sufficiency import InsufficientEvidenceReply, QueryConditionSet

_VECTORS_PATH = Path(__file__).parents[3] / "docs" / "contracts" / "answer-execution-contract-vectors.json"


def _vectors() -> dict:
    return json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))


def test_answer_execution_contract_vectors_cover_python_qcs_reason_and_state_seams() -> None:
    vectors = _vectors()
    assert vectors["schema"] == "answer_execution_contract_vectors/v1"

    for vector in vectors["question_condition_vectors"]:
        assert QueryConditionSet.from_question(vector["question"]).to_record()["conditions"] == vector["conditions"]

    for vector in vectors["invalid_question_condition_vectors"]:
        with pytest.raises(ValueError):
            QueryConditionSet.from_question(vector["question"])

    unicode_vector = vectors["unicode_code_point_condition_vector"]
    unicode_condition = unicode_vector["condition"]
    assert QueryConditionSet.from_records(
        normalized_question=unicode_vector["question"],
        records=[
            {
                "condition_id": unicode_condition["condition_id"],
                "field": unicode_condition["field"],
                "operator": unicode_condition["operator"],
                "value": unicode_condition["value_character"] * unicode_condition["value_code_points"],
            }
        ],
    ).to_record()["conditions"][0]["value"] == (
        unicode_condition["value_character"] * unicode_condition["value_code_points"]
    )

    for vector in vectors["insufficient_evidence_presentations"]:
        assert (
            InsufficientEvidenceReply(
                reason=vector["reason"],
                query_condition_set_identity="qcs-contract-vector",
            ).reason
            == vector["reason"]
        )

    for vector in vectors["execution_presentations"]:
        assert AnswerExecutionState(vector["state"]).value == vector["state"]
        if vector["state"] == "completed":
            assert AnswerOutcome(vector["outcome"]).value == vector["outcome"]
