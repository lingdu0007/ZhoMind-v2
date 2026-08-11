import importlib.util
import re
from pathlib import Path

import pytest

from app.rag.prompt_injection_corpus import (
    ADVERSARIAL_INJECTION_CASES,
    CASE_KINDS,
    FORGED_SOURCE,
    INSTRUCTION_OVERRIDE,
    SECRET_EXTRACTION,
    UNSUPPORTED_ANSWER_PRESSURE,
)

# Import the real secret-scan patterns so the corpus guarantee never drifts
# from the actual scan the PR Gate runs.
_SCAN_SECRETS_PATH = Path(__file__).resolve().parents[3] / "scripts" / "scan-secrets.py"
_SPEC = importlib.util.spec_from_file_location("scan_secrets_module", _SCAN_SECRETS_PATH)
_SCAN_SECRETS = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader is not None
_SPEC.loader.exec_module(_SCAN_SECRETS)
_SCAN_SHAPES = [pattern for _, pattern in _SCAN_SECRETS.PATTERNS]


def test_corpus_covers_all_four_accepted_kinds() -> None:
    kinds = {case.kind for case in ADVERSARIAL_INJECTION_CASES}
    assert kinds == {
        INSTRUCTION_OVERRIDE,
        SECRET_EXTRACTION,
        FORGED_SOURCE,
        UNSUPPORTED_ANSWER_PRESSURE,
    }


def test_case_ids_are_stable_and_unique() -> None:
    ids = [case.case_id for case in ADVERSARIAL_INJECTION_CASES]
    assert len(ids) == len(set(ids))
    for case_id in ids:
        assert re.fullmatch(r"injection-[a-z0-9-]+", case_id)


def test_kinds_are_closed_and_case_factory_validates() -> None:
    assert CASE_KINDS == {
        INSTRUCTION_OVERRIDE,
        SECRET_EXTRACTION,
        FORGED_SOURCE,
        UNSUPPORTED_ANSWER_PRESSURE,
    }


def test_seeded_cases_carry_an_inert_answer_marker() -> None:
    seeded = [case for case in ADVERSARIAL_INJECTION_CASES if case.document_source is not None]
    assert seeded
    for case in seeded:
        assert case.answer_marker is not None
        for pattern in _SCAN_SHAPES:
            assert pattern.search(case.answer_marker) is None, f"{case.case_id} marker matches a secret shape"
            assert pattern.search(case.question) is None
            assert pattern.search(case.document_source) is None


def test_only_unsupported_answer_pressure_case_is_seedless() -> None:
    seedless = [case for case in ADVERSARIAL_INJECTION_CASES if case.document_source is None]
    assert seedless
    assert all(case.kind == UNSUPPORTED_ANSWER_PRESSURE for case in seedless)
    assert all(case.answer_marker is None for case in seedless)


def test_unsupported_answer_pressure_question_has_no_lexical_overlap_with_seeded_content() -> None:
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    # Reuse the real lexical scorer so the corpus guarantee is tested against
    # the product implementation, not a reimplementation of its rules.
    service = MixedModeDocumentRetrieverService.__new__(MixedModeDocumentRetrieverService)
    for case in ADVERSARIAL_INJECTION_CASES:
        if case.kind != UNSUPPORTED_ANSWER_PRESSURE:
            continue
        for other in ADVERSARIAL_INJECTION_CASES:
            if other.document_source is None:
                continue
            score = service._score_chunk(query=case.question, content=other.document_source)
            assert score == 0.0, f"{case.case_id} scores {score} against {other.case_id} seed"
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", case.question):
            for other in ADVERSARIAL_INJECTION_CASES:
                if other.document_source is not None and token in other.document_source:
                    raise AssertionError(f"{case.case_id} question token {token!r} overlaps {other.case_id} seed")
        # A digit-free question cannot collide with the letter-only verification
        # sentinel embedded in seeded chunks by the live runner.
        assert not re.search(r"\d", case.question), f"{case.case_id} question must stay digit-free"


def test_case_factory_rejects_invalid_kind() -> None:
    from app.rag.prompt_injection_corpus import InjectionCase

    with pytest.raises(ValueError):
        InjectionCase(case_id="injection-invalid-01", kind="unknown_kind", question="问题", answer_marker=None)
