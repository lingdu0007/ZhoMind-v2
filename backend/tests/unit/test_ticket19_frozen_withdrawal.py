import asyncio
from types import SimpleNamespace

from app.api.v1.documents import _tombstone_document
from app.rag.answer_evidence import AnswerEvidence
from app.rag.evidence_sufficiency import AnswerEvidenceSet, QueryConditionSet


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _Result:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarRows:
        return _ScalarRows(self._rows)


class _TombstoneSession:
    def __init__(self, *, jobs: list[object], messages: list[object]) -> None:
        self._results = (_Result([]), _Result(jobs), _Result(messages))
        self._position = 0

    async def execute(self, _statement: object) -> _Result:
        result = self._results[self._position]
        self._position += 1
        return result

    async def scalars(self, _statement: object) -> _ScalarRows:
        return _ScalarRows([])


def _frozen_trace(document_id: str) -> dict:
    evidence = AnswerEvidence(
        source_id="withdrawn-chunk",
        document_id=document_id,
        generation=1,
        chunk_index=0,
        title="Withdrawn editorial source",
        publication_version="v1",
        excerpt="This frozen excerpt must disappear after the source is withdrawn.",
        retrieval_source=None,
        score=None,
        metadata_items=(
            ("entry_id", "withdrawn-entry-001"),
            ("entry_identity", "entry:withdrawn-entry-001"),
            ("editorial_revision_identity", "editorial_revision:withdrawn-entry-001.r1"),
            ("section_id", "recommendation_or_reviewed_branches"),
            ("title", "Withdrawn editorial source"),
            ("publication_version", "v1"),
        ),
    )
    evidence_set = AnswerEvidenceSet.freeze(
        query_conditions=QueryConditionSet.from_question("What applies for environment=production?"),
        items=(evidence,),
        governing_item=evidence,
    )
    record = evidence_set.to_record()
    return {
        "evidence": [record["items"][0]["evidence"]],
        "answer_evidence_set": record,
    }


def test_tombstone_redacts_frozen_answer_evidence_excerpts_alongside_legacy_trace_evidence() -> None:
    document_id = "withdrawn-document"
    trace = _frozen_trace(document_id)
    message = SimpleNamespace(rag_trace=trace)
    document = SimpleNamespace(
        id=document_id,
        deleted_at=None,
        status="ready",
        latest_requested_generation=2,
        published_generation=2,
        active_build_generation=2,
        active_build_job_id="active-job",
        active_build_heartbeat_at="2026-09-06T00:00:00Z",
    )
    job = SimpleNamespace(status="running", stage="build", progress=100, message="building")
    session = _TombstoneSession(jobs=[job], messages=[message])

    asyncio.run(_tombstone_document(session, document=document))

    frozen_item = message.rag_trace["answer_evidence_set"]["items"][0]
    assert frozen_item["withdrawn"] is True
    assert frozen_item["evidence"]["withdrawn"] is True
    assert "content_preview" not in frozen_item["evidence"]
    assert "This frozen excerpt" not in str(message.rag_trace)
