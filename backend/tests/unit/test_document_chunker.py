from __future__ import annotations

import json
from dataclasses import is_dataclass

import pytest

from app.documents.chunker import CHUNK_STRATEGY_PRESETS, chunk_document
from app.documents.types import ChunkRecord, ParsedDocument


def test_chunk_record_is_dataclass() -> None:
    assert is_dataclass(ChunkRecord)


def test_chunk_document_uses_supported_presets_and_metadata_keys() -> None:
    parsed = ParsedDocument(
        source_file="guide.txt",
        file_type="txt",
        text=("0123456789 " * 400).strip(),
    )

    for strategy in ("general", "paper", "qa"):
        chunks = chunk_document(parsed, strategy=strategy)

        assert chunks
        preset = CHUNK_STRATEGY_PRESETS[strategy]
        for index, chunk in enumerate(chunks):
            assert chunk.chunk_index == index
            assert chunk.metadata["source_file"] == "guide.txt"
            assert chunk.metadata["strategy"] == strategy
            assert chunk.metadata["chunk_size"] == preset["chunk_size"]
            assert chunk.metadata["chunk_overlap"] == preset["chunk_overlap"]
            assert chunk.metadata["input_length"] == len(parsed.text)


def test_chunk_index_is_stable_for_same_input() -> None:
    parsed = ParsedDocument(
        source_file="stable.md",
        file_type="md",
        text=("lorem ipsum dolor sit amet " * 300).strip(),
    )

    first = chunk_document(parsed, strategy="general")
    second = chunk_document(parsed, strategy="general")

    assert [chunk.chunk_index for chunk in first] == [chunk.chunk_index for chunk in second]
    assert [chunk.content for chunk in first] == [chunk.content for chunk in second]


def test_chunk_document_preserves_whitespace_at_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = "whitespace_fidelity"
    monkeypatch.setitem(CHUNK_STRATEGY_PRESETS, strategy, {"chunk_size": 5, "chunk_overlap": 2})

    parsed = ParsedDocument(
        source_file="spaces.txt",
        file_type="txt",
        text="abc  def",
    )

    chunks = chunk_document(parsed, strategy=strategy)

    assert [chunk.content for chunk in chunks] == ["abc  ", "  def", "ef"]


def test_agent_chunking_preserves_sections_code_blocks_and_entry_identity() -> None:
    long_recommendation = "。".join([f"第 {index} 条建议包含可验证条件" for index in range(90)]) + "。"
    code_block = "```python\ndef execute_once(key: str) -> None:\n    print(key)\n```"
    parsed = ParsedDocument(
        source_file="workflow.md",
        file_type="md",
        text=(
            "# Decision Question\n\n什么时候使用 workflow？\n\n"
            f"## Recommendation\n\n{long_recommendation}\n\n{code_block}\n\n"
            "## Validation\n\n重复调用必须得到同一个结果。"
        ),
        metadata={
            "entry_id": "pae-workflow-001",
            "title": "Prefer deterministic workflows when the path is known",
            "domain": "workflow-vs-agent",
            "sources": [],
        },
    )

    chunks = chunk_document(parsed, strategy="agent")

    assert len(chunks) >= 4
    assert all(chunk.metadata["strategy"] == "agent" for chunk in chunks)
    assert all(chunk.metadata["entry_id"] == "pae-workflow-001" for chunk in chunks)
    assert all(chunk.metadata["entry_title"] == parsed.metadata["title"] for chunk in chunks)
    assert all(chunk.metadata["domain"] == "workflow-vs-agent" for chunk in chunks)
    assert {chunk.metadata["section_id"] for chunk in chunks} == {
        "decision-question",
        "recommendation",
        "validation",
    }
    assert all(
        not ("# Decision Question" in chunk.content and "## Recommendation" in chunk.content)
        for chunk in chunks
    )
    assert sum(code_block in chunk.content for chunk in chunks) == 1
    assert all("```python" not in chunk.content or "```" in chunk.content.removeprefix("```python") for chunk in chunks)


def test_agent_chunking_binds_each_reviewed_section_to_its_contract_source_snapshot() -> None:
    contract = {
        "claims": [
            {"evidence": [{"section_id": "decision-question", "source_id": "source-workflow"}]},
            {"evidence": [{"section_id": "recommendation", "source_id": "source-agent"}]},
        ]
    }
    parsed = ParsedDocument(
        source_file="workflow.md",
        file_type="md",
        text="# Decision Question\n\nKnown paths.\n\n## Recommendation\n\nBound local Agent work.",
        metadata={
            "entry_id": "pae-workflow-contract-001",
            "title": "Reviewed claim contract",
            "domain": "workflow-vs-agent",
            "claim_evidence_contract": json.dumps(contract),
            "claim_evidence_contract_sha256": "a" * 64,
            "sources": [
                {
                    "source_id": "source-workflow",
                    "title": "Workflow source",
                    "authority": "Authority A",
                    "url": "https://example.com/workflow",
                    "version": "v1",
                    "availability": "verified",
                    "review_date": "2026-08-13",
                    "freshness_days": 90,
                },
                {
                    "source_id": "source-agent",
                    "title": "Agent source",
                    "authority": "Authority B",
                    "url": "https://example.com/agent",
                    "version": "v2",
                    "availability": "verified",
                    "review_date": "2026-08-13",
                    "freshness_days": 90,
                },
            ],
        },
    )

    chunks = chunk_document(parsed, strategy="agent")
    source_by_section = {chunk.metadata["section_id"]: chunk.metadata["source_id"] for chunk in chunks}

    assert source_by_section == {"decision-question": "source-workflow", "recommendation": "source-agent"}
    assert {chunk.metadata["source_title"] for chunk in chunks} == {"Workflow source", "Agent source"}
    assert all(chunk.metadata["claim_evidence_contract_sha256"] == "a" * 64 for chunk in chunks)


def test_agent_chunking_preserves_the_access_appropriate_controlled_source_locator() -> None:
    parsed = ParsedDocument(
        source_file="controlled.md",
        file_type="md",
        text="# Recommendation\n\nUse the reviewed controlled source only within its stated boundary.",
        metadata={
            "entry_id": "ticket19-controlled-source-001",
            "title": "Controlled source projection",
            "domain": "evidence-sufficiency",
            "sources": [
                {
                    "source_id": "ticket19-controlled-source",
                    "title": "Controlled operational evidence",
                    "authority": "ZhoMind architecture group",
                    "access_scope": "controlled_internal",
                    "controlled_locator": "controlled://knowledge/ticket19-controlled-source",
                    "source_tier": "bounded_internal_case",
                    "version": "2026-09-06",
                    "availability": "verified",
                    "review_date": "2026-09-06",
                    "freshness_days": 90,
                }
            ],
        },
    )

    [chunk] = chunk_document(parsed, strategy="agent")

    assert chunk.metadata["source_url"] == "controlled://knowledge/ticket19-controlled-source"
    assert chunk.metadata["source_access_scope"] == "controlled_internal"
    assert chunk.metadata["source_tier"] == "bounded_internal_case"
