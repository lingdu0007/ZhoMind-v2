"""bind Candidate jobs to a complete immutable input hash

Revision ID: 20260906_0018
Revises: 20260905_0017
Create Date: 2026-09-06
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import sqlalchemy as sa

from alembic import op

revision = "20260906_0018"
down_revision = "20260905_0017"
branch_labels = None
depends_on = None

_FROZEN_INPUT_HASH_FIELDS = (
    "schema",
    "bundle_id",
    "bundle_sha256",
    "bundle_item_id",
    "bundle_item_sha256",
    "entry_identity",
    "document_identity",
    "requested_generation",
    "editorial_source_revision",
    "input_sha256",
    "chunk_strategy",
    "embedding_configuration",
)


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    snapshot = {field: payload[field] for field in _FROZEN_INPUT_HASH_FIELDS}
    encoded = json.dumps(snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload(value: object) -> Mapping[str, object]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise RuntimeError("existing Candidate Build input record has no canonical payload")
    if any(field not in value for field in _FROZEN_INPUT_HASH_FIELDS):
        raise RuntimeError("existing Candidate Build input record is incomplete")
    return value


def upgrade() -> None:
    with op.batch_alter_table("candidate_build_jobs") as batch:
        batch.add_column(sa.Column("frozen_input_sha256", sa.String(length=64), nullable=True))

    bind = op.get_bind()
    # Existing canonical records and events are append-only. The application
    # recognizes their exact legacy dispatch form only when it matches this
    # recomputed immutable input binding and the historical admin evidence.
    jobs = bind.execute(sa.text("SELECT id FROM candidate_build_jobs")).mappings()
    for job in jobs:
        job_id = str(job["id"])
        input_payload = bind.execute(
            sa.text("SELECT payload FROM canonical_records WHERE stable_id = :stable_id"),
            {"stable_id": f"build_generation:{job_id}"},
        ).scalar_one_or_none()
        frozen_input_sha256 = _canonical_sha256(_payload(input_payload))
        bind.execute(
            sa.text(
                "UPDATE candidate_build_jobs "
                "SET frozen_input_sha256 = :frozen_input_sha256 "
                "WHERE id = :job_id"
            ),
            {"frozen_input_sha256": frozen_input_sha256, "job_id": job_id},
        )

    with op.batch_alter_table("candidate_build_jobs") as batch:
        batch.alter_column(
            "frozen_input_sha256",
            existing_type=sa.String(length=64),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("candidate_build_jobs") as batch:
        batch.drop_column("frozen_input_sha256")
