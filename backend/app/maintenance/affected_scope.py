from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.maintenance.history import invalid_history
from app.maintenance.schemas import AffectedScope
from app.model.canonical import CanonicalRecordModel
from app.model.knowledge_feedback import KnowledgeFeedbackSignal


async def retain_affected_scope(
    session: AsyncSession, signals: list[KnowledgeFeedbackSignal], prior: dict | None = None,
) -> dict:
    """Keep only concrete canonical targets; the raw reporting envelope remains deletable."""
    scope = {key: list((prior or {}).get(key, [])) for key in ("entry_versions", "gap_contexts")}
    try:
        for signal in signals:
            metadata = signal.normalized_metadata
            if signal.entry_id is None:
                gap = metadata["gap_context"]
                value = {"reason": gap["reason"], "query_condition_set_identity": metadata["query_condition_set_identity"]}
                if gap["query_condition_set_identity"] != value["query_condition_set_identity"]:
                    raise ValueError("gap binding")
                if value not in scope["gap_contexts"]:
                    scope["gap_contexts"].append(value)
                continue
            versions = metadata["knowledge_version_identities"]
            if not isinstance(versions, list) or not versions:
                raise ValueError("publication identities")
            matching = []
            for identity in versions:
                if not isinstance(identity, str):
                    raise ValueError("publication identity")
                record = await session.get(CanonicalRecordModel, identity)
                if (
                    record is None
                    or record.identity_kind != "published_knowledge_version"
                    or record.payload.get("schema") != "published_knowledge_version/v1"
                ):
                    raise ValueError("publication binding")
                if record.payload["entry_identity"] == f"entry:{signal.entry_id}":
                    matching.append({
                        "entry_identity": record.payload["entry_identity"],
                        "publication_identity": record.stable_id,
                        "revision_identity": record.payload["editorial_revision_identity"],
                    })
            if not matching:
                raise ValueError("affected entry binding")
            scope["entry_versions"].extend(value for value in matching if value not in scope["entry_versions"])
        scope["entry_versions"].sort(key=lambda value: (value["entry_identity"], value["publication_identity"]))
        scope["gap_contexts"].sort(key=lambda value: (value["query_condition_set_identity"], value["reason"]))
        return AffectedScope.model_validate(scope).model_dump(mode="json")
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        raise invalid_history() from exc
