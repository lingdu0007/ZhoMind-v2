from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from app.common.config import Settings

_COLLECTION_PREFIX = "document_chunks_"


@dataclass(frozen=True)
class DenseEmbeddingContract:
    base_url: str
    model: str
    dimension: int
    active: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> "DenseEmbeddingContract":
        base_url = settings.embedding_base_url_normalized
        model = settings.embedding_model_normalized
        dimension = settings.dense_embedding_dim
        active = bool(
            settings.embedding_api_key_configured
            and base_url
            and model
            and dimension > 0
            and settings.milvus_uri_normalized
        )
        return cls(
            base_url=base_url,
            model=model,
            dimension=dimension,
            active=active,
        )

    def fingerprint_payload(self) -> dict[str, str | int]:
        return {
            "embedding_base_url": self.base_url,
            "embedding_model": self.model,
            "dense_embedding_dim": self.dimension,
        }


def dense_mode_active(settings: Settings) -> bool:
    return DenseEmbeddingContract.from_settings(settings).active


def build_embedding_contract_fingerprint(settings: Settings) -> str:
    contract = DenseEmbeddingContract.from_settings(settings)
    payload = json.dumps(contract.fingerprint_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_milvus_collection_name(fingerprint: str) -> str:
    normalized_fingerprint = "".join(ch for ch in fingerprint.lower() if ch in "0123456789abcdef")
    if not normalized_fingerprint:
        raise ValueError("fingerprint must include at least one hexadecimal character")
    return f"{_COLLECTION_PREFIX}{normalized_fingerprint}"
