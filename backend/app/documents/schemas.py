from typing import Literal

from pydantic import BaseModel, Field

ChunkStrategy = Literal["general", "paper", "qa"]


class BuildDocumentRequest(BaseModel):
    chunk_strategy: ChunkStrategy = "general"


class BatchBuildRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    chunk_strategy: ChunkStrategy = "general"


class BatchDeleteRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
