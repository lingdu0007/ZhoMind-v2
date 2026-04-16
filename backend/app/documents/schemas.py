from pydantic import BaseModel, Field


class BuildDocumentRequest(BaseModel):
    chunk_strategy: str = "general"


class BatchBuildRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    chunk_strategy: str = "general"


class BatchDeleteRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
