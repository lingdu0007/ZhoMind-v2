from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryConditionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str = Field(min_length=1, max_length=160)
    field: str = Field(min_length=1, max_length=160)
    operator: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=512)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)
    query_conditions: list[QueryConditionInput] | None = Field(default=None, max_length=32)
    inherit_conditions: bool = False

    @model_validator(mode="after")
    def _require_one_condition_source(self) -> "ChatRequest":
        if self.inherit_conditions and self.query_conditions is not None:
            raise ValueError("inherited conditions cannot be combined with explicit query_conditions")
        return self
