import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|password|secret|token)\s*[:=]\s*\S+"
)
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class KnowledgeFeedbackCreate(BaseModel):
    answer_id: str = Field(min_length=1, max_length=64)
    entry_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    label: Literal["helpful", "insufficient_evidence", "outdated", "out_of_scope"]
    note: str | None = Field(default=None, max_length=500)

    @field_validator("answer_id", "entry_id")
    @classmethod
    def strip_identity(cls, value: str) -> str:
        return value.strip()

    @field_validator("note")
    @classmethod
    def validate_explicit_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        note = value.strip()
        if not note:
            return None
        if _CONTROL_CHARACTER.search(note):
            raise ValueError("feedback note contains unsupported control characters")
        if _SECRET_ASSIGNMENT.search(note):
            raise ValueError("feedback note must not contain credentials or secrets")
        return note


class ReviewWorkItemUpdate(BaseModel):
    classification: Literal["p0", "p1", "p2", "p3", "no_action"]
    status: Literal["reviewed", "dismissed"]
