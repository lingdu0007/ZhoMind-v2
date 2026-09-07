from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.model.base import Base


class GenerationRouteSecret(Base):
    __tablename__ = "generation_route_secrets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ciphertext: Mapped[str] = mapped_column(String(2048), nullable=False)


class GenerationRouteState(Base):
    __tablename__ = "generation_route_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    draft_identity: Mapped[str | None] = mapped_column(String(192))
    active_identity: Mapped[str | None] = mapped_column(String(192))
