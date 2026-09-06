from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from app.editorial_authority.service import EditorialAuthorityService


class CanonicalEditorialExportVerifier:
    """Verify a bundle artifact by reconstructing it from private authority."""

    def __init__(self, session: AsyncSession) -> None:
        self._authority = EditorialAuthorityService(session)

    async def verify(self, artifact: dict, artifact_sha256: str) -> dict:
        return await self._authority.verify_approved_export(artifact, artifact_sha256)

    @asynccontextmanager
    async def verify_for_candidate_finalization(
        self,
        artifact: dict,
        artifact_sha256: str,
    ) -> AsyncIterator[dict]:
        async with self._authority.verify_approved_export_for_candidate_finalization(
            artifact,
            artifact_sha256,
        ) as verified:
            yield verified
