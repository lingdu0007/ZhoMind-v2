from fastapi import APIRouter

from app.common.config import get_settings
from app.common.request_id import get_request_id
from app.common.responses import ok_response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return ok_response(
        data={"status": "up", "service": settings.app_name, "version": settings.app_version},
        request_id=get_request_id(),
    )
