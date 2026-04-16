from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.common.request_id import get_request_id
from app.common.responses import error_response


@dataclass
class AppError(Exception):
    status_code: int
    code: str
    message: str
    detail: Any | None = None


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        rid = get_request_id() or request.headers.get("x-request-id", "")
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(code=exc.code, message=exc.message, detail=exc.detail, request_id=rid),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = get_request_id() or request.headers.get("x-request-id", "")
        return JSONResponse(
            status_code=500,
            content=error_response(code="INTERNAL_ERROR", message="internal server error", request_id=rid),
        )
