from typing import Any


def ok_response(data: Any, request_id: str) -> dict[str, Any]:
    return {
        "code": "OK",
        "message": "success",
        "data": data,
        "request_id": request_id,
    }


def error_response(code: str, message: str, request_id: str, detail: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id,
    }
    if detail is not None:
        payload["detail"] = detail
    return payload
