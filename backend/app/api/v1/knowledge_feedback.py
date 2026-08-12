from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import get_current_user, require_admin
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.knowledge_feedback.schemas import KnowledgeFeedbackCreate, ReviewWorkItemUpdate
from app.knowledge_feedback.service import KnowledgeFeedbackService

router = APIRouter(tags=["knowledge-feedback"])


def _ok(data: dict) -> dict:
    payload = ok_response(data=data, request_id=get_request_id())
    payload.update(data)
    return payload


@router.post("/knowledge-feedback")
async def submit_knowledge_feedback(
    payload: KnowledgeFeedbackCreate,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await KnowledgeFeedbackService(session).submit(user_id=current_user.username, payload=payload)
    return _ok(data)


@router.delete("/knowledge-feedback/{signal_id}")
async def delete_knowledge_feedback(
    signal_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await KnowledgeFeedbackService(session).delete(user_id=current_user.username, signal_id=signal_id)
    return _ok(data)


@router.get("/knowledge-review-queue")
async def list_knowledge_review_queue(
    _current_admin=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await KnowledgeFeedbackService(session).list_review_queue()
    return _ok(data)


@router.patch("/knowledge-review-queue/{item_id}")
async def classify_knowledge_review_item(
    item_id: str,
    payload: ReviewWorkItemUpdate,
    _current_admin=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await KnowledgeFeedbackService(session).classify(item_id=item_id, payload=payload)
    return _ok(data)
