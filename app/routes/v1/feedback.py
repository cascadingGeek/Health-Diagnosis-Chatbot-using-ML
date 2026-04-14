"""HTTP route for submitting session feedback.

POST /api/v1/feedback
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.controllers.v1 import feedback_controller
from app.core.exceptions import SessionNotFoundError
from app.database.session import get_async_session
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback for a completed diagnosis session",
)
async def post_feedback(
    body: FeedbackRequest,
    db: AsyncSession = Depends(get_async_session),
) -> FeedbackResponse:
    """Persist a user rating and optional comment for a session.

    Args:
        body: Validated request containing session_id, rating, and comment.
        db:   Injected async database session.

    Returns:
        ``FeedbackResponse`` with ``success=True``.

    Raises:
        404: If the referenced session does not exist.
    """
    try:
        return await feedback_controller.submit_feedback(body, db)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
