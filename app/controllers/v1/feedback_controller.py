"""Orchestration layer for feedback submission."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.services import session_service

logger = logging.getLogger(__name__)


async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession,
) -> FeedbackResponse:
    """Persist a user feedback record and return an acknowledgement.

    Args:
        request: Validated ``FeedbackRequest``.
        db:      Async database session.

    Returns:
        ``FeedbackResponse`` with ``success=True``.
    """
    await session_service.create_feedback(db, request)
    logger.info(
        "Feedback submitted",
        extra={
            "session_id": str(request.session_id),
            "rating": request.rating,
        },
    )
    return FeedbackResponse(success=True)
