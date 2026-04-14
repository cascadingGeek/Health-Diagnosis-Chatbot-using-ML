"""Pydantic schemas for feedback submission."""

import uuid

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """User feedback on a completed diagnosis session.

    Attributes:
        session_id: UUID of the completed ``ChatSession``.
        rating:     Score from 1 (poor) to 5 (excellent).
        comment:    Optional free-text comment.
    """

    session_id: uuid.UUID
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(None, max_length=2048)


class FeedbackResponse(BaseModel):
    """Acknowledgement returned after successful feedback persistence.

    Attributes:
        success: Always ``True`` on success.
    """

    success: bool = True
