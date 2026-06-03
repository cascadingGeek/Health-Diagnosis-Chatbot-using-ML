"""CRUD operations for ChatSession and Feedback ORM records.

All database access goes through this service.  Higher layers (controllers)
should never import SQLAlchemy directly.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import SessionNotFoundError
from app.database.models.feedback import Feedback
from app.database.models.session_log import ChatSession
from app.schemas.chat import ChatSessionResponse
from app.schemas.feedback import FeedbackRequest

logger = logging.getLogger(__name__)


async def create_session(db: AsyncSession) -> ChatSession:
    """Create and persist a new ``ChatSession`` in GREETING state.

    Args:
        db: Async SQLAlchemy session.

    Returns:
        The newly created ``ChatSession`` instance.
    """
    session = ChatSession(
        id=uuid.uuid4(),
        created_at=datetime.now(UTC),
        state="GREETING",
        confirmed_symptoms=[],
        asked_symptoms=[],
        primary_symptom=None,
        followup_queue=[],
        denied_symptoms=[],
        conversation_history=[],
        questions_asked=0,
        final_diagnosis=None,
        predicted_disease=None,
        confidence=None,
        completed=False,
    )
    db.add(session)
    await db.flush()  # populate id without committing
    logger.info("Chat session created", extra={"session_id": str(session.id)})
    return session


async def get_session(db: AsyncSession, session_id: uuid.UUID) -> ChatSession:
    """Retrieve an existing ``ChatSession`` by primary key.

    Args:
        db:         Async SQLAlchemy session.
        session_id: UUID of the session to retrieve.

    Returns:
        The ``ChatSession`` ORM object.

    Raises:
        SessionNotFoundError: If no session with that UUID exists.
    """
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise SessionNotFoundError(str(session_id))
    return chat_session


async def save_session(db: AsyncSession, session: ChatSession) -> None:
    """Merge and persist an updated ``ChatSession``.

    Args:
        db:      Async SQLAlchemy session.
        session: The ``ChatSession`` to persist (must already be tracked).
    """
    db.add(session)
    await db.flush()
    logger.debug(
        "Chat session updated",
        extra={"session_id": str(session.id), "state": session.state},
    )


async def create_feedback(
    db: AsyncSession,
    request: FeedbackRequest,
) -> Feedback:
    """Persist a user feedback record.

    Args:
        db:      Async SQLAlchemy session.
        request: Validated ``FeedbackRequest`` schema.

    Returns:
        The newly created ``Feedback`` instance.

    Raises:
        SessionNotFoundError: If the referenced session does not exist.
    """
    # Verify the session exists before creating feedback.
    await get_session(db, request.session_id)

    feedback = Feedback(
        id=uuid.uuid4(),
        session_id=request.session_id,
        rating=request.rating,
        comment=request.comment,
        created_at=datetime.now(UTC),
    )
    db.add(feedback)
    await db.flush()
    logger.info(
        "Feedback created",
        extra={
            "feedback_id": str(feedback.id),
            "session_id": str(request.session_id),
            "rating": request.rating,
        },
    )
    return feedback


def serialise_session(session: ChatSession) -> ChatSessionResponse:
    """Convert a ``ChatSession`` ORM object to a response schema.

    Args:
        session: ORM object to serialise.

    Returns:
        ``ChatSessionResponse`` Pydantic model.
    """
    return ChatSessionResponse(
        session_id=session.id,
        state=session.state,
        confirmed_symptoms=list(session.confirmed_symptoms or []),
        asked_symptoms=list(session.asked_symptoms or []),
        predicted_disease=session.predicted_disease,
        confidence=session.confidence,
        completed=session.completed,
    )
