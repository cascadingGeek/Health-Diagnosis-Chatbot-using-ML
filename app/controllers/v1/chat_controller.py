"""Orchestration layer for the multi-turn chat flow.

Receives a validated request from the route layer, calls the appropriate
services, and returns a response schema.  No SQLAlchemy or HTTP primitives
leak beyond this layer.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.model_registry import ModelRegistry
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse, ChatSessionResponse
from app.services import dialogue_service, session_service

logger = logging.getLogger(__name__)


async def handle_message(
    request: ChatMessageRequest,
    db: AsyncSession,
    registry: ModelRegistry,
) -> ChatMessageResponse:
    """Process one user message and advance the dialogue state machine.

    Creates a new session when ``request.session_id`` is ``None``.

    Args:
        request:  Validated ``ChatMessageRequest``.
        db:       Async database session (provided by FastAPI dependency).
        registry: Loaded ``ModelRegistry`` (provided by FastAPI dependency).

    Returns:
        ``ChatMessageResponse`` with the bot's reply and updated session info.
    """
    # ── Obtain or create session ──────────────────────────────────────────────
    if request.session_id is None:
        chat_session = await session_service.create_session(db)
        logger.info(
            "New chat session created",
            extra={"session_id": str(chat_session.id)},
        )
    else:
        chat_session = await session_service.get_session(db, request.session_id)

    # ── Advance the state machine ─────────────────────────────────────────────
    turn = dialogue_service.process_message(
        session=chat_session,
        user_message=request.message,
        registry=registry,
    )

    # ── Persist updated session ───────────────────────────────────────────────
    await session_service.save_session(db, chat_session)

    return ChatMessageResponse(
        session_id=chat_session.id,
        state=chat_session.state,
        bot_message=turn.bot_message,
        confirmed_symptoms=list(chat_session.confirmed_symptoms or []),
        diagnosis=turn.diagnosis,
    )


async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession,
) -> ChatSessionResponse:
    """Return the full state of an existing chat session.

    Args:
        session_id: UUID of the session to retrieve.
        db:         Async database session.

    Returns:
        ``ChatSessionResponse`` with all session fields.
    """
    chat_session = await session_service.get_session(db, session_id)
    return session_service.serialise_session(chat_session)
