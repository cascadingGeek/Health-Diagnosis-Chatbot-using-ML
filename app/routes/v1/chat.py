"""HTTP routes for the multi-turn chat API.

POST /api/v1/chat/message           — send a message, get a bot reply
GET  /api/v1/chat/session/{id}      — retrieve a session's full state
GET  /api/v1/chat/health/llm        — temporary: verify ANTHROPIC_API_KEY is loaded
"""

from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.controllers.v1 import chat_controller
from app.core.exceptions import SessionNotFoundError
from app.database.session import get_async_session
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse, ChatSessionResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_registry(request: Request):
    """FastAPI dependency: retrieve the loaded ModelRegistry from app state."""
    return request.app.state.model_registry


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message and advance the diagnostic dialogue",
)
async def post_message(
    body: ChatMessageRequest,
    db: AsyncSession = Depends(get_async_session),
    registry=Depends(_get_registry),
) -> ChatMessageResponse:
    """Process one user message and return the bot's response.

    Args:
        body:     Validated request body.
        db:       Injected async database session.
        registry: Injected model registry from app state.

    Returns:
        ``ChatMessageResponse`` with bot reply and current session state.

    Raises:
        404: If a ``session_id`` is provided but does not exist.
    """
    try:
        return await chat_controller.handle_message(body, db, registry)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/health/llm",
    include_in_schema=False,
    summary="Temporary: verify ANTHROPIC_API_KEY is loaded in this container",
)
async def llm_health_check() -> JSONResponse:
    """Confirm the Anthropic API key is accessible at runtime.

    Reads from settings (the single source of truth) rather than os.getenv
    so that .env-file-based local deployments and container env vars both work.
    Remove this endpoint once deployment has been verified.
    """
    from app.core.config import settings
    key = settings.anthropic_api_key
    return JSONResponse(
        {
            "api_key_loaded": bool(key),
            "api_key_prefix": (key[:10] + "...") if key else "NOT SET",
            "model": settings.anthropic_model,
        }
    )


@router.get(
    "/session/{session_id}",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve the full state of a chat session",
)
async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> ChatSessionResponse:
    """Return all stored fields for an existing session.

    Args:
        session_id: UUID path parameter.
        db:         Injected async database session.

    Returns:
        ``ChatSessionResponse``.

    Raises:
        404: If the session does not exist.
    """
    try:
        return await chat_controller.get_session(session_id, db)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
