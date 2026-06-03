"""Orchestration layer for the multi-turn chat flow.

Receives a validated request from the route layer, calls the LLM orchestrator
(three-layer hybrid pipeline), and returns a response schema.
No SQLAlchemy or HTTP primitives leak beyond this layer.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.model_registry import ModelRegistry
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse, ChatSessionResponse
from app.schemas.diagnosis import LLMDiagnosisResult
from app.services import session_service
from app.services.llm_orchestrator import process_message as orchestrate

logger = logging.getLogger(__name__)


async def handle_message(
    request: ChatMessageRequest,
    db: AsyncSession,
    registry: ModelRegistry,  # noqa: ARG001 — kept in signature for DI compatibility
) -> ChatMessageResponse:
    """Process one user message through the three-layer hybrid pipeline.

    Creates a new session when ``request.session_id`` is ``None``.
    The orchestrator mutates the session object; this function persists it.

    Args:
        request:  Validated ``ChatMessageRequest``.
        db:       Async database session (provided by FastAPI dependency).
        registry: Loaded ``ModelRegistry`` (kept for interface compatibility;
            the orchestrator accesses the module-level singleton directly).

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

    # ── GREETING state: emit welcome without calling the LLM ─────────────────
    if chat_session.state == "GREETING":
        chat_session.state = "COLLECTING"
        await session_service.save_session(db, chat_session)
        return ChatMessageResponse(
            session_id=chat_session.id,
            state="COLLECTING",
            bot_message=(
                "Hello! I'm Dr. Melvis, your AI health assistant. "
                "I'll ask you a few questions about your symptoms to help identify "
                "a possible condition.\n\n"
                "Please describe how you are feeling in your own words — "
                "a full sentence is fine."
            ),
            confirmed_symptoms=[],
            response_type="question",
        )

    # ── All other states: run the three-layer hybrid pipeline ─────────────────
    result = await orchestrate(
        session=chat_session,
        user_message=request.message,
    )

    # ── Persist updated session ───────────────────────────────────────────────
    await session_service.save_session(db, chat_session)

    # ── Build diagnosis schema if pipeline returned one ───────────────────────
    llm_diagnosis: LLMDiagnosisResult | None = None
    raw_diagnosis = result.get("diagnosis")
    if raw_diagnosis:
        llm_diagnosis = LLMDiagnosisResult(
            disease=raw_diagnosis.get("display_disease"),
            confidence=raw_diagnosis.get("display_confidence"),
            description=raw_diagnosis.get("explanation", ""),
            precautions=raw_diagnosis.get("precautions", []),
            is_plausible=raw_diagnosis.get("is_plausible", False),
            urgency=raw_diagnosis.get("urgency", "moderate"),
            when_to_see_doctor=raw_diagnosis.get("when_to_see_doctor", ""),
            disclaimer=raw_diagnosis.get("disclaimer", ""),
        )

    return ChatMessageResponse(
        session_id=chat_session.id,
        state=result["session_state"],
        bot_message=result["message"],
        confirmed_symptoms=list(chat_session.confirmed_symptoms or []),
        diagnosis=llm_diagnosis,
        response_type=result["response_type"],
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
