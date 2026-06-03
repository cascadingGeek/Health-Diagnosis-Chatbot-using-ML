"""Pydantic schemas for the multi-turn chat API."""

import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.diagnosis import LLMDiagnosisResult


class ChatMessageRequest(BaseModel):
    """Incoming message from the frontend.

    Attributes:
        session_id: Existing session UUID, or ``None`` to start a new session.
        message:    The user's raw text input.
    """

    session_id: Optional[uuid.UUID] = None
    message: str = Field(..., min_length=0, max_length=4096)


class ChatMessageResponse(BaseModel):
    """Bot response returned after processing a user message.

    Attributes:
        session_id:         The active (possibly newly created) session UUID.
        state:              Current dialogue state after processing.
        bot_message:        Text to display to the user.
        confirmed_symptoms: Symptoms confirmed so far in this session.
        diagnosis:          Full LLM pipeline result (populated on DONE state).
        response_type:      "question" | "diagnosis" | "inconclusive" | "error".
    """

    session_id: uuid.UUID
    state: str
    bot_message: str
    confirmed_symptoms: list[str]
    diagnosis: Optional[LLMDiagnosisResult] = None
    response_type: str = "question"


class ChatSessionResponse(BaseModel):
    """Full session record returned by the GET endpoint.

    Attributes:
        session_id:        Session UUID.
        state:             Current dialogue state.
        confirmed_symptoms: Confirmed symptom list.
        asked_symptoms:    All symptom strings presented to the user.
        predicted_disease: Disease name if prediction was made.
        confidence:        Confidence score if prediction was made.
        completed:         Whether the session has reached DONE state.
    """

    session_id: uuid.UUID
    state: str
    confirmed_symptoms: list[str]
    asked_symptoms: list[str]
    predicted_disease: Optional[str] = None
    confidence: Optional[float] = None
    completed: bool
