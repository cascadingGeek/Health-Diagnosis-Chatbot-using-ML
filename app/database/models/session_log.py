"""ORM model for a chat session."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ChatSession(Base):
    """Persists the full state of one multi-turn diagnostic conversation.

    Columns
    -------
    id                   Primary key (UUID v4).
    created_at           UTC timestamp of session creation.
    state                Current dialogue state (one of the DialogueState enum values).
    confirmed_symptoms   JSON list of symptom strings the user confirmed.
    asked_symptoms       JSON list of symptom strings that were presented to the user.
    primary_symptom      The user's first reported symptom (used by symptom router).
    followup_queue       Ordered list of remaining follow-up symptom tokens to ask.
    denied_symptoms      JSON list of symptom strings the user explicitly denied.
    conversation_history Full turn-by-turn dialogue as [{role, content}] (LLM pipeline).
    questions_asked      Count of follow-up questions asked this session (LLM pipeline).
    final_diagnosis      Full Layer 3 result JSON (LLM pipeline).
    predicted_disease    Disease name returned by the model (set in PREDICTING state).
    confidence           Model confidence score in [0, 1] (set in PREDICTING state).
    completed            True once the session has reached DONE state.
    """

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    state: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="GREETING",
    )
    confirmed_symptoms: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    asked_symptoms: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    primary_symptom: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        default=None,
    )
    followup_queue: Mapped[list] = mapped_column(
        JSON,
        nullable=True,
        default=list,
    )
    denied_symptoms: Mapped[list] = mapped_column(
        JSON,
        nullable=True,
        default=list,
    )
    conversation_history: Mapped[list] = mapped_column(
        JSON,
        nullable=True,
        default=list,
    )
    questions_asked: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    final_diagnosis: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        default=None,
    )
    predicted_disease: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        default=None,
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        default=None,
    )
    completed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
