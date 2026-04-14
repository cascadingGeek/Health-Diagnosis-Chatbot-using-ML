"""ORM model for user feedback on a completed diagnosis session."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Feedback(Base):
    """Stores a user's post-diagnosis rating and optional comment.

    Columns
    -------
    id          Primary key (UUID v4).
    session_id  Foreign key to the associated ``ChatSession``.
    rating      Integer in [1, 5].
    comment     Optional free-text comment.
    created_at  UTC timestamp of submission.
    """

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationship (lazy="noload" keeps queries explicit)
    session = relationship(
        "ChatSession",
        back_populates=None,
        lazy="noload",
    )
