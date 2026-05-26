"""add_routing_columns

Adds three columns to chat_sessions required by the symptom routing system:
  - primary_symptom:  the user's first reported symptom token
  - followup_queue:   ordered JSON list of remaining follow-up symptom tokens
  - denied_symptoms:  JSON list of symptom tokens the user explicitly denied

Revision ID: a3f1c8d9e2b4
Revises: e9ee42093b4d
Create Date: 2026-05-26 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a3f1c8d9e2b4"
down_revision: Union[str, None] = "e9ee42093b4d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("primary_symptom", sa.String(length=256), nullable=True),
    )
    op.add_column(
        "chat_sessions",
        sa.Column(
            "followup_queue",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            server_default=sa.text("'[]'::json"),
        ),
    )
    op.add_column(
        "chat_sessions",
        sa.Column(
            "denied_symptoms",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            server_default=sa.text("'[]'::json"),
        ),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "denied_symptoms")
    op.drop_column("chat_sessions", "followup_queue")
    op.drop_column("chat_sessions", "primary_symptom")
