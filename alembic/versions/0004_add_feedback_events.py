"""Add feedback_events table for minimal user feedback.

Revision ID: 0004_add_feedback_events
Revises: 0003_add_request_hash
Create Date: 2026-08-13
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0004_add_feedback_events"
down_revision: Union[str, None] = "0003_add_request_hash"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create feedback_events table."""
    op.create_table(
        "feedback_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "agent_run_id",
            sa.Integer(),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("feedback_type", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(256), nullable=True),
        sa.Column("idempotency_key", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_feedback_events_agent_run_id", "feedback_events", ["agent_run_id"])
    op.create_index("ix_feedback_events_idempotency", "feedback_events", ["idempotency_key"])


def downgrade() -> None:
    """Drop feedback_events table."""
    op.drop_index("ix_feedback_events_idempotency", table_name="feedback_events")
    op.drop_index("ix_feedback_events_agent_run_id", table_name="feedback_events")
    op.drop_table("feedback_events")
