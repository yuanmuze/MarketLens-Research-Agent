"""Add request_hash to agent_runs for idempotency/conflict detection.

Revision ID: 0003_add_request_hash
Revises: 0002_add_pgvector_embeddings
Create Date: 2026-08-13
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0003_add_request_hash"
down_revision: Union[str, None] = "0002_add_pgvector_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add request_hash column (nullable for existing rows)."""
    op.add_column("agent_runs", sa.Column("request_hash", sa.String(64), nullable=True))
    op.create_index("ix_agent_runs_request_hash", "agent_runs", ["request_hash"])


def downgrade() -> None:
    """Drop request_hash column."""
    op.drop_index("ix_agent_runs_request_hash", table_name="agent_runs")
    op.drop_column("agent_runs", "request_hash")
