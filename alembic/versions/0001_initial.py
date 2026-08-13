"""Initial schema: products, agent_runs, agent_tool_calls.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-13

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# Portable JSON: JSONB on PostgreSQL, plain JSON elsewhere (SQLite tests).
JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create products, agent_runs, and agent_tool_calls tables."""
    op.create_table(
        "products",
        sa.Column("product_id", sa.String(128), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("brand", sa.String(256), nullable=True),
        sa.Column("category", sa.String(64), nullable=True),
        sa.Column("price", sa.Numeric(12, 2), nullable=True),
        sa.Column("rating", sa.Numeric(3, 2), nullable=True),
        sa.Column("review_count", sa.Integer(), nullable=True),
        sa.Column("metadata", JSONType, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_products_brand", "products", ["brand"])
    op.create_index("ix_products_price", "products", ["price"])
    op.create_index("ix_products_rating", "products", ["rating"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("mode_requested", sa.String(16), nullable=False, server_default="balanced"),
        sa.Column("mode_used", sa.String(16), nullable=True),
        sa.Column("degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("constraints", JSONType, nullable=True),
        sa.Column("response", JSONType, nullable=True),
        sa.Column("error_type", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Numeric(12, 2), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_agent_runs_request_id", "agent_runs", ["request_id"], unique=True)

    op.create_table(
        "agent_tool_calls",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("agent_run_id", sa.Integer(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_number", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.String(64), nullable=False),
        sa.Column("arguments", JSONType, nullable=True),
        sa.Column("result_product_ids", JSONType, nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("error_type", sa.String(64), nullable=True),
        sa.Column("latency_ms", sa.Numeric(12, 2), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_agent_tool_calls_agent_run_id", "agent_tool_calls", ["agent_run_id"])


def downgrade() -> None:
    """Drop agent_tool_calls, agent_runs, and products tables."""
    op.drop_index("ix_agent_tool_calls_agent_run_id", table_name="agent_tool_calls")
    op.drop_table("agent_tool_calls")
    op.drop_index("ix_agent_runs_request_id", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_index("ix_products_rating", table_name="products")
    op.drop_index("ix_products_price", table_name="products")
    op.drop_index("ix_products_brand", table_name="products")
    op.drop_table("products")
