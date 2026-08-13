"""Add pgvector product embeddings table.

Revision ID: 0002_add_pgvector_embeddings
Revises: 0001_initial
Create Date: 2026-08-13

Design:
  - Separate `product_embeddings` table (not adding vector column to products)
    so embeddings are versioned by model and can be re-imported independently
    of product metadata.
  - `embedding` column is vector(384), matching all-MiniLM-L6-v2.
  - Unique constraint on (product_id, model_name) prevents duplicate writes.
  - HNSW cosine index for efficient top-k similarity search.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "0002_add_pgvector_embeddings"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


def upgrade() -> None:
    """Enable vector extension and create product_embeddings table."""
    # Enable pgvector extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "product_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "product_id",
            sa.String(128),
            sa.ForeignKey("products.product_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(256), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("product_id", "model_name", name="uq_product_embedding"),
    )
    op.create_index(
        "ix_product_embeddings_product_id", "product_embeddings", ["product_id"]
    )
    op.create_index(
        "ix_product_embeddings_model", "product_embeddings", ["model_name"]
    )
    # HNSW cosine index for efficient top-k similarity search.
    # Cosine distance operator (<=>) matches L2-normalized embedding dot-product
    # similarity used by the in-memory backend, preserving result consistency.
    op.execute(
        "CREATE INDEX ix_product_embeddings_embedding_cosine "
        "ON product_embeddings USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Drop product_embeddings table (keep vector extension)."""
    op.execute("DROP INDEX IF EXISTS ix_product_embeddings_embedding_cosine")
    op.drop_index("ix_product_embeddings_model", table_name="product_embeddings")
    op.drop_index("ix_product_embeddings_product_id", table_name="product_embeddings")
    op.drop_table("product_embeddings")
