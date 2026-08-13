"""SQLAlchemy 2.0 ORM models for PostgreSQL persistence.

Three tables:
  - products: product catalog data (JSON → PostgreSQL)
  - agent_runs: agent request lifecycle records
  - agent_tool_calls: per-tool-call records, FK to agent_runs

Uses SQLAlchemy 2.0 Mapped[] / mapped_column style.
No API keys, auth headers, hidden reasoning, or full provider
responses are ever persisted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Portable JSON: JSONB on PostgreSQL, plain JSON elsewhere (SQLite tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    """Return current UTC datetime (naive, matching TIMESTAMP WITHOUT TZ)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base for all persistence models."""


class ProductRecord(Base):
    """Product catalog row."""

    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    brand: Mapped[str | None] = mapped_column(String(256), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    rating: Mapped[float | None] = mapped_column(Numeric(3, 2), nullable=True)
    review_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Flexible extended fields (attributes, images, url) in JSONB.
    # Named `extra` because `metadata` is reserved in SQLAlchemy Declarative.
    extra: Mapped[dict] = mapped_column("metadata", JSONType, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_products_brand", "brand"),
        Index("ix_products_price", "price"),
        Index("ix_products_rating", "rating"),
    )


class AgentRunRecord(Base):
    """Agent request lifecycle record."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    mode_requested: Mapped[str] = mapped_column(String(16), nullable=False, default="balanced")
    mode_used: Mapped[str] = mapped_column(String(16), nullable=True)
    degraded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    constraints: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    response: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    tool_calls: Mapped[list[AgentToolCallRecord]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan", passive_deletes=True
    )


class AgentToolCallRecord(Base):
    """Per-tool-call record, FK to agent_runs."""

    __tablename__ = "agent_tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    arguments: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    result_product_ids: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    agent_run: Mapped[AgentRunRecord] = relationship(back_populates="tool_calls")


class ProductEmbeddingRecord(Base):
    """Product embedding vector (pgvector), versioned by model name."""

    __tablename__ = "product_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(384), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("product_id", "model_name", name="uq_product_embedding"),
        Index("ix_product_embeddings_product_id", "product_id"),
        Index("ix_product_embeddings_model", "model_name"),
        Index(
            "ix_product_embeddings_embedding_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
