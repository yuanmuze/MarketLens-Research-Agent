"""SQLAlchemy database setup for MarketLens API.

Supports SQLite (default, for testing/local) and PostgreSQL (production).
Uses pgvector extension for PostgreSQL when available.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Database URL from environment, default to SQLite
DATABASE_URL = os.environ.get(
    "MARKETLENS_DATABASE_URL",
    "sqlite:///marketlens.db",
)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class ResearchJobRecord(Base):
    """SQLAlchemy model for persisting research jobs."""

    __tablename__ = "research_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    request_id = Column(String(64), nullable=False)
    query = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    max_results = Column(Integer, default=10)
    enable_web_search = Column(Integer, default=0)

    # Timing
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Float, nullable=True)

    # Results
    report_text = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Metrics
    product_count = Column(Integer, nullable=True)
    tool_calls = Column(Integer, default=0)
    retries = Column(Integer, default=0)
    evidence_count = Column(Integer, nullable=True)
    constraints_satisfied = Column(Integer, nullable=True)  # 0 or 1


class SearchQueryRecord(Base):
    """SQLAlchemy model for search queries."""

    __tablename__ = "search_queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    top_k = Column(Integer, default=20)
    result_count = Column(Integer, default=0)
    duration_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source = Column(String(50), default="hybrid")  # bm25, embedding, hybrid, reranked


# Engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        connect_args = {}
        if DATABASE_URL.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
    return _engine


def get_session() -> Session:
    """Get a new SQLAlchemy session.

    Returns:
        A new SQLAlchemy Session.
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=get_engine())


def drop_db() -> None:
    """Drop all tables (for testing only)."""
    Base.metadata.drop_all(bind=get_engine())
