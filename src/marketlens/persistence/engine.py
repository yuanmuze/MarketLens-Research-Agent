"""Database engine and session factory.

Reads MARKETLENS_DATABASE_URL from environment. Defaults to SQLite
for tests/local development. PostgreSQL is used for the `postgres`
catalog backend.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = os.environ.get(
    "MARKETLENS_DATABASE_URL",
    "sqlite:///marketlens_persistence.db",
)

# Catalog backend: json (default) or postgres
CATALOG_BACKEND = os.environ.get("MARKETLENS_CATALOG_BACKEND", "json")

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine (lazy, cached)."""
    global _engine
    if _engine is None:
        connect_args: dict = {}
        if DATABASE_URL.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the session factory (lazy, cached)."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session in a transactional scope.

    Commits on success, rolls back on exception. Re-raises after rollback.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Reset cached engine/session (used by tests to swap DB URL)."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
