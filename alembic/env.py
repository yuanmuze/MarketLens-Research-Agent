"""Alembic migration environment.

Reads MARKETLENS_DATABASE_URL from environment (no hardcoded passwords).
Targets the persistence models' metadata so migrations match ORM.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context
from marketlens.persistence.models import Base as PersistenceBase

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The legacy API creates these runtime-owned tables through ``create_all``.
# They predate the migration-managed persistence schema, so Alembic must
# neither create them in a fresh database nor drop them from an existing one.
RUNTIME_OWNED_TABLES = frozenset({"research_jobs", "search_queries"})
target_metadata = PersistenceBase.metadata


def include_object(
    object_: object,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: object | None,
) -> bool:
    """Exclude legacy runtime-owned tables from schema comparisons."""
    del object_, reflected, compare_to
    return not (type_ == "table" and name in RUNTIME_OWNED_TABLES)


def get_url() -> str:
    """Return DB URL from env var or alembic.ini."""
    url = os.environ.get("MARKETLENS_DATABASE_URL", "")
    if not url:
        url = config.get_main_option("sqlalchemy.url", "")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    url = get_url()
    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
