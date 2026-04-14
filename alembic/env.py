"""Alembic environment — async SQLAlchemy engine.

All ORM models are imported here so Alembic can auto-generate migrations from
the current ``Base.metadata``.
"""

import asyncio
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root so DATABASE_URL is available when running
# alembic directly (outside of the FastAPI lifespan).
load_dotenv(Path(__file__).parent.parent / ".env")

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Import all models so they register with Base.metadata ────────────────────
from app.database.base import Base
from app.database.models import feedback, session_log  # noqa: F401

# ── Alembic Config object ─────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── Offline migrations ────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without a live DB).

    This mode is useful for reviewing SQL before applying it to production.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online (async) migrations ─────────────────────────────────────────────────

def do_run_migrations(connection: Connection) -> None:
    """Execute pending migrations on an established connection.

    Args:
        connection: A synchronous ``Connection`` provided by ``run_sync``.
    """
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    import os

    url = config.get_main_option("sqlalchemy.url") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No database URL found. Set sqlalchemy.url in alembic.ini "
            "or export DATABASE_URL."
        )

    section = dict(config.get_section(config.config_ini_section) or {})
    section["sqlalchemy.url"] = url

    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
