"""Async SQLAlchemy engine and session factory.

Usage
-----
Inject ``get_async_session`` as a FastAPI dependency to obtain a scoped
``AsyncSession`` that is automatically committed/rolled back and closed.

The engine is initialised once at startup via ``init_engine()`` and disposed
via ``dispose_engine()`` during shutdown.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


async def init_engine(database_url: str) -> None:
    """Create the async engine and session factory.

    Args:
        database_url: Full asyncpg connection string
            (e.g. ``postgresql+asyncpg://user:pass@host/db``).
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )
    logger.info("Async DB engine created")


async def dispose_engine() -> None:
    """Dispose the async engine, closing all pooled connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Async DB engine disposed")


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a scoped async DB session.

    Yields:
        An ``AsyncSession`` that is committed on success or rolled back
        on exception, and always closed afterwards.
    """
    if _session_factory is None:
        raise RuntimeError(
            "Database engine has not been initialised. "
            "Ensure init_engine() is called during application startup."
        )

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
