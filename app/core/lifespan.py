"""Application lifespan: startup and shutdown hooks.

Executed once per process.  Responsible for:
  1. Configuring structured logging.
  2. Loading ML artifacts into ``app.state.model_registry``.
  3. Validating the artifact contract (feature count must match metadata).
  4. Initialising the async database engine and running Alembic migrations.
  5. Disposing the engine cleanly on shutdown.

If any startup step fails the exception propagates and uvicorn will NOT start.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.core.config import settings
from app.core.exceptions import ArtifactLoadError
from app.core.logging import configure_logging
from app.database.session import dispose_engine, init_engine
from app.ml.model_registry import model_registry

logger = logging.getLogger(__name__)


async def _run_migrations() -> None:
    """Run ``alembic upgrade head`` as a subprocess.

    Running in-process via run_in_executor causes a deadlock because
    alembic's env.py calls asyncio.run() inside the executor thread while
    the main event loop is already live (asyncpg conflict).
    """
    import asyncio
    import sys

    from app.core.config import PROJECT_ROOT

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "alembic", "upgrade", "head",
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Alembic migration failed (exit {proc.returncode}):\n"
            + stdout.decode()
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI while the app is running.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging()
    logger.info("Application starting up")

    # Load ML artifacts — raises ArtifactLoadError on failure, halting startup.
    try:
        model_registry.load(settings.artifacts_dir)
    except ArtifactLoadError:
        logger.exception("Failed to load ML artifacts — aborting startup")
        raise

    # Validate feature contract.
    model = model_registry.get_model()
    metadata = model_registry.get_metadata()
    expected_features = len(metadata["symptom_list"])
    actual_features = int(model.n_features_in_)
    if expected_features != actual_features:
        msg = (
            f"Artifact mismatch: metadata lists {expected_features} symptoms "
            f"but model expects {actual_features} features."
        )
        logger.error(msg)
        raise ArtifactLoadError(msg)

    logger.info(
        "ML artifacts loaded",
        extra={
            "model_version": metadata.get("model_version"),
            "n_symptoms": expected_features,
            "n_classes": len(metadata.get("class_names", [])),
        },
    )

    # Expose registry on app.state for controllers that receive the Request.
    app.state.model_registry = model_registry

    # Initialise async DB engine.
    await init_engine(settings.database_url)
    logger.info("Database engine initialised")

    # Run migrations.
    try:
        await _run_migrations()
        logger.info("Alembic migrations applied")
    except Exception:
        logger.exception("Alembic migration failed — aborting startup")
        raise

    logger.info("Application ready")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Application shutting down")
    await dispose_engine()
    logger.info("Database engine disposed")
