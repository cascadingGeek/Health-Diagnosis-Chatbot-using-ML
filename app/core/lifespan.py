"""Application lifespan: startup and shutdown hooks.

Executed once per process.  Responsible for:
  1. Configuring structured logging.
  2. Validating the Anthropic API key (real test call — fails fast on 401 or zero credits).
  3. Loading ML artifacts into ``app.state.model_registry``.
  4. Validating the artifact contract (feature count must match metadata).
  5. Initialising the async database engine and verifying connectivity (SELECT 1).
  6. Running Alembic migrations.
  7. Disposing the engine cleanly on shutdown.

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


async def _validate_anthropic_key() -> None:
    """Confirm the Anthropic API key is valid and the account has credits.

    Makes a real (minimal) API call so that an invalid key, revoked key,
    or zero-credit account causes startup to abort immediately with a clear
    error message — rather than failing silently on the first user request.

    Raises:
        RuntimeError: If the key is missing, invalid, or the account has no credits.
    """
    import anthropic as _anthropic

    key = settings.anthropic_api_key
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or to your Railway/Render environment variables."
        )

    client = _anthropic.Anthropic(api_key=key)
    try:
        client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        logger.info(
            "Anthropic API key validated — key valid, account has credits",
            extra={"model": settings.anthropic_model},
        )
    except _anthropic.AuthenticationError as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is invalid or has been revoked. "
            "Go to console.anthropic.com → API Keys and generate a new key, "
            "then update your .env file (or Railway/Render environment variables). "
            f"Anthropic detail: {exc}"
        ) from exc
    except _anthropic.PermissionDeniedError as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY does not have permission to use this model. "
            f"Anthropic detail: {exc}"
        ) from exc
    except _anthropic.APIStatusError as exc:
        err_lower = str(exc).lower()
        if "credit balance" in err_lower or "insufficient" in err_lower:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is valid but the account has zero credits. "
                "Go to console.anthropic.com → Billing to add credits before starting the server."
            ) from exc
        # Other transient API errors (rate limit, 5xx) — log but do not abort.
        logger.warning(
            "Anthropic API key check returned a non-fatal status %s: %s",
            exc.status_code, exc,
        )
    except Exception as exc:
        # Network error — key might still be valid, don't block startup.
        logger.warning("Anthropic API key validation inconclusive (network error?): %s", exc)


async def _verify_database() -> None:
    """Execute SELECT 1 to confirm the database is actually reachable.

    ``init_engine`` only builds the SQLAlchemy engine object — it does not
    open a real connection.  This check opens a connection and runs a trivial
    query so that a wrong connection string or unreachable host aborts startup
    immediately rather than surfacing as a cryptic error on the first request.

    Raises:
        RuntimeError: If the database cannot be reached.
    """
    from sqlalchemy import text
    from app.database.session import _session_factory  # noqa: PLC2701

    if _session_factory is None:
        raise RuntimeError("Database engine was not initialised before connectivity check.")

    try:
        async with _session_factory() as session:
            await session.execute(text("SELECT 1"))
        logger.info("Database connectivity verified")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach the database: {exc}. "
            "Check DATABASE_URL in your .env file and confirm the server is running."
        ) from exc


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

    # Step 1 — Validate Anthropic API key (real call; aborts on 401 or zero credits).
    try:
        await _validate_anthropic_key()
    except RuntimeError:
        logger.exception("Anthropic API key validation failed — aborting startup")
        raise

    # Step 2 — Load ML artifacts.
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

    # Step 3 — Initialise async DB engine and verify connectivity.
    await init_engine(settings.database_url)
    logger.info("Database engine initialised")

    try:
        await _verify_database()
    except RuntimeError:
        logger.exception("Database connectivity check failed — aborting startup")
        raise

    # Step 4 — Run Alembic migrations.
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
