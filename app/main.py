"""Application factory for the Health Diagnosis Chatbot API.

Creates and configures the FastAPI application:
  - Attaches the lifespan context manager (artifact loading, DB init).
  - Registers middleware (CORS, rate limiting).
  - Mounts all v1 routers under ``/api/v1``.
  - Exposes a ``/health`` liveness probe.

Run with::

    uv run -m app.main
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.lifespan import lifespan
from app.middleware.cors import add_cors_middleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.routes.v1 import chat, diagnosis, feedback

app = FastAPI(
    title="Health Symptom Diagnosis Chatbot",
    description=(
        "A machine-learning–based chatbot that collects symptoms through "
        "structured multi-turn dialogue and predicts a likely condition using "
        "a Decision Tree classifier."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
add_cors_middleware(app)
app.add_middleware(RateLimitMiddleware, max_requests=120, window_seconds=60)

# ── Routers ────────────────────────────────────────────────────────────────────
_V1_PREFIX = "/api/v1"

app.include_router(chat.router, prefix=_V1_PREFIX)
app.include_router(diagnosis.router, prefix=_V1_PREFIX)
app.include_router(feedback.router, prefix=_V1_PREFIX)


# ── Health probe ───────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
async def health_check() -> JSONResponse:
    """Liveness probe — returns 200 if the application is running."""
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
