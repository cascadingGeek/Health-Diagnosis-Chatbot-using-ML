"""CORS middleware configuration.

Imported by ``app/main.py`` and applied to the FastAPI application.
Allowed origins are read from ``settings.cors_origins`` so there are no
hardcoded URLs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings


def add_cors_middleware(app: FastAPI) -> None:
    """Attach CORSMiddleware to *app* using the configured allowed origins.

    Args:
        app: The FastAPI application instance.
    """
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
