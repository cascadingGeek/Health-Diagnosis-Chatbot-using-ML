"""Structured JSON logging configuration.

Call ``configure_logging()`` once at application startup (in the lifespan).
After that, every module obtains a logger via ``logging.getLogger(__name__)``.

All output is JSON, one object per line, making it easy to ingest with log
aggregators (Datadog, Loki, CloudWatch, etc.).
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Formats a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exc"] = record.exc_text

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Merge any extra fields the caller passed via `extra=`
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "message", "module", "msecs", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            }:
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Set up root logger with a JSON handler on stdout.

    Args:
        level: Logging level string (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
