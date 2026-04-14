"""Simple in-memory token-bucket rate limiter.

Limits each client IP to a configurable number of requests per minute.
Since this is a single-process academic project, an in-memory store is
sufficient.  No Redis or external dependencies required.

Usage in main.py::

    from app.middleware.rate_limit import RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)
"""

import time
from collections import defaultdict, deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP.

    Args:
        app:            The ASGI application to wrap.
        max_requests:   Maximum allowed requests per *window_seconds*.
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # ip → deque of request timestamps
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Check rate limit for the incoming request.

        Args:
            request:   Incoming Starlette request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP 429 if the limit is exceeded, otherwise the normal response.
        """
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = self._buckets[client_ip]

        # Evict timestamps outside the window.
        while bucket and bucket[0] < now - self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            return JSONResponse(
                content={"error": "rate_limit_exceeded", "message": "Too many requests."},
                status_code=429,
            )

        bucket.append(now)
        return await call_next(request)
