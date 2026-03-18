"""Request timeout middleware for vault-operator-agent.

Enforces a configurable overall request timeout (default: 120 seconds) on all
incoming requests. When the timeout is exceeded, the client receives a structured
HTTP 504 JSON error response.

The timeout value is read from ``APIConfig.request_timeout`` at middleware
construction time.

Implementation notes:
    Uses ``asyncio.wait_for()`` to wrap the downstream ASGI call chain.
    This approach is compatible with Starlette's ``BaseHTTPMiddleware`` and
    correctly cancels the downstream coroutine on timeout.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.api.schemas import ErrorResponse
from src.logging import get_logger

logger = get_logger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces an overall request processing timeout.

    If the downstream handler does not complete within ``timeout_seconds``,
    the request is cancelled and a structured HTTP 504 response is returned.

    Parameters
    ----------
    app:
        The ASGI application.
    timeout_seconds:
        Maximum seconds to wait for the request to complete.
        Default is 120 seconds per specification.
    """

    def __init__(self, app: Any, timeout_seconds: int = 120) -> None:
        super().__init__(app)
        self._timeout = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with a timeout wrapper."""
        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self._timeout,
            )
            return response
        except asyncio.TimeoutError:
            logger.error(
                "api.request_timeout",
                path=request.url.path,
                method=request.method,
                timeout_seconds=self._timeout,
            )
            body = ErrorResponse(
                error="Gateway Timeout",
                detail=f"Request processing exceeded the {self._timeout}s timeout limit.",
            )
            return JSONResponse(
                status_code=504,
                content=body.model_dump(),
            )
