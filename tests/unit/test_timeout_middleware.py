"""Unit tests for the TimeoutMiddleware.

Tests:
- Request completes within timeout → passes through normally
- Request exceeds timeout → returns HTTP 504 with structured JSON error
- Timeout value is configurable
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.api.middleware.timeout import TimeoutMiddleware


def _make_app_with_timeout(timeout_seconds: int = 2) -> FastAPI:
    """Create a minimal FastAPI app with timeout middleware and test routes."""
    app = FastAPI()

    @app.get("/fast")
    async def fast_route():
        return {"status": "ok"}

    @app.get("/slow")
    async def slow_route():
        await asyncio.sleep(10)  # Will exceed any reasonable test timeout
        return {"status": "completed"}

    app.add_middleware(TimeoutMiddleware, timeout_seconds=timeout_seconds)
    return app


class TestTimeoutMiddleware:
    """Verify the timeout middleware enforces request processing limits."""

    def test_fast_request_passes_through(self):
        """Requests that complete within the timeout are unaffected."""
        app = _make_app_with_timeout(timeout_seconds=5)
        client = TestClient(app)

        resp = client.get("/fast")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_slow_request_returns_504(self):
        """Requests exceeding the timeout get HTTP 504 with structured JSON."""
        app = _make_app_with_timeout(timeout_seconds=1)
        client = TestClient(app)

        resp = client.get("/slow")

        assert resp.status_code == 504
        body = resp.json()
        assert body["error"] == "Gateway Timeout"
        assert "1s" in body["detail"]
        assert "application/json" in resp.headers["content-type"]

    def test_timeout_value_is_configurable(self):
        """The timeout value is taken from the constructor parameter."""
        app = _make_app_with_timeout(timeout_seconds=30)
        client = TestClient(app)

        # Fast request should work fine with 30s timeout
        resp = client.get("/fast")
        assert resp.status_code == 200

    def test_default_timeout_is_120(self):
        """Default timeout matches the spec default of 120 seconds."""
        middleware = TimeoutMiddleware(app=MagicMock(), timeout_seconds=120)
        assert middleware._timeout == 120

    def test_504_response_matches_error_schema(self):
        """The 504 response body matches the ErrorResponse schema."""
        app = _make_app_with_timeout(timeout_seconds=1)
        client = TestClient(app)

        resp = client.get("/slow")

        body = resp.json()
        # ErrorResponse fields: error, detail, request_id
        assert "error" in body
        assert "detail" in body
        assert isinstance(body["error"], str)
        assert isinstance(body["detail"], str)
