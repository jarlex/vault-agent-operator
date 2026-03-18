"""FastAPI application factory for vault-operator-agent.

Creates the FastAPI app with:
- API route registration (v1)
- mTLS middleware
- Global exception handlers (structured JSON for all errors)
- CORS middleware (configurable)
- Request ID propagation

The app is created via ``create_app()`` and wired up with a lifespan
context manager in ``src/main.py``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware.mtls import MTLSMiddleware
from src.api.middleware.timeout import TimeoutMiddleware
from src.api.routes import health, models, tasks
from src.api.schemas import ErrorResponse
from src.config.models import APIConfig, MTLSConfig
from src.logging import get_logger

logger = get_logger(__name__)


def create_app(
    mtls_config: MTLSConfig | None = None,
    api_config: APIConfig | None = None,
    lifespan: Callable | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    mtls_config:
        mTLS configuration. When ``None``, a default (disabled) config is used.
    api_config:
        API configuration (includes request_timeout). When ``None``, defaults apply.
    lifespan:
        FastAPI lifespan context manager for startup/shutdown events.

    Returns
    -------
    FastAPI
        The configured application, ready to be served by uvicorn.
    """
    app = FastAPI(
        title="vault-operator-agent",
        description="AI agent for HashiCorp Vault operations via MCP",
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- Routes ---
    app.include_router(
        tasks.router,
        prefix="/api/v1",
        tags=["tasks"],
    )
    app.include_router(
        health.router,
        prefix="/api/v1",
        tags=["health"],
    )
    app.include_router(
        models.router,
        prefix="/api/v1",
        tags=["models"],
    )

    # --- Middleware (applied in reverse order — last added = outermost) ---

    # CORS — permissive for API consumers; tighten for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # mTLS identity extraction
    effective_mtls = mtls_config or MTLSConfig(enabled=False)
    app.add_middleware(MTLSMiddleware, config=effective_mtls)

    # Request timeout enforcement
    effective_api = api_config or APIConfig()
    app.add_middleware(TimeoutMiddleware, timeout_seconds=effective_api.request_timeout)

    # --- Exception handlers ---
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _global_error_handler)

    logger.info(
        "app.created",
        mtls_enabled=effective_mtls.enabled,
        request_timeout_seconds=effective_api.request_timeout,
    )

    return app


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic/FastAPI validation errors with structured JSON."""
    errors = exc.errors()

    # Build a human-readable summary
    messages: list[str] = []
    for err in errors:
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "Unknown validation error")
        messages.append(f"{loc}: {msg}")

    detail = "; ".join(messages)

    logger.warning(
        "api.validation_error",
        path=request.url.path,
        detail=detail,
    )

    body = ErrorResponse(
        error="Validation error",
        detail=detail,
    )
    return JSONResponse(status_code=422, content=body.model_dump())


async def _global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler — ensures clients always get structured JSON.

    This handler catches any unhandled exception that slips through route-level
    error handling. It logs the full stack trace internally but returns only a
    generic message to the client (no raw tracebacks).
    """
    logger.error(
        "api.unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_type=type(exc).__name__,
        exc_info=True,
    )

    body = ErrorResponse(
        error="Internal server error",
        detail="An unexpected error occurred. Check server logs for details.",
    )
    return JSONResponse(status_code=500, content=body.model_dump())
