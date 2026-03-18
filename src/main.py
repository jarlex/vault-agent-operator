"""Entry point for vault-operator-agent.

Creates the FastAPI app with a lifespan that:
- Startup: loads settings, configures logging, initialises LLM provider,
  connects MCP client, creates AgentCore, stores singletons in app.state.
- Shutdown: disconnects MCP client, cleans up resources.

Provides a uvicorn entry point with optional mTLS SSL context.

Usage::

    # Via uvicorn directly:
    uvicorn src.main:app --host 0.0.0.0 --port 8000

    # Via the console script:
    vault-operator-agent

    # Programmatically:
    from src.main import main
    main()
"""

from __future__ import annotations

import ssl
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.agent.core import AgentCore
from src.api.app import create_app
from src.config.settings import Settings, get_settings
from src.llm.provider import LLMProvider
from src.logging import get_logger, setup_logging
from src.mcp.client import MCPClient

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan — startup and shutdown events.

    Startup:
        1. Load and validate settings.
        2. Configure structured logging.
        3. Create LLM provider.
        4. Create and connect MCP client.
        5. Create AgentCore.
        6. Store singletons in ``app.state``.

    Shutdown:
        1. Disconnect MCP client.
    """
    # --- Startup ---
    start_time = time.monotonic()

    try:
        settings = get_settings()
    except Exception as exc:
        # Settings validation failure — fatal, cannot start
        print(f"FATAL: Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Configure logging
    setup_logging(
        level=settings.logging.level,
        fmt=settings.logging.format,
        redact_patterns=settings.logging.redact_patterns,
    )

    logger.info(
        "startup.begin",
        api_host=settings.api.host,
        api_port=settings.api.port,
        mtls_enabled=settings.mtls.enabled,
        mcp_transport=settings.mcp.transport,
        default_model=settings.llm.default_model,
        scheduler_enabled=settings.scheduler.enabled,
    )

    if not settings.mtls.enabled:
        logger.warning(
            "startup.mtls_disabled",
            message="mTLS is DISABLED — running in development mode. Do NOT use in production.",
        )

    # Create LLM Provider
    llm_provider = LLMProvider(
        config=settings.llm,
        api_key=settings.github_token,
    )
    logger.info("startup.llm_provider_created", default_model=settings.llm.default_model)

    # Create and connect MCP Client
    mcp_client = MCPClient(config=settings.mcp)
    try:
        await mcp_client.connect()
        logger.info("startup.mcp_connected", tool_count=len(mcp_client.get_tools()))
    except Exception as exc:
        logger.error(
            "startup.mcp_connect_failed",
            error=str(exc),
            exc_type=type(exc).__name__,
            message="MCP client failed to connect. Agent will start in degraded mode.",
        )
        # Don't fatal — the agent can start degraded and reconnect later

    # Create AgentCore
    agent = AgentCore(
        llm_provider=llm_provider,
        mcp_client=mcp_client,
        config=settings.agent,
        vault_addr=settings.mcp.vault_addr,
    )
    logger.info("startup.agent_created")

    # Store singletons in app.state for dependency injection
    app.state.settings = settings
    app.state.llm_provider = llm_provider
    app.state.mcp_client = mcp_client
    app.state.agent = agent
    app.state.start_time = start_time

    logger.info("startup.complete", uptime_ms=int((time.monotonic() - start_time) * 1000))

    yield

    # --- Shutdown ---
    logger.info("shutdown.begin")

    try:
        await mcp_client.disconnect()
        logger.info("shutdown.mcp_disconnected")
    except Exception as exc:
        logger.error(
            "shutdown.mcp_disconnect_error",
            error=str(exc),
            exc_type=type(exc).__name__,
        )

    logger.info("shutdown.complete")


def _build_ssl_context(settings: Settings) -> ssl.SSLContext | None:
    """Build an SSL context for mTLS if enabled.

    Returns ``None`` when mTLS is disabled (plain HTTP).

    The SSL context is configured with:
    - TLS 1.2+ (server protocol)
    - Server certificate + key
    - CA certificate for client cert validation
    - ``ssl.CERT_REQUIRED`` — enforces client certificate
    """
    if not settings.mtls.enabled:
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Load server certificate and key
    ctx.load_cert_chain(
        certfile=settings.mtls.server_cert_path,
        keyfile=settings.mtls.server_key_path,
    )

    # Load CA for client certificate verification
    ctx.load_verify_locations(cafile=settings.mtls.ca_cert_path)

    # Require client certificate
    ctx.verify_mode = ssl.CERT_REQUIRED

    return ctx


# Create the app at module level so uvicorn can import it as `src.main:app`
app = create_app(lifespan=lifespan)
# Note: api_config and mtls_config are applied during lifespan startup
# via settings reload. The module-level app uses defaults; the lifespan
# handler stores settings in app.state for runtime use.


def main() -> None:
    """Console entry point — runs uvicorn with optional mTLS.

    This function is referenced by ``pyproject.toml`` as the console script
    entry point.
    """
    import uvicorn

    try:
        settings = get_settings()
    except Exception as exc:
        print(f"FATAL: Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Configure logging early so startup messages are formatted
    setup_logging(
        level=settings.logging.level,
        fmt=settings.logging.format,
        redact_patterns=settings.logging.redact_patterns,
    )

    ssl_context = _build_ssl_context(settings)

    uvicorn_kwargs: dict = {
        "app": "src.main:app",
        "host": settings.api.host,
        "port": settings.api.port,
        "log_level": settings.logging.level.lower(),
    }

    if ssl_context is not None:
        uvicorn_kwargs["ssl"] = ssl_context
        logger.info(
            "main.starting",
            message="Starting with mTLS enabled",
            host=settings.api.host,
            port=settings.api.port,
        )
    else:
        logger.info(
            "main.starting",
            message="Starting WITHOUT mTLS (development mode)",
            host=settings.api.host,
            port=settings.api.port,
        )

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
