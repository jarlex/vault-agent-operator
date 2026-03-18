"""GET /api/v1/health — health check endpoint.

Reports the aggregate health of the agent and its dependencies:
- Agent itself (always ok if responding)
- MCP client (vault-mcp-server connection)
- Vault server (reachable via MCP health check)

Status logic:
- **healthy**: Agent is up, MCP connected, Vault reachable.
- **degraded**: Agent is up, MCP connected, but LLM provider might be
  unreachable (non-critical for health — tasks will fail but health reports).
- **unhealthy**: MCP client is disconnected (Vault operations impossible).

HTTP status codes:
- 200 for healthy or degraded
- 503 for unhealthy

This endpoint does NOT require mTLS authentication (for infrastructure probes).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Response

from src.api.dependencies import get_mcp_client, get_start_time
from src.api.schemas import HealthResponse
from src.logging import get_logger
from src.mcp.client import MCPClient

logger = get_logger(__name__)

router = APIRouter()

# Version is read from the package
_VERSION: str | None = None


def _get_version() -> str:
    """Get the agent version from the package metadata."""
    global _VERSION
    if _VERSION is None:
        try:
            from src import __version__
            _VERSION = __version__
        except (ImportError, AttributeError):
            _VERSION = "unknown"
    return _VERSION


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns agent and dependency health status.",
)
async def health_check(
    response: Response,
    mcp_client: MCPClient = Depends(get_mcp_client),
    start_time: float = Depends(get_start_time),
) -> HealthResponse:
    """Check health of the agent and its dependencies."""
    uptime_seconds = time.monotonic() - start_time

    # Check MCP client connectivity
    mcp_status = "disconnected"
    vault_status = "unknown"

    if mcp_client.is_connected:
        mcp_status = "connected"
        # Probe the MCP server (which in turn checks Vault)
        try:
            mcp_healthy = await mcp_client.health_check()
            vault_status = "reachable" if mcp_healthy else "unreachable"
        except Exception as exc:
            logger.warning(
                "health.vault_check_failed",
                error=str(exc),
                exc_type=type(exc).__name__,
            )
            vault_status = f"error: {type(exc).__name__}"
    else:
        vault_status = "unknown (mcp disconnected)"

    # Determine overall status
    if mcp_status == "connected" and vault_status == "reachable":
        status = "healthy"
    elif mcp_status == "connected":
        # MCP connected but vault check failed — degraded
        status = "degraded"
    else:
        # MCP disconnected — cannot perform any operations
        status = "unhealthy"
        response.status_code = 503

    logger.debug(
        "health.check",
        status=status,
        mcp_status=mcp_status,
        vault_status=vault_status,
        uptime_seconds=round(uptime_seconds, 1),
    )

    return HealthResponse(
        status=status,
        agent="ok",
        vault_mcp=mcp_status,
        vault_server=vault_status,
        uptime_seconds=round(uptime_seconds, 2),
        version=_get_version(),
    )
