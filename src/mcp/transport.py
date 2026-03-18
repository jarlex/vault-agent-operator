"""Transport factory for MCP client — stdio (subprocess) or HTTP/SSE.

Provides ``create_transport()`` which returns the appropriate context-manager
pair (read_stream, write_stream) based on the ``MCPConfig.transport`` setting.

Usage::

    from src.config import MCPConfig
    from src.mcp.transport import create_transport

    async with create_transport(config) as (read, write):
        # Use read/write with the mcp ClientSession
        ...
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client

from src.config.models import MCPConfig
from src.logging import get_logger

logger = get_logger(__name__)


class TransportError(Exception):
    """Raised when transport creation or connection fails."""


@asynccontextmanager
async def create_transport(config: MCPConfig) -> AsyncIterator[tuple[Any, Any]]:
    """Create an MCP transport based on config.

    Yields a ``(read_stream, write_stream)`` tuple suitable for
    ``mcp.ClientSession``.

    Parameters
    ----------
    config:
        ``MCPConfig`` with transport type and connection parameters.

    Yields
    ------
    tuple[read_stream, write_stream]
        Streams for the MCP ``ClientSession``.

    Raises
    ------
    TransportError
        If the transport type is unsupported or connection fails.
    """
    if config.transport == "stdio":
        async with _create_stdio_transport(config) as streams:
            yield streams
    elif config.transport == "http":
        async with _create_sse_transport(config) as streams:
            yield streams
    else:
        raise TransportError(f"Unsupported MCP transport: {config.transport!r}")


@asynccontextmanager
async def _create_stdio_transport(config: MCPConfig) -> AsyncIterator[tuple[Any, Any]]:
    """Spawn the vault-mcp-server as a subprocess with stdio transport.

    Environment variables ``VAULT_ADDR`` and ``VAULT_TOKEN`` are forwarded to
    the child process so the MCP server can authenticate with Vault.
    """
    env = _build_mcp_env(config)

    server_params = StdioServerParameters(
        command=config.server_binary,
        args=[],
        env=env,
    )

    logger.info(
        "mcp.transport.stdio.starting",
        binary=config.server_binary,
        vault_addr=config.vault_addr,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        logger.info("mcp.transport.stdio.connected")
        yield (read_stream, write_stream)


@asynccontextmanager
async def _create_sse_transport(config: MCPConfig) -> AsyncIterator[tuple[Any, Any]]:
    """Connect to an MCP server over HTTP/SSE transport."""
    logger.info(
        "mcp.transport.sse.connecting",
        url=config.server_url,
    )

    async with sse_client(config.server_url) as (read_stream, write_stream):
        logger.info("mcp.transport.sse.connected", url=config.server_url)
        yield (read_stream, write_stream)


def _build_mcp_env(config: MCPConfig) -> dict[str, str]:
    """Build environment dict for the vault-mcp-server subprocess.

    Inherits the current process environment and injects/overrides
    ``VAULT_ADDR`` and ``VAULT_TOKEN`` from the agent's configuration.
    """
    env = dict(os.environ)
    env["VAULT_ADDR"] = config.vault_addr
    if config.vault_token:
        env["VAULT_TOKEN"] = config.vault_token
    return env
