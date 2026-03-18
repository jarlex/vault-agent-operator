"""MCP Client — connect to vault-mcp-server, discover tools, invoke tool calls.

Provides:
- ``connect()`` with stdio transport (spawns vault-mcp-server subprocess)
- ``disconnect()`` for clean shutdown
- Tool discovery via MCP ``tools/list``
- ``get_tools()`` and ``get_tools_as_openai_format()`` for LLM integration
- ``call_tool(name, arguments)`` with timeout handling
- ``health_check()`` ping
- Auto-reconnection with exponential backoff

Usage::

    from src.config import MCPConfig
    from src.mcp.client import MCPClient

    client = MCPClient(config=mcp_config)
    await client.connect()
    tools = client.get_tools_as_openai_format()
    result = await client.call_tool("vault_kv_read", {"path": "secret/myapp"})
    await client.disconnect()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession

from src.config.models import MCPConfig
from src.logging import get_logger
from src.mcp.transport import TransportError, create_transport

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes (matching design doc interfaces exactly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCPTool:
    """An MCP tool discovered from the server."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of an MCP tool invocation."""

    content: str
    is_error: bool


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCPError(Exception):
    """Base exception for MCP client errors."""


class MCPConnectionError(MCPError):
    """Connection to vault-mcp-server failed or was lost."""


class MCPToolError(MCPError):
    """A tool invocation returned an error."""


class MCPTimeoutError(MCPError):
    """A tool invocation or connection timed out."""


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class MCPClient:
    """MCP client for communicating with the vault-mcp-server.

    Parameters
    ----------
    config:
        ``MCPConfig`` with transport type, binary path, timeouts, and
        reconnection settings.
    """

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._session: ClientSession | None = None
        self._tools: list[MCPTool] = []
        self._connected: bool = False

        # Manage the transport context manager's lifecycle
        self._transport_cm: Any = None
        self._read_stream: Any = None
        self._write_stream: Any = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Whether the client has an active connection to the MCP server."""
        return self._connected and self._session is not None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to the vault-mcp-server and discover tools.

        For stdio transport this spawns the server binary as a subprocess.
        For HTTP/SSE transport this connects to the server URL.

        After connecting, performs tool discovery via MCP ``tools/list``.

        Raises
        ------
        MCPConnectionError
            If the connection cannot be established.
        """
        if self._connected:
            logger.debug("mcp.client.already_connected")
            return

        logger.info("mcp.client.connecting", transport=self._config.transport)

        try:
            # Enter the transport context manager
            self._transport_cm = create_transport(self._config)
            streams = await self._transport_cm.__aenter__()
            self._read_stream, self._write_stream = streams

            # Create and initialise the MCP client session
            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.__aenter__()
            await self._session.initialize()

            self._connected = True
            logger.info("mcp.client.connected", transport=self._config.transport)

            # Discover available tools
            await self._discover_tools()

        except Exception as exc:
            self._connected = False
            logger.error(
                "mcp.client.connect_failed",
                transport=self._config.transport,
                error=str(exc),
                exc_type=type(exc).__name__,
            )
            await self._cleanup()
            raise MCPConnectionError(
                f"Failed to connect to vault-mcp-server: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Cleanly close the MCP connection."""
        if not self._connected and self._session is None:
            return

        logger.info("mcp.client.disconnecting")
        await self._cleanup()
        logger.info("mcp.client.disconnected")

    async def reconnect(self) -> None:
        """Disconnect and reconnect with exponential backoff.

        Retries until successful, with delays from
        ``reconnect_initial_delay`` to ``reconnect_max_delay``.
        """
        await self.disconnect()

        delay = self._config.reconnect_initial_delay
        max_delay = self._config.reconnect_max_delay
        attempt = 0

        while True:
            attempt += 1
            try:
                logger.info(
                    "mcp.client.reconnecting",
                    attempt=attempt,
                    delay_s=delay,
                )
                await self.connect()
                logger.info("mcp.client.reconnected", attempt=attempt)
                return
            except MCPConnectionError:
                logger.warning(
                    "mcp.client.reconnect_failed",
                    attempt=attempt,
                    next_delay_s=min(delay * 2, max_delay),
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def get_tools(self) -> list[MCPTool]:
        """Return the list of discovered MCP tools."""
        return list(self._tools)

    def get_tools_as_openai_format(self) -> list[dict[str, Any]]:
        """Convert MCP tools to OpenAI function-calling format.

        Returns a list of dicts suitable for passing as ``tools`` to
        ``litellm.acompletion()``.

        The OpenAI function-calling API requires that every ``"object"``-type
        schema has a ``"properties"`` key (at minimum ``{}``).  MCP tools may
        omit it when the tool takes no parameters (e.g. ``list_mounts``), so
        we normalise the schema defensively before returning it.

        Format::

            [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "description": "Tool description",
                        "parameters": { ... JSON Schema ... }
                    }
                },
                ...
            ]
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._normalise_schema(tool.input_schema),
                },
            }
            for tool in self._tools
        ]

    # ------------------------------------------------------------------
    # Schema normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Ensure an MCP input schema is valid for the OpenAI function-calling API.

        OpenAI requires every ``"object"``-type schema to have a
        ``"properties"`` key.  MCP servers may emit ``{"type": "object"}``
        for tools that accept no parameters, which is valid JSON Schema but
        rejected by OpenAI / GitHub Models.

        This method:
        * Defaults ``type`` to ``"object"`` when missing.
        * Adds ``"properties": {}`` when the type is ``"object"`` and
          ``properties`` is absent.
        * Recursively normalises nested object schemas inside ``properties``.

        The input dict is **not** mutated; a shallow copy is returned when
        changes are needed.
        """
        if not schema:
            return {"type": "object", "properties": {}}

        needs_copy = False

        # Ensure type is present
        schema_type = schema.get("type", "object")
        if "type" not in schema:
            needs_copy = True

        if schema_type == "object":
            # Ensure properties key exists
            if "properties" not in schema:
                needs_copy = True

            # Check if any nested property schemas need normalising
            props = schema.get("properties", {})
            normalised_props: dict[str, Any] | None = None
            for key, prop_schema in props.items():
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
                    norm = MCPClient._normalise_schema(prop_schema)
                    if norm is not prop_schema:
                        if normalised_props is None:
                            normalised_props = dict(props)
                        normalised_props[key] = norm

            if normalised_props is not None:
                needs_copy = True

            if needs_copy:
                result = dict(schema)
                result.setdefault("type", "object")
                if normalised_props is not None:
                    result["properties"] = normalised_props
                else:
                    result.setdefault("properties", {})
                return result

        return schema

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke an MCP tool by name with the given arguments.

        Parameters
        ----------
        name:
            Tool name (must match a discovered tool).
        arguments:
            Tool arguments as a dict (passed as JSON to the MCP server).

        Returns
        -------
        ToolResult
            The tool's response content and error flag.

        Raises
        ------
        MCPConnectionError
            If not connected to the MCP server.
        MCPTimeoutError
            If the tool invocation exceeds ``tool_timeout``.
        MCPError
            For other invocation failures.
        """
        if not self.is_connected or self._session is None:
            raise MCPConnectionError("Not connected to vault-mcp-server")

        start = time.monotonic()

        # INFO: only tool name (no arguments — they may contain secrets after
        # placeholder restoration).
        logger.info("mcp.tool.calling", tool_name=name)

        # WARNING: DEBUG logging below may expose secret values (Vault tokens,
        # passwords, API keys, etc.).  This is acceptable in development but
        # MUST NOT be used in production.  Ensure LOG_LEVEL is INFO or higher
        # in any production deployment.
        logger.debug(
            "mcp.tool.request",
            tool_name=name,
            arguments=arguments,
        )

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=self._config.tool_timeout,
            )

            duration_ms = int((time.monotonic() - start) * 1000)

            # Extract content from the MCP result
            content = self._extract_content(result)
            is_error = getattr(result, "isError", False)

            logger.info(
                "mcp.tool.result",
                tool_name=name,
                is_error=is_error,
                duration_ms=duration_ms,
            )

            # WARNING: DEBUG logging of raw tool result content may expose
            # secret values.  Only enable DEBUG level in development.
            logger.debug(
                "mcp.tool.response_raw",
                tool_name=name,
                is_error=is_error,
                duration_ms=duration_ms,
                content=content,
            )

            return ToolResult(content=content, is_error=is_error)

        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "mcp.tool.timeout",
                tool_name=name,
                timeout_s=self._config.tool_timeout,
                duration_ms=duration_ms,
            )
            raise MCPTimeoutError(
                f"Tool '{name}' timed out after {self._config.tool_timeout}s"
            ) from exc

        except MCPTimeoutError:
            raise  # Don't wrap our own timeout errors

        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "mcp.tool.error",
                tool_name=name,
                duration_ms=duration_ms,
                error=str(exc),
                exc_type=type(exc).__name__,
            )
            raise MCPError(f"Tool '{name}' invocation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Ping the MCP server to verify connectivity.

        Returns ``True`` if the server responds, ``False`` otherwise.
        Does NOT raise exceptions — designed for health endpoint use.
        """
        if not self.is_connected or self._session is None:
            return False

        try:
            # Re-list tools as a lightweight connectivity check
            result = await asyncio.wait_for(
                self._session.list_tools(),
                timeout=min(self._config.tool_timeout, 10),
            )
            return result is not None
        except Exception as exc:
            logger.warning(
                "mcp.health_check.failed",
                error=str(exc),
                exc_type=type(exc).__name__,
            )
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _discover_tools(self) -> None:
        """Query the MCP server for available tools and cache them."""
        if self._session is None:
            return

        logger.info("mcp.tools.discovering")

        result = await self._session.list_tools()
        self._tools = [
            MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
            )
            for tool in result.tools
        ]

        logger.info(
            "mcp.tools.discovered",
            tool_count=len(self._tools),
            tool_names=[t.name for t in self._tools],
        )

    @staticmethod
    def _extract_content(result: Any) -> str:
        """Extract text content from an MCP tool result.

        MCP tool results contain a ``content`` list of content blocks.
        We concatenate all text blocks into a single string.
        """
        if not hasattr(result, "content") or not result.content:
            return ""

        parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                # Binary/image content — return as-is for now
                parts.append(str(block.data))
        return "\n".join(parts) if parts else ""

    async def _cleanup(self) -> None:
        """Clean up session and transport resources."""
        # Close the session
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass  # Best-effort cleanup
            self._session = None

        # Exit the transport context manager
        if self._transport_cm is not None:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            except Exception:
                pass  # Best-effort cleanup
            self._transport_cm = None

        self._read_stream = None
        self._write_stream = None
        self._connected = False
        self._tools = []
