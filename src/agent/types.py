"""Agent-specific types for vault-operator-agent.

Defines the data classes returned by the Agent Core's reasoning loop:

- **ToolCallRecord**: Audit trail entry for a single MCP tool invocation.
- **AgentResult**: The complete result of an ``AgentCore.execute()`` call,
  including status, result text, and tool call log for internal debugging.

These types are designed to be serializable (for API responses) and to carry
all the information needed by the API layer to construct a ``TaskResponse``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Audit trail entry for a single MCP tool invocation during the reasoning loop.

    Attributes
    ----------
    tool_name:
        The MCP tool that was invoked (e.g. ``"vault_kv_read"``).
    arguments:
        The arguments passed to the tool. NOTE: these are the **redacted**
        arguments (with placeholders, not real secret values) for safe logging
        and audit.
    result:
        The **redacted** tool result string as it was shown to the LLM.
    is_error:
        Whether the MCP tool reported an error.
    duration_ms:
        Wall-clock time for the tool invocation in milliseconds.
    """

    tool_name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    duration_ms: int


@dataclass(slots=True)
class AgentResult:
    """Complete result of an ``AgentCore.execute()`` invocation.

    This is the primary output of the agent's reasoning loop. The API layer
    uses it to construct the consumer-facing ``TaskResponse``.

    **Raw-Response Architecture**:
        The LLM is used ONLY for input processing — understanding the user's
        intent and selecting the right tool(s) with the right arguments. After
        tool execution, raw Vault results are returned directly to the API
        caller without an LLM summarization step. The ``result`` field contains
        the raw tool output (JSON string) when tools were called, or the LLM's
        text when no tools were invoked (error/explanation case).

    Attributes
    ----------
    status:
        Outcome status — ``"completed"`` for successful execution (including
        when the LLM reports a Vault error in its text), ``"error"`` for
        infrastructure failures (LLM down, MCP unreachable, etc.).
    result:
        When tools were called: a JSON string of the raw tool result(s).
        When no tools were called: the LLM's text response (error/explanation).
        When ``status`` is ``"error"``: an error message.
    tool_calls:
        Ordered list of tool invocations made during the reasoning loop.
        Arguments and results are **redacted** (safe for logging).
        Kept for internal debugging/logging — not exposed in the API response.
    model_used:
        The LLM model ID that was used for completions.
    iterations:
        Number of reasoning loop iterations executed.
    raw_tool_results:
        Structured raw tool results for the API consumer. Each entry is a dict
        with ``"tool_name"``, ``"result"`` (the raw MCP response), and
        ``"is_error"`` keys. Populated when tool calls were made. The API layer
        uses this as the primary ``data`` field in the response.
    warning:
        Optional warning message (e.g. ``"max iterations reached"``).
    error_code:
        Machine-readable error code when ``status`` is ``"error"``
        (e.g. ``"llm_timeout"``, ``"mcp_error"``, ``"max_iterations"``).
    """

    status: Literal["completed", "error"]
    result: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_used: str = ""
    iterations: int = 0
    raw_tool_results: list[dict[str, Any]] = field(default_factory=list)
    warning: str | None = None
    error_code: str | None = None
