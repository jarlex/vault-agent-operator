"""Agent-specific types for vault-operator-agent.

Defines the data classes returned by the Agent Core's reasoning loop:

- **ToolCallRecord**: Audit trail entry for a single MCP tool invocation.
- **AgentResult**: The complete result of an ``AgentCore.execute()`` call,
  including status, result text, tool call log, and unredacted secret data
  for the API consumer.

These types are designed to be serializable (for API responses) and to carry
all the information needed by the API layer to construct a ``TaskResponse``.

Security note:
    ``AgentResult.unredacted_responses`` contains real secret values intended
    ONLY for the API consumer. This field MUST NOT be logged or sent to the LLM.
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

    Attributes
    ----------
    status:
        Outcome status — ``"completed"`` for successful execution (including
        when the LLM reports a Vault error in its text), ``"error"`` for
        infrastructure failures (LLM down, MCP unreachable, etc.).
    result:
        The agent's final text response from the LLM, or an error message
        if ``status`` is ``"error"``.
    tool_calls:
        Ordered list of tool invocations made during the reasoning loop.
        Arguments and results are **redacted** (safe for logging).
    model_used:
        The LLM model ID that was used for completions.
    iterations:
        Number of reasoning loop iterations executed.
    unredacted_responses:
        Full, unredacted MCP tool responses for the API consumer. Each entry
        is a dict with ``"tool_name"`` and ``"response"`` keys. This data
        contains real secret values and MUST NOT be logged or sent to the LLM.
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
    unredacted_responses: list[dict[str, Any]] = field(default_factory=list)
    warning: str | None = None
    error_code: str | None = None
