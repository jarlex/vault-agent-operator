"""Request/response Pydantic models for the vault-operator-agent API.

All API payloads are validated through these schemas. The models enforce
constraints from the specification (e.g. prompt max length 4096) and provide
structured, consistent response formats.

Security note:
    ``TaskResponse.unredacted_data`` contains real secret values intended ONLY
    for the API consumer. This field is populated from ``AgentResult.unredacted_responses``
    and MUST NOT be logged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class TaskRequest(BaseModel):
    """POST /api/v1/tasks request body."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Natural-language instruction for the agent.",
    )
    model: str | None = Field(
        default=None,
        description="LLM model alias override. When null, the default model is used.",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Override the maximum number of reasoning loop iterations.",
    )
    secret_data: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Structured secret data to pass to the agent. All values are "
            "treated as secrets and replaced with placeholder tokens before "
            "the LLM sees the prompt."
        ),
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ToolCallDetail(BaseModel):
    """A single tool invocation in the audit trail."""

    tool_name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    duration_ms: int


class TaskResponse(BaseModel):
    """POST /api/v1/tasks response body."""

    status: Literal["completed", "error"]
    result: str = Field(description="Agent's final text response or error message.")
    model_used: str = Field(description="LLM model ID that was used.")
    tool_calls: list[ToolCallDetail] = Field(
        default_factory=list,
        description="Ordered audit trail of MCP tool invocations.",
    )
    duration_ms: int = Field(description="Total processing time in milliseconds.")
    error: str | None = Field(default=None, description="Error message if status is 'error'.")
    unredacted_data: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Full unredacted MCP tool responses for the consumer. "
            "Contains real secret values — the consumer explicitly requested this data."
        ),
    )


class HealthResponse(BaseModel):
    """GET /api/v1/health response body."""

    status: Literal["healthy", "degraded", "unhealthy"]
    agent: str = Field(description="Agent status ('ok' or error description).")
    vault_mcp: str = Field(description="MCP server connection status.")
    vault_server: str = Field(description="Vault server reachability status.")
    uptime_seconds: float = Field(description="Seconds since the agent started.")
    version: str = Field(description="Agent version string.")


class ModelDetail(BaseModel):
    """A single model entry in the models listing."""

    name: str = Field(description="Alias (e.g. 'default', 'fast').")
    provider: str = Field(description="Provider identifier (e.g. 'github', 'openai').")
    model_id: str = Field(description="Full model ID for LiteLLM.")
    supports_tool_calling: bool
    is_default: bool = Field(description="Whether this is the default model.")


class ModelsResponse(BaseModel):
    """GET /api/v1/models response body."""

    default_model: str = Field(description="The default model alias.")
    available_models: list[ModelDetail]


class ErrorResponse(BaseModel):
    """Structured error response returned for all error cases."""

    error: str = Field(description="Human-readable error message.")
    detail: str | None = Field(default=None, description="Additional error details.")
    request_id: str | None = Field(default=None, description="Correlation request ID.")
