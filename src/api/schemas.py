"""Request/response Pydantic models for the vault-operator-agent API.

All API payloads are validated through these schemas. The models enforce
constraints from the specification (e.g. prompt max length 4096) and provide
structured, consistent response formats.
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


class TaskResponse(BaseModel):
    """POST /api/v1/tasks response body.

    **Raw-Response Architecture**:
        The ``data`` field contains the raw Vault tool results — this is the
        primary structured output for API consumers. The ``result`` field
        contains a string representation (JSON for tool results, or LLM text
        if no tools were called).
    """

    status: Literal["completed", "error"]
    result: str = Field(description=(
        "Raw tool output (JSON string) when tools were called, "
        "or LLM text when no tools were invoked (error/explanation)."
    ))
    data: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Structured raw Vault tool results. Each entry has 'tool_name', "
            "'result' (raw MCP response), and 'is_error'. This is the primary "
            "output for API consumers — use this instead of parsing 'result'."
        ),
    )
    model_used: str = Field(description="LLM model ID that was used.")
    duration_ms: int = Field(description="Total processing time in milliseconds.")
    error: str | None = Field(default=None, description="Error message if status is 'error'.")


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
