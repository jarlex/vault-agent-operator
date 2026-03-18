"""Configuration data models for vault-operator-agent.

All configuration sections are defined as Pydantic BaseModel subclasses with
sensible defaults per the specification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    request_timeout: int = Field(default=120, ge=1, description="Overall request timeout in seconds")


class MTLSConfig(BaseModel):
    """Mutual TLS authentication configuration."""

    enabled: bool = True
    ca_cert_path: str = "/certs/ca.pem"
    server_cert_path: str = "/certs/server.pem"
    server_key_path: str = "/certs/server-key.pem"


class MCPConfig(BaseModel):
    """MCP client connection configuration."""

    transport: Literal["stdio", "http"] = "stdio"
    server_binary: str = "/usr/local/bin/vault-mcp-server"
    server_url: str = "http://vault-mcp-server:3000"
    vault_addr: str = "http://vault:8200"
    vault_token: str = ""
    tool_timeout: int = Field(default=30, ge=1, description="Per-tool invocation timeout in seconds")
    reconnect_initial_delay: float = Field(default=1.0, ge=0.1)
    reconnect_max_delay: float = Field(default=60.0, ge=1.0)


class ModelInfo(BaseModel):
    """LLM model definition."""

    name: str = Field(description="Alias (e.g. 'default', 'fast', 'local')")
    provider: str = Field(description="Provider identifier (e.g. 'github', 'openai', 'anthropic')")
    model_id: str = Field(description="Full model ID for LiteLLM (e.g. 'github/gpt-4o')")
    supports_tool_calling: bool = True

    @field_validator("model_id")
    @classmethod
    def model_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id must not be empty")
        return v


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    default_model: str = "default"
    request_timeout: int = Field(default=60, ge=1, description="LLM request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Max retries for transient LLM errors")
    models: list[ModelInfo] = Field(
        default_factory=lambda: [
            ModelInfo(
                name="default",
                provider="github",
                model_id="github/gpt-4o",
                supports_tool_calling=True,
            ),
        ]
    )

    @field_validator("models")
    @classmethod
    def at_least_one_model(cls, v: list[ModelInfo]) -> list[ModelInfo]:
        if not v:
            raise ValueError("At least one model must be configured")
        return v

    def get_model(self, name: str | None = None) -> ModelInfo | None:
        """Look up a model by alias name. Returns None if not found."""
        target = name or self.default_model
        return next((m for m in self.models if m.name == target), None)


class AgentConfig(BaseModel):
    """Agent core configuration."""

    max_iterations: int = Field(default=10, ge=1, le=100, description="Maximum reasoning loop iterations")
    system_prompt_path: str = "config/prompts/system.md"


class ScheduledTask(BaseModel):
    """A single scheduled task definition."""

    id: str = Field(description="Unique task identifier")
    cron: str = Field(description="Cron expression for scheduling")
    prompt: str = Field(description="Natural-language prompt to execute")
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Task id must not be empty")
        return v


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""

    enabled: bool = True
    tasks: list[ScheduledTask] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    redact_patterns: list[str] = Field(
        default_factory=lambda: [
            "token",
            "password",
            "secret",
            "key",
            "authorization",
            "credential",
        ],
        description="Field name patterns whose values will be redacted in logs",
    )
