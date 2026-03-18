"""Settings loader — Pydantic BaseSettings with YAML base + env var overrides.

Configuration loading order (highest priority wins):
1. Environment variables
2. config/default.yaml
3. Hardcoded defaults in Pydantic models
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.models import (
    AgentConfig,
    APIConfig,
    LLMConfig,
    LoggingConfig,
    MCPConfig,
    MTLSConfig,
    SchedulerConfig,
)

_DEFAULT_CONFIG_PATH = "config/default.yaml"


def _load_yaml_config(path: str | None = None) -> dict[str, Any]:
    """Load YAML configuration file. Returns empty dict if file not found."""
    config_path = Path(path or _DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        return {}
    with config_path.open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class Settings(BaseSettings):
    """Root settings — loads from YAML then env vars.

    Environment variable names are derived from the field path with double
    underscore separators.  For example, ``mcp.vault_addr`` can be overridden
    by setting ``MCP__VAULT_ADDR`` or the flat alias ``VAULT_ADDR``.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Sub-configs --------------------------------------------------------
    api: APIConfig = APIConfig()
    mtls: MTLSConfig = MTLSConfig()
    mcp: MCPConfig = MCPConfig()
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    logging: LoggingConfig = LoggingConfig()

    # --- Flat convenience aliases (env var friendly) -----------------------
    # These are mapped into sub-configs by the model_validator below.
    github_token: str = ""
    vault_addr: str = ""
    vault_token: str = ""
    mtls_enabled: bool | None = None
    log_level: str | None = None
    mcp_transport: str | None = None
    default_model: str | None = None

    @field_validator("github_token")
    @classmethod
    def _github_token_required(cls, v: str) -> str:
        # Deferred: validated in _post_validate to allow YAML fallback
        return v

    @model_validator(mode="before")
    @classmethod
    def _merge_yaml(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load YAML config and merge it as defaults under env vars."""
        yaml_path = os.environ.get("CONFIG_PATH", _DEFAULT_CONFIG_PATH)
        yaml_data = _load_yaml_config(yaml_path)

        # YAML provides defaults — env vars (already in `values`) win.
        merged = _deep_merge(yaml_data, values)
        return merged

    @model_validator(mode="after")
    def _post_validate(self) -> "Settings":
        """Push flat aliases into sub-configs and validate required fields."""
        # Push flat env-var aliases into their nested locations
        if self.vault_addr:
            self.mcp.vault_addr = self.vault_addr
        if self.vault_token:
            self.mcp.vault_token = self.vault_token
        if self.mtls_enabled is not None:
            self.mtls.enabled = self.mtls_enabled
        if self.log_level:
            self.logging.level = self.log_level.upper()  # type: ignore[assignment]
        if self.mcp_transport:
            self.mcp.transport = self.mcp_transport  # type: ignore[assignment]
        if self.default_model:
            self.llm.default_model = self.default_model

        # Validate required fields (fail fast at startup)
        missing: list[str] = []
        if not self.github_token:
            missing.append("GITHUB_TOKEN")
        if not self.mcp.vault_addr or self.mcp.vault_addr == "http://vault:8200":
            # Allow the default for docker-compose, but warn on truly empty
            pass
        if not self.mcp.vault_token:
            missing.append("VAULT_TOKEN (or MCP__VAULT_TOKEN)")

        if missing:
            raise ValueError(f"Required configuration missing: {', '.join(missing)}")

        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*. Override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        elif val is not None:
            result[key] = val
    return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton-like accessor for application settings.

    Call ``get_settings.cache_clear()`` in tests to force reload.
    """
    return Settings()  # type: ignore[call-arg]
