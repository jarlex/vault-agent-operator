"""Unit tests for configuration loading, env var overrides, defaults, and validation.

Tests cover:
- YAML loading and parsing
- Environment variable overrides take precedence over YAML
- Default values when no YAML or env is provided
- Validation failures for missing required config
- Deep merge logic
- Sub-config models (LLMConfig, MCPConfig, etc.)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from src.config.models import (
    AgentConfig,
    APIConfig,
    LLMConfig,
    LoggingConfig,
    MCPConfig,
    ModelInfo,
    MTLSConfig,
    ScheduledTask,
    SchedulerConfig,
)
from src.config.settings import Settings, _deep_merge, _load_yaml_config


# ===================================================================
# _load_yaml_config
# ===================================================================


class TestLoadYamlConfig:
    """Tests for the YAML config file loader."""

    def test_load_existing_yaml(self, tmp_path: Path) -> None:
        """GIVEN a valid YAML file, WHEN loaded, THEN returns parsed dict."""
        config = {"api": {"host": "127.0.0.1", "port": 9000}}
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(config))

        result = _load_yaml_config(str(yaml_file))
        assert result["api"]["host"] == "127.0.0.1"
        assert result["api"]["port"] == 9000

    def test_load_missing_yaml_returns_empty(self) -> None:
        """GIVEN a non-existent path, WHEN loaded, THEN returns empty dict."""
        result = _load_yaml_config("/nonexistent/path/config.yaml")
        assert result == {}

    def test_load_empty_yaml_returns_empty(self, tmp_path: Path) -> None:
        """GIVEN an empty YAML file, WHEN loaded, THEN returns empty dict."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        result = _load_yaml_config(str(yaml_file))
        assert result == {}

    def test_load_non_dict_yaml_returns_empty(self, tmp_path: Path) -> None:
        """GIVEN a YAML file with a list root, WHEN loaded, THEN returns empty dict."""
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n")

        result = _load_yaml_config(str(yaml_file))
        assert result == {}


# ===================================================================
# _deep_merge
# ===================================================================


class TestDeepMerge:
    """Tests for the recursive dict merge utility."""

    def test_override_wins_on_conflict(self) -> None:
        """GIVEN overlapping keys, WHEN merged, THEN override value wins."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """GIVEN nested dicts, WHEN merged, THEN nested keys are merged recursively."""
        base = {"api": {"host": "0.0.0.0", "port": 8000}}
        override = {"api": {"port": 9000}}
        result = _deep_merge(base, override)
        assert result["api"]["host"] == "0.0.0.0"
        assert result["api"]["port"] == 9000

    def test_none_values_in_override_skipped(self) -> None:
        """GIVEN override has None value, WHEN merged, THEN base value preserved."""
        base = {"a": 1}
        override = {"a": None}
        result = _deep_merge(base, override)
        assert result["a"] == 1

    def test_empty_base(self) -> None:
        """GIVEN empty base, WHEN merged, THEN result is the override."""
        result = _deep_merge({}, {"x": 1})
        assert result == {"x": 1}


# ===================================================================
# Config Data Models
# ===================================================================


class TestAPIConfig:
    """Tests for APIConfig defaults and validation."""

    def test_defaults(self) -> None:
        """GIVEN no arguments, WHEN created, THEN uses spec defaults."""
        cfg = APIConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.request_timeout == 120

    def test_port_validation(self) -> None:
        """GIVEN invalid port, WHEN created, THEN raises ValidationError."""
        with pytest.raises(ValidationError):
            APIConfig(port=0)
        with pytest.raises(ValidationError):
            APIConfig(port=70000)


class TestMTLSConfig:
    """Tests for MTLSConfig defaults."""

    def test_defaults(self) -> None:
        """GIVEN no arguments, WHEN created, THEN mTLS is enabled by default."""
        cfg = MTLSConfig()
        assert cfg.enabled is True
        assert cfg.ca_cert_path == "/certs/ca.pem"


class TestMCPConfig:
    """Tests for MCPConfig defaults and validation."""

    def test_defaults(self) -> None:
        """GIVEN no arguments, WHEN created, THEN uses stdio transport."""
        cfg = MCPConfig()
        assert cfg.transport == "stdio"
        assert cfg.tool_timeout == 30

    def test_invalid_transport(self) -> None:
        """GIVEN invalid transport type, WHEN created, THEN raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPConfig(transport="grpc")


class TestModelInfo:
    """Tests for ModelInfo validation."""

    def test_valid_model(self) -> None:
        """GIVEN valid fields, WHEN created, THEN model is valid."""
        info = ModelInfo(
            name="default",
            provider="github",
            model_id="github/gpt-4o",
            supports_tool_calling=True,
        )
        assert info.model_id == "github/gpt-4o"

    def test_empty_model_id_rejected(self) -> None:
        """GIVEN empty model_id, WHEN created, THEN raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelInfo(name="x", provider="y", model_id="   ")


class TestLLMConfig:
    """Tests for LLMConfig validation and model lookup."""

    def test_defaults_include_one_model(self) -> None:
        """GIVEN no arguments, WHEN created, THEN has at least one default model."""
        cfg = LLMConfig()
        assert len(cfg.models) >= 1
        assert cfg.default_model == "default"

    def test_empty_models_rejected(self) -> None:
        """GIVEN empty models list, WHEN created, THEN raises ValidationError."""
        with pytest.raises(ValidationError):
            LLMConfig(models=[])

    def test_get_model_by_name(self) -> None:
        """GIVEN models configured, WHEN get_model(name), THEN returns the correct model."""
        cfg = LLMConfig(models=[
            ModelInfo(name="fast", provider="test", model_id="test/fast"),
            ModelInfo(name="default", provider="test", model_id="test/default"),
        ])
        model = cfg.get_model("fast")
        assert model is not None
        assert model.model_id == "test/fast"

    def test_get_model_unknown_returns_none(self) -> None:
        """GIVEN models configured, WHEN get_model(unknown), THEN returns None."""
        cfg = LLMConfig()
        assert cfg.get_model("nonexistent") is None

    def test_get_model_none_returns_default(self) -> None:
        """GIVEN models configured, WHEN get_model(None), THEN returns the default model."""
        cfg = LLMConfig()
        model = cfg.get_model(None)
        assert model is not None
        assert model.name == "default"


class TestAgentConfig:
    """Tests for AgentConfig validation."""

    def test_defaults(self) -> None:
        """GIVEN no arguments, WHEN created, THEN max_iterations is 10."""
        cfg = AgentConfig()
        assert cfg.max_iterations == 10

    def test_max_iterations_bounds(self) -> None:
        """GIVEN out-of-range max_iterations, WHEN created, THEN raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentConfig(max_iterations=0)
        with pytest.raises(ValidationError):
            AgentConfig(max_iterations=101)


class TestScheduledTask:
    """Tests for ScheduledTask validation."""

    def test_valid_task(self) -> None:
        """GIVEN valid fields, WHEN created, THEN task is valid."""
        task = ScheduledTask(id="test", cron="*/5 * * * *", prompt="test prompt")
        assert task.enabled is True

    def test_empty_id_rejected(self) -> None:
        """GIVEN empty id, WHEN created, THEN raises ValidationError."""
        with pytest.raises(ValidationError):
            ScheduledTask(id="  ", cron="* * * * *", prompt="test")


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_defaults(self) -> None:
        """GIVEN no arguments, WHEN created, THEN level is INFO, format is json."""
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format == "json"

    def test_invalid_level_rejected(self) -> None:
        """GIVEN invalid log level, WHEN created, THEN raises ValidationError."""
        with pytest.raises(ValidationError):
            LoggingConfig(level="TRACE")


# ===================================================================
# Settings (integration with YAML + env vars)
# ===================================================================


class TestSettings:
    """Tests for the root Settings class with YAML + env var merging."""

    def test_missing_required_config_raises(self, tmp_path: Path) -> None:
        """GIVEN GITHUB_TOKEN and VAULT_TOKEN are not set, WHEN Settings loads, THEN fails with clear error."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump({"api": {"port": 8000}}))

        # Build a clean env: keep essential system vars but remove anything
        # that pydantic-settings might interpret as field values (e.g. AGENT=1).
        clean_env = {
            "CONFIG_PATH": str(yaml_file),
            "HOME": os.environ.get("HOME", "/tmp"),
            "PATH": os.environ.get("PATH", "/usr/bin"),
        }
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(ValueError, match="Required configuration missing"):
                Settings()

    def test_env_var_overrides_yaml(self, tmp_path: Path) -> None:
        """GIVEN YAML sets port=8000 and env sets API__PORT=9000, WHEN loaded, THEN port is 9000."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump({
            "api": {"port": 8000},
        }))

        clean_env = {
            "CONFIG_PATH": str(yaml_file),
            "GITHUB_TOKEN": "test-github-token",
            "VAULT_TOKEN": "test-vault-token",
            "API__PORT": "9000",
            "HOME": os.environ.get("HOME", "/tmp"),
            "PATH": os.environ.get("PATH", "/usr/bin"),
        }
        with patch.dict(os.environ, clean_env, clear=True):
            settings = Settings()
            assert settings.api.port == 9000

    def test_flat_aliases_pushed_to_sub_configs(self, tmp_path: Path) -> None:
        """GIVEN flat env vars (VAULT_ADDR, LOG_LEVEL), WHEN loaded, THEN values appear in sub-configs."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump({}))

        clean_env = {
            "CONFIG_PATH": str(yaml_file),
            "GITHUB_TOKEN": "test-github-token",
            "VAULT_TOKEN": "test-vault-token",
            "VAULT_ADDR": "http://custom-vault:8200",
            "LOG_LEVEL": "DEBUG",
            "HOME": os.environ.get("HOME", "/tmp"),
            "PATH": os.environ.get("PATH", "/usr/bin"),
        }
        with patch.dict(os.environ, clean_env, clear=True):
            settings = Settings()
            assert settings.mcp.vault_addr == "http://custom-vault:8200"
            assert settings.logging.level == "DEBUG"

    def test_defaults_used_when_no_yaml_no_env(self, tmp_path: Path) -> None:
        """GIVEN only required env vars, WHEN loaded, THEN defaults are used for everything else."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        clean_env = {
            "CONFIG_PATH": str(yaml_file),
            "GITHUB_TOKEN": "test-github-token",
            "VAULT_TOKEN": "test-vault-token",
            "HOME": os.environ.get("HOME", "/tmp"),
            "PATH": os.environ.get("PATH", "/usr/bin"),
        }
        with patch.dict(os.environ, clean_env, clear=True):
            settings = Settings()
            assert settings.api.port == 8000
            assert settings.agent.max_iterations == 10
            assert settings.llm.default_model == "default"
