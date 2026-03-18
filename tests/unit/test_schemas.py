"""Unit tests for API schemas (Pydantic model validation).

Tests cover:
- TaskRequest: prompt constraints, optional fields, secret_data
- TaskResponse: status literals, tool calls serialization
- HealthResponse: status literals, required fields
- ModelsResponse: model listing
- ErrorResponse: error structure
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelDetail,
    ModelsResponse,
    TaskRequest,
    TaskResponse,
    ToolCallDetail,
)


# ===================================================================
# TaskRequest
# ===================================================================


class TestTaskRequest:
    """Test TaskRequest validation."""

    def test_valid_minimal_request(self) -> None:
        """GIVEN a valid prompt, WHEN TaskRequest is created, THEN it validates."""
        req = TaskRequest(prompt="read kv/myapp/db")
        assert req.prompt == "read kv/myapp/db"
        assert req.model is None
        assert req.max_iterations is None
        assert req.secret_data is None

    def test_valid_full_request(self) -> None:
        """GIVEN all fields provided, WHEN TaskRequest is created, THEN it validates."""
        req = TaskRequest(
            prompt="write to kv/prod/db",
            model="gpt-4o",
            max_iterations=5,
            secret_data={"password": "secret"},
        )
        assert req.model == "gpt-4o"
        assert req.max_iterations == 5
        assert req.secret_data == {"password": "secret"}

    def test_empty_prompt_rejected(self) -> None:
        """GIVEN an empty prompt, WHEN TaskRequest is created, THEN ValidationError is raised."""
        with pytest.raises(ValidationError):
            TaskRequest(prompt="")

    def test_prompt_exceeds_max_length(self) -> None:
        """GIVEN a prompt exceeding 4096 chars, WHEN created, THEN ValidationError is raised.

        Spec: "HTTP 400 if prompt exceeds 4096 characters"
        """
        long_prompt = "x" * 4097
        with pytest.raises(ValidationError):
            TaskRequest(prompt=long_prompt)

    def test_prompt_at_max_length_accepted(self) -> None:
        """GIVEN a prompt of exactly 4096 chars, WHEN created, THEN it validates."""
        prompt = "x" * 4096
        req = TaskRequest(prompt=prompt)
        assert len(req.prompt) == 4096

    def test_max_iterations_lower_bound(self) -> None:
        """GIVEN max_iterations=0, WHEN created, THEN ValidationError is raised."""
        with pytest.raises(ValidationError):
            TaskRequest(prompt="test", max_iterations=0)

    def test_max_iterations_upper_bound(self) -> None:
        """GIVEN max_iterations=101, WHEN created, THEN ValidationError is raised."""
        with pytest.raises(ValidationError):
            TaskRequest(prompt="test", max_iterations=101)

    def test_max_iterations_valid_range(self) -> None:
        """GIVEN max_iterations in valid range, WHEN created, THEN it validates."""
        req = TaskRequest(prompt="test", max_iterations=1)
        assert req.max_iterations == 1

        req2 = TaskRequest(prompt="test", max_iterations=100)
        assert req2.max_iterations == 100

    def test_secret_data_dict_accepted(self) -> None:
        """GIVEN a secret_data dict, WHEN created, THEN it validates."""
        req = TaskRequest(
            prompt="write secrets",
            secret_data={"key1": "val1", "nested": {"deep": "value"}},
        )
        assert req.secret_data is not None
        assert "key1" in req.secret_data

    def test_missing_prompt_field(self) -> None:
        """GIVEN no prompt field, WHEN TaskRequest is created, THEN ValidationError is raised.

        Spec: "HTTP 400 if the prompt field is missing"
        """
        with pytest.raises(ValidationError):
            TaskRequest()  # type: ignore[call-arg]


# ===================================================================
# TaskResponse
# ===================================================================


class TestTaskResponse:
    """Test TaskResponse serialization and validation."""

    def test_valid_completed_response(self) -> None:
        """GIVEN a completed result, WHEN TaskResponse is created, THEN it validates."""
        resp = TaskResponse(
            status="completed",
            result="The secret contains keys: username, password.",
            model_used="gpt-4o",
            tool_calls=[
                ToolCallDetail(
                    tool_name="vault_kv_read",
                    arguments={"path": "secret/db"},
                    result='{"keys": ["username", "password"]}',
                    is_error=False,
                    duration_ms=150,
                ),
            ],
            duration_ms=2500,
        )
        assert resp.status == "completed"
        assert resp.error is None
        assert len(resp.tool_calls) == 1

    def test_valid_error_response(self) -> None:
        """GIVEN an error result, WHEN TaskResponse is created, THEN it validates."""
        resp = TaskResponse(
            status="error",
            result="LLM error: connection refused",
            model_used="",
            duration_ms=100,
            error="Connection to LLM provider failed",
        )
        assert resp.status == "error"
        assert resp.error is not None

    def test_invalid_status_rejected(self) -> None:
        """GIVEN an invalid status value, WHEN created, THEN ValidationError is raised."""
        with pytest.raises(ValidationError):
            TaskResponse(
                status="unknown",  # type: ignore[arg-type]
                result="x",
                model_used="x",
                duration_ms=1,
            )

    def test_unredacted_data_field(self) -> None:
        """GIVEN unredacted_data provided, WHEN created, THEN it's included."""
        resp = TaskResponse(
            status="completed",
            result="Done",
            model_used="gpt-4o",
            duration_ms=100,
            unredacted_data=[{"tool_name": "vault_kv_read", "response": {"password": "real"}}],
        )
        assert resp.unredacted_data is not None
        assert len(resp.unredacted_data) == 1

    def test_serialization_roundtrip(self) -> None:
        """GIVEN a TaskResponse, WHEN serialized to JSON and back, THEN it roundtrips correctly."""
        resp = TaskResponse(
            status="completed",
            result="Done",
            model_used="test",
            duration_ms=100,
            tool_calls=[
                ToolCallDetail(
                    tool_name="tool1",
                    arguments={"a": 1},
                    result="ok",
                    is_error=False,
                    duration_ms=50,
                ),
            ],
        )
        data = resp.model_dump()
        restored = TaskResponse(**data)
        assert restored.status == "completed"
        assert len(restored.tool_calls) == 1


# ===================================================================
# HealthResponse
# ===================================================================


class TestHealthResponse:
    """Test HealthResponse validation."""

    def test_healthy_response(self) -> None:
        """GIVEN all systems ok, WHEN created, THEN status is 'healthy'."""
        resp = HealthResponse(
            status="healthy",
            agent="ok",
            vault_mcp="connected",
            vault_server="reachable",
            uptime_seconds=1234.5,
            version="0.1.0",
        )
        assert resp.status == "healthy"

    def test_degraded_response(self) -> None:
        """GIVEN LLM unreachable, WHEN created, THEN status is 'degraded'."""
        resp = HealthResponse(
            status="degraded",
            agent="ok",
            vault_mcp="connected",
            vault_server="reachable",
            uptime_seconds=100.0,
            version="0.1.0",
        )
        assert resp.status == "degraded"

    def test_unhealthy_response(self) -> None:
        """GIVEN MCP down, WHEN created, THEN status is 'unhealthy'."""
        resp = HealthResponse(
            status="unhealthy",
            agent="ok",
            vault_mcp="disconnected",
            vault_server="unreachable",
            uptime_seconds=50.0,
            version="0.1.0",
        )
        assert resp.status == "unhealthy"

    def test_invalid_status_rejected(self) -> None:
        """GIVEN invalid health status, WHEN created, THEN raises ValidationError."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="unknown",  # type: ignore[arg-type]
                agent="ok",
                vault_mcp="ok",
                vault_server="ok",
                uptime_seconds=0,
                version="0.1.0",
            )


# ===================================================================
# ModelsResponse
# ===================================================================


class TestModelsResponse:
    """Test ModelsResponse validation."""

    def test_valid_models_response(self) -> None:
        """GIVEN a list of models, WHEN created, THEN it validates."""
        resp = ModelsResponse(
            default_model="default",
            available_models=[
                ModelDetail(
                    name="default",
                    provider="github",
                    model_id="github/gpt-4o",
                    supports_tool_calling=True,
                    is_default=True,
                ),
                ModelDetail(
                    name="fast",
                    provider="github",
                    model_id="github/gpt-4o-mini",
                    supports_tool_calling=True,
                    is_default=False,
                ),
            ],
        )
        assert len(resp.available_models) == 2
        defaults = [m for m in resp.available_models if m.is_default]
        assert len(defaults) == 1

    def test_empty_models_list(self) -> None:
        """GIVEN an empty models list, WHEN created, THEN it validates (no minimum enforced at schema level)."""
        resp = ModelsResponse(default_model="default", available_models=[])
        assert len(resp.available_models) == 0


# ===================================================================
# ErrorResponse
# ===================================================================


class TestErrorResponse:
    """Test ErrorResponse structure."""

    def test_minimal_error(self) -> None:
        """GIVEN just an error message, WHEN created, THEN it validates."""
        resp = ErrorResponse(error="Something went wrong")
        assert resp.error == "Something went wrong"
        assert resp.detail is None
        assert resp.request_id is None

    def test_full_error(self) -> None:
        """GIVEN all error fields, WHEN created, THEN it validates."""
        resp = ErrorResponse(
            error="Validation failed",
            detail="prompt field is required",
            request_id="abc-123",
        )
        assert resp.detail == "prompt field is required"
        assert resp.request_id == "abc-123"
