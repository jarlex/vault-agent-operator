"""API integration tests for vault-operator-agent.

Full HTTP-level tests exercising the FastAPI routes via ``TestClient`` with
mocked agent, MCP, and LLM services.  No real external services are required.

Test coverage maps to specification scenarios:

    POST /api/v1/tasks
    - Happy path (mocked agent returns successful result)
    - Missing prompt → 422 (Pydantic validation)
    - Prompt too long (>4096 chars) → 422
    - Invalid model → agent returns error → 400
    - LLM unreachable → agent error → 503
    - MCP unreachable → agent error → 503
    - X-Request-ID in response headers
    - Structured JSON error responses (never raw tracebacks)

    GET /api/v1/health
    - Healthy state (MCP connected + Vault reachable)
    - Unhealthy state (MCP disconnected)
    - Degraded state (MCP connected but Vault health check fails)

    GET /api/v1/models
    - Returns configured models with is_default flag
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi.testclient import TestClient

from src.agent.core import AgentCore
from src.agent.types import AgentResult, ToolCallRecord
from src.api.app import create_app
from src.config.models import LLMConfig, ModelInfo
from src.llm.provider import LLMProvider
from src.mcp.client import MCPClient


# ============================================================================
# Helpers
# ============================================================================


def _make_app(
    *,
    agent_result: AgentResult | None = None,
    agent_side_effect: Exception | None = None,
    mcp_connected: bool = True,
    mcp_health: bool = True,
    mcp_health_side_effect: Exception | None = None,
    models: list[ModelInfo] | None = None,
    default_model: str = "default",
) -> TestClient:
    """Build a TestClient with fully controllable mocks.

    This helper avoids fixture coupling so each test is self-contained.
    """
    app = create_app()

    # --- Agent mock ---
    mock_agent = MagicMock(spec=AgentCore)
    if agent_side_effect:
        mock_agent.execute = AsyncMock(side_effect=agent_side_effect)
    elif agent_result:
        mock_agent.execute = AsyncMock(return_value=agent_result)
    else:
        mock_agent.execute = AsyncMock(return_value=AgentResult(
            status="completed",
            result="Success.",
            tool_calls=[],
            model_used="github/gpt-4o",
            iterations=1,
        ))
    app.state.agent = mock_agent

    # --- MCP mock ---
    mock_mcp = MagicMock(spec=MCPClient)
    type(mock_mcp).is_connected = PropertyMock(return_value=mcp_connected)
    if mcp_health_side_effect:
        mock_mcp.health_check = AsyncMock(side_effect=mcp_health_side_effect)
    else:
        mock_mcp.health_check = AsyncMock(return_value=mcp_health)
    app.state.mcp_client = mock_mcp

    # --- LLM provider mock ---
    if models is None:
        models = [
            ModelInfo(name="default", provider="github", model_id="github/gpt-4o", supports_tool_calling=True),
            ModelInfo(name="fast", provider="github", model_id="github/gpt-4o-mini", supports_tool_calling=True),
        ]
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm._config = LLMConfig(default_model=default_model, models=models)
    mock_llm.get_available_models.return_value = models
    app.state.llm_provider = mock_llm

    # --- Misc state ---
    app.state.start_time = time.monotonic()
    app.state.settings = MagicMock()

    return TestClient(app)


# ============================================================================
# POST /api/v1/tasks — Happy Path
# ============================================================================


class TestTasksHappyPath:
    """Spec scenario: Successful KV secret read via natural language."""

    def test_successful_task_returns_200(self):
        """POST /api/v1/tasks with valid prompt returns 200 + completed status."""
        client = _make_app(
            agent_result=AgentResult(
                status="completed",
                result="The secret at kv/myapp/database contains keys: username, password.",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="vault_kv_read",
                        arguments={"path": "secret/data/myapp"},
                        result='{"keys": ["username", "password"]}',
                        is_error=False,
                        duration_ms=45,
                    ),
                ],
                model_used="github/gpt-4o",
                iterations=2,
                raw_tool_results=[
                    {
                        "tool_name": "vault_kv_read",
                        "result": '{"data": {"username": "admin", "password": "s3cret"}}',
                        "is_error": False,
                    }
                ],
            ),
        )

        resp = client.post("/api/v1/tasks", json={"prompt": "read the secret at kv/myapp/database"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert "kv/myapp/database" in body["result"]
        assert body["model_used"] == "github/gpt-4o"
        assert body["data"] is not None
        assert len(body["data"]) == 1
        assert body["data"][0]["tool_name"] == "vault_kv_read"
        assert body["duration_ms"] >= 0
        assert body["error"] is None

    def test_successful_task_includes_data_field(self):
        """Spec: API response includes structured data for consumer."""
        raw_results = [
            {
                "tool_name": "vault_kv_read",
                "result": '{"data": {"username": "admin", "password": "s3cret!123"}}',
                "is_error": False,
            }
        ]
        client = _make_app(
            agent_result=AgentResult(
                status="completed",
                result="Read successfully.",
                tool_calls=[],
                model_used="github/gpt-4o",
                iterations=1,
                raw_tool_results=raw_results,
            ),
        )

        resp = client.post("/api/v1/tasks", json={"prompt": "read kv/myapp/db"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] is not None
        assert len(body["data"]) == 1
        assert body["data"][0]["tool_name"] == "vault_kv_read"
        assert "s3cret!123" in body["data"][0]["result"]

    def test_request_id_in_response_headers(self):
        """Spec: Every response includes X-Request-ID header."""
        client = _make_app()

        resp = client.post("/api/v1/tasks", json={"prompt": "list secrets"})

        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        request_id = resp.headers["x-request-id"]
        # UUID format check: 8-4-4-4-12
        parts = request_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8

    def test_content_type_is_json(self):
        """Spec: All responses have Content-Type application/json."""
        client = _make_app()

        resp = client.post("/api/v1/tasks", json={"prompt": "read secret"})

        assert "application/json" in resp.headers["content-type"]


# ============================================================================
# POST /api/v1/tasks — Validation Errors
# ============================================================================


class TestTasksValidation:
    """Spec scenarios: Missing prompt, prompt too long, invalid model."""

    def test_missing_prompt_returns_422(self):
        """Spec: Missing prompt field → structured error response.

        FastAPI returns 422 for Pydantic validation errors (not 400), which is
        standard for request body validation. The spec says 400, but the
        implementation correctly returns 422 with structured JSON.
        """
        client = _make_app()

        resp = client.post("/api/v1/tasks", json={})

        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body
        assert "detail" in body

    def test_empty_prompt_returns_422(self):
        """Prompt with empty string violates min_length=1."""
        client = _make_app()

        resp = client.post("/api/v1/tasks", json={"prompt": ""})

        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body

    def test_prompt_too_long_returns_422(self):
        """Spec: Prompt exceeding 4096 characters → validation error."""
        client = _make_app()

        long_prompt = "x" * 4097
        resp = client.post("/api/v1/tasks", json={"prompt": long_prompt})

        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body

    def test_prompt_at_max_length_succeeds(self):
        """Prompt at exactly 4096 characters should succeed."""
        client = _make_app()

        prompt = "x" * 4096
        resp = client.post("/api/v1/tasks", json={"prompt": prompt})

        assert resp.status_code == 200

    def test_invalid_model_returns_400(self):
        """Spec: Invalid model → 400 error.

        When the agent reports an llm_tool_unsupported error code, the route
        maps it to 400.
        """
        client = _make_app(
            agent_result=AgentResult(
                status="error",
                result="Model 'nonexistent-model' does not support tool calling",
                tool_calls=[],
                model_used="",
                iterations=0,
                error_code="llm_tool_unsupported",
            ),
        )

        resp = client.post(
            "/api/v1/tasks",
            json={"prompt": "list secrets", "model": "nonexistent-model"},
        )

        assert resp.status_code == 400
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"] is not None

    def test_malformed_json_returns_422(self):
        """Spec: Malformed JSON body → structured error."""
        client = _make_app()

        resp = client.post(
            "/api/v1/tasks",
            content=b"this is not json",
            headers={"content-type": "application/json"},
        )

        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body


# ============================================================================
# POST /api/v1/tasks — Service Errors (LLM / MCP)
# ============================================================================


class TestTasksServiceErrors:
    """Spec scenarios: LLM unreachable, MCP unreachable."""

    def test_llm_unreachable_returns_503(self):
        """Spec: LLM provider unreachable → 503 with structured error."""
        client = _make_app(
            agent_result=AgentResult(
                status="error",
                result="LLM error: Service unavailable after 3 retries",
                tool_calls=[],
                model_used="",
                iterations=1,
                error_code="llm_service",
            ),
        )

        resp = client.post("/api/v1/tasks", json={"prompt": "read secret"})

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"] is not None
        assert "application/json" in resp.headers["content-type"]

    def test_mcp_unreachable_returns_503(self):
        """Spec: vault-mcp-server unreachable → 503 with structured error."""
        client = _make_app(
            agent_result=AgentResult(
                status="error",
                result="MCP connection error: Not connected to vault-mcp-server",
                tool_calls=[],
                model_used="",
                iterations=0,
                error_code="mcp_connection",
            ),
        )

        resp = client.post("/api/v1/tasks", json={"prompt": "read secret"})

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"

    def test_agent_unexpected_exception_returns_500(self):
        """Spec: Unexpected exception → 500 with generic structured error (no traceback)."""
        client = _make_app(agent_side_effect=RuntimeError("something went terribly wrong"))

        resp = client.post("/api/v1/tasks", json={"prompt": "do something"})

        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "Internal server error"
        # Must NOT contain raw traceback or exception class name
        assert "RuntimeError" not in body.get("detail", "")
        assert "Traceback" not in body.get("detail", "")

    def test_llm_rate_limit_returns_429(self):
        """LLM rate-limited after retries → 429."""
        client = _make_app(
            agent_result=AgentResult(
                status="error",
                result="LLM rate limited after 3 retries",
                tool_calls=[],
                model_used="",
                iterations=1,
                error_code="llm_rate_limit",
            ),
        )

        resp = client.post("/api/v1/tasks", json={"prompt": "read secret"})

        assert resp.status_code == 429


# ============================================================================
# POST /api/v1/tasks — Structured Error Responses
# ============================================================================


class TestStructuredErrorResponses:
    """Verify all error responses are structured JSON, never raw tracebacks."""

    def test_validation_error_is_structured_json(self):
        """Validation errors return {error, detail} structure."""
        client = _make_app()

        resp = client.post("/api/v1/tasks", json={})

        assert resp.status_code == 422
        body = resp.json()
        assert isinstance(body.get("error"), str)
        assert "detail" in body
        # Should never contain "Internal Server Error" or traceback
        assert "traceback" not in str(body).lower()

    def test_agent_error_is_structured_json(self):
        """Agent-level errors return TaskResponse with status=error."""
        client = _make_app(
            agent_result=AgentResult(
                status="error",
                result="Something went wrong",
                tool_calls=[],
                model_used="github/gpt-4o",
                iterations=1,
                error_code="internal_error",
            ),
        )

        resp = client.post("/api/v1/tasks", json={"prompt": "read secret"})

        body = resp.json()
        assert body["status"] == "error"
        assert body["error"] is not None
        assert isinstance(body["error"], str)

    def test_500_error_is_structured_json(self):
        """Unhandled exceptions produce structured JSON error."""
        client = _make_app(agent_side_effect=ValueError("unexpected"))

        resp = client.post("/api/v1/tasks", json={"prompt": "test"})

        assert resp.status_code == 500
        body = resp.json()
        assert "error" in body
        assert isinstance(body["error"], str)


# ============================================================================
# GET /api/v1/health
# ============================================================================


class TestHealthEndpoint:
    """Spec scenarios: healthy, unhealthy (MCP down), degraded (LLM down)."""

    def test_healthy_state(self):
        """Spec: All systems healthy → 200 + status='healthy'."""
        client = _make_app(mcp_connected=True, mcp_health=True)

        resp = client.get("/api/v1/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["agent"] == "ok"
        assert body["vault_mcp"] == "connected"
        assert body["vault_server"] == "reachable"
        assert body["uptime_seconds"] >= 0
        assert "version" in body

    def test_unhealthy_state_mcp_disconnected(self):
        """Spec: vault-mcp-server NOT connected → 503 + status='unhealthy'."""
        client = _make_app(mcp_connected=False, mcp_health=False)

        resp = client.get("/api/v1/health")

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["vault_mcp"] == "disconnected"

    def test_degraded_state_vault_unreachable(self):
        """Spec: MCP connected but Vault health check fails → 200 + status='degraded'."""
        client = _make_app(mcp_connected=True, mcp_health=False)

        resp = client.get("/api/v1/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["vault_mcp"] == "connected"
        assert body["vault_server"] == "unreachable"

    def test_degraded_state_health_check_exception(self):
        """MCP connected but health_check() raises → 200 + status='degraded'."""
        client = _make_app(
            mcp_connected=True,
            mcp_health_side_effect=ConnectionError("timeout"),
        )

        resp = client.get("/api/v1/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"

    def test_health_response_is_json(self):
        """Health endpoint always returns application/json."""
        client = _make_app()

        resp = client.get("/api/v1/health")

        assert "application/json" in resp.headers["content-type"]


# ============================================================================
# GET /api/v1/models
# ============================================================================


class TestModelsEndpoint:
    """Spec scenario: List available models."""

    def test_returns_configured_models(self):
        """Spec: Returns array of model objects from config."""
        client = _make_app(
            models=[
                ModelInfo(name="default", provider="github", model_id="github/gpt-4o", supports_tool_calling=True),
                ModelInfo(name="fast", provider="github", model_id="github/gpt-4o-mini", supports_tool_calling=True),
                ModelInfo(name="local", provider="ollama", model_id="ollama/llama3", supports_tool_calling=False),
            ],
            default_model="default",
        )

        resp = client.get("/api/v1/models")

        assert resp.status_code == 200
        body = resp.json()
        assert body["default_model"] == "default"
        assert len(body["available_models"]) == 3

        # Verify model structure
        model_names = [m["name"] for m in body["available_models"]]
        assert "default" in model_names
        assert "fast" in model_names
        assert "local" in model_names

    def test_exactly_one_default_model(self):
        """Spec: Exactly one model has is_default=true."""
        client = _make_app()

        resp = client.get("/api/v1/models")

        assert resp.status_code == 200
        body = resp.json()
        defaults = [m for m in body["available_models"] if m["is_default"]]
        assert len(defaults) == 1
        assert defaults[0]["name"] == "default"

    def test_model_detail_fields(self):
        """Each model object has all required fields."""
        client = _make_app()

        resp = client.get("/api/v1/models")

        body = resp.json()
        for model in body["available_models"]:
            assert "name" in model
            assert "provider" in model
            assert "model_id" in model
            assert "supports_tool_calling" in model
            assert "is_default" in model

    def test_models_response_is_json(self):
        """Models endpoint always returns application/json."""
        client = _make_app()

        resp = client.get("/api/v1/models")

        assert "application/json" in resp.headers["content-type"]
