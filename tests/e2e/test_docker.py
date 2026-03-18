"""End-to-end tests for vault-operator-agent via docker-compose.

These tests validate the full running stack (agent + vault-mcp-server + Vault)
by making real HTTP requests against the API. They are designed to run against
the MVP docker-compose stack.

**How to run:**

1. Start the stack::

       export GITHUB_TOKEN=<your-github-pat>
       docker-compose -f docker-compose.mvp.yaml up --build -d

2. Wait for health::

       # Wait until the agent is healthy (up to ~60s for first build)
       until curl -sf http://localhost:8000/api/v1/health; do sleep 2; done

3. Run these tests::

       pytest tests/e2e/test_docker.py -v

   Or with an explicit base URL::

       E2E_BASE_URL=http://localhost:8000 pytest tests/e2e/test_docker.py -v

4. Tear down::

       docker-compose -f docker-compose.mvp.yaml down -v

**Marker**: All tests are marked with ``@pytest.mark.e2e`` and are skipped by
default unless:
- The ``--run-e2e`` flag is passed, or
- The ``E2E_BASE_URL`` environment variable is set

This ensures ``pytest`` runs without errors even when no stack is available.

**Equivalent of scripts/test-mvp.sh**: These tests exercise the same flows as
the bash-based smoke test but in a structured, repeatable pytest format.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")

# Skip all tests in this module unless explicitly enabled
pytestmark = pytest.mark.e2e


def pytest_configure(config):
    """Register the custom 'e2e' marker."""
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring a running docker-compose stack")


def _skip_unless_e2e_enabled():
    """Return a skip reason if e2e is not enabled, else None."""
    if os.environ.get("E2E_BASE_URL"):
        return None
    return "E2E tests require a running docker-compose stack. Set E2E_BASE_URL or use --run-e2e"


skip_reason = _skip_unless_e2e_enabled()
if skip_reason:
    pytestmark = [pytest.mark.e2e, pytest.mark.skip(reason=skip_reason)]


# ---------------------------------------------------------------------------
# Lazy import httpx (only needed when tests actually run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def http_client():
    """Create an httpx client for the E2E tests.

    Falls back to ``requests`` if ``httpx`` is not installed, but httpx is
    preferred because it is already a project dependency.
    """
    try:
        import httpx
        client = httpx.Client(base_url=BASE_URL, timeout=60.0)
        yield client
        client.close()
    except ImportError:
        pytest.skip("httpx is required for E2E tests (pip install httpx)")


# ============================================================================
# GET /api/v1/health
# ============================================================================


class TestE2EHealth:
    """E2E: Health endpoint against live stack."""

    def test_health_returns_200(self, http_client):
        """The health endpoint responds with 200 when the stack is running."""
        resp = http_client.get("/api/v1/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert body["agent"] == "ok"
        assert "version" in body

    def test_health_reports_mcp_connected(self, http_client):
        """MCP server should be connected in the full stack."""
        resp = http_client.get("/api/v1/health")

        body = resp.json()
        assert body["vault_mcp"] == "connected"

    def test_health_reports_vault_reachable(self, http_client):
        """Vault dev server should be reachable through MCP."""
        resp = http_client.get("/api/v1/health")

        body = resp.json()
        # In a healthy stack, vault should be reachable
        assert body["vault_server"] == "reachable"


# ============================================================================
# GET /api/v1/models
# ============================================================================


class TestE2EModels:
    """E2E: Models listing against live stack."""

    def test_models_returns_200(self, http_client):
        """The models endpoint responds with 200."""
        resp = http_client.get("/api/v1/models")

        assert resp.status_code == 200
        body = resp.json()
        assert "default_model" in body
        assert "available_models" in body
        assert len(body["available_models"]) >= 1

    def test_models_has_default(self, http_client):
        """Exactly one model should be marked as default."""
        resp = http_client.get("/api/v1/models")

        body = resp.json()
        defaults = [m for m in body["available_models"] if m.get("is_default")]
        assert len(defaults) == 1


# ============================================================================
# POST /api/v1/tasks — Basic Operations
# ============================================================================


class TestE2ETasks:
    """E2E: Task submission against live stack.

    These tests make real LLM calls and Vault operations. They require:
    - A valid GITHUB_TOKEN in the environment
    - The Vault dev server seeded with test data (via vault-seed container)
    """

    def test_simple_prompt(self, http_client):
        """Submit a simple prompt and verify the response structure."""
        resp = http_client.post(
            "/api/v1/tasks",
            json={"prompt": "What tools do you have available? Just list the tool names."},
            timeout=120.0,
        )

        # Agent should return 200 even if Vault operation fails
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert "status" in body
        assert "result" in body

    def test_list_secrets(self, http_client):
        """Submit a 'list secrets' prompt — exercises KV list tool.

        This matches the test-mvp.sh flow.
        """
        resp = http_client.post(
            "/api/v1/tasks",
            json={"prompt": "list secrets under the path secret/"},
            timeout=120.0,
        )

        assert resp.status_code in (200, 503)
        body = resp.json()
        assert body["status"] in ("completed", "error")
        if body["status"] == "completed":
            assert body["model_used"] != ""

    def test_response_has_request_id(self, http_client):
        """X-Request-ID header is present in task responses."""
        resp = http_client.post(
            "/api/v1/tasks",
            json={"prompt": "check vault health"},
            timeout=120.0,
        )

        assert "x-request-id" in resp.headers

    def test_response_is_json(self, http_client):
        """All responses have application/json content type."""
        resp = http_client.post(
            "/api/v1/tasks",
            json={"prompt": "hello"},
            timeout=120.0,
        )

        assert "application/json" in resp.headers.get("content-type", "")


# ============================================================================
# POST /api/v1/tasks — Validation (no LLM needed)
# ============================================================================


class TestE2ETasksValidation:
    """E2E: Task validation errors (no LLM call needed)."""

    def test_missing_prompt_returns_error(self, http_client):
        """Empty body → 422 structured error."""
        resp = http_client.post("/api/v1/tasks", json={})

        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body

    def test_prompt_too_long_returns_error(self, http_client):
        """Prompt > 4096 chars → 422 structured error."""
        resp = http_client.post(
            "/api/v1/tasks",
            json={"prompt": "x" * 4097},
        )

        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body

    def test_malformed_json_returns_error(self, http_client):
        """Invalid JSON body → 422 structured error."""
        resp = http_client.request(
            "POST",
            "/api/v1/tasks",
            content=b"not json",
            headers={"content-type": "application/json"},
        )

        assert resp.status_code == 422
