"""Shared test fixtures for vault-operator-agent.

Provides:
- Mock LLMProvider (configurable responses with/without tool calls)
- Mock MCPClient (configurable tool results)
- Mock Settings with test defaults
- AgentCore factory for integration-style unit tests
- Sample tool definitions in OpenAI format
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.core import AgentCore
from src.agent.redaction.context import SecretContext
from src.agent.types import AgentResult
from src.config.models import (
    AgentConfig,
    APIConfig,
    LLMConfig,
    LoggingConfig,
    MCPConfig,
    ModelInfo,
    MTLSConfig,
    SchedulerConfig,
)
from src.llm.provider import LLMProvider, LLMResponse, ToolCall, TokenUsage
from src.mcp.client import MCPClient, MCPTool, ToolResult


# ---------------------------------------------------------------------------
# Sample tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

SAMPLE_TOOLS_OPENAI: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "vault_kv_read",
            "description": "Read a secret from a KV v2 secrets engine mount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The secret path"},
                    "mount": {"type": "string", "description": "The mount point"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_kv_write",
            "description": "Write a secret to a KV v2 secrets engine mount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The secret path"},
                    "data": {"type": "object", "description": "Key-value data to write"},
                    "mount": {"type": "string", "description": "The mount point"},
                },
                "required": ["path", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_pki_issue",
            "description": "Issue a new certificate from a PKI secrets engine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "common_name": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["common_name"],
            },
        },
    },
]

SAMPLE_MCP_TOOLS: list[MCPTool] = [
    MCPTool(
        name="vault_kv_read",
        description="Read a secret from a KV v2 secrets engine mount.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "mount": {"type": "string"},
            },
            "required": ["path"],
        },
    ),
    MCPTool(
        name="vault_kv_write",
        description="Write a secret to a KV v2 secrets engine mount.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "data": {"type": "object"},
                "mount": {"type": "string"},
            },
            "required": ["path", "data"],
        },
    ),
    MCPTool(
        name="vault_pki_issue",
        description="Issue a new certificate from a PKI secrets engine.",
        input_schema={
            "type": "object",
            "properties": {
                "common_name": {"type": "string"},
                "role": {"type": "string"},
            },
            "required": ["common_name"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Mock LLMProvider
# ---------------------------------------------------------------------------


class MockLLMProvider:
    """Mock LLMProvider that returns configurable responses.

    Usage::

        provider = MockLLMProvider(responses=[
            LLMResponse(content=None, tool_calls=[...], model="test", usage=None),
            LLMResponse(content="Done!", tool_calls=None, model="test", usage=None),
        ])
    """

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Return the next pre-configured response."""
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "model": model,
        })
        if self._call_index >= len(self._responses):
            raise RuntimeError(
                f"MockLLMProvider exhausted: called {self._call_index + 1} times "
                f"but only {len(self._responses)} responses configured"
            )
        response = self._responses[self._call_index]
        self._call_index += 1
        return response

    def get_available_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                name="default",
                provider="test",
                model_id="test/mock-model",
                supports_tool_calling=True,
            ),
        ]


# ---------------------------------------------------------------------------
# Mock MCPClient
# ---------------------------------------------------------------------------


class MockMCPClient:
    """Mock MCPClient that returns configurable tool results.

    Usage::

        client = MockMCPClient(tool_results={
            "vault_kv_read": ToolResult(content='{"data": {...}}', is_error=False),
        })
    """

    def __init__(
        self,
        tool_results: dict[str, ToolResult] | None = None,
        tools: list[MCPTool] | None = None,
    ) -> None:
        self._tool_results = tool_results or {}
        self._tools = tools or SAMPLE_MCP_TOOLS
        self.calls: list[dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        return True

    def get_tools(self) -> list[MCPTool]:
        return list(self._tools)

    def get_tools_as_openai_format(self) -> list[dict[str, Any]]:
        from src.mcp.client import MCPClient

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": MCPClient._normalise_schema(tool.input_schema),
                },
            }
            for tool in self._tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Return the pre-configured result for the given tool."""
        self.calls.append({"name": name, "arguments": arguments})
        if name in self._tool_results:
            return self._tool_results[name]
        return ToolResult(
            content=json.dumps({"error": f"Unknown tool: {name}"}),
            is_error=True,
        )

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_agent_config() -> AgentConfig:
    """AgentConfig with test-friendly defaults (low max_iterations)."""
    return AgentConfig(
        max_iterations=5,
        system_prompt_path="config/prompts/system.md",
    )


@pytest.fixture
def test_llm_config() -> LLMConfig:
    """LLMConfig with test defaults."""
    return LLMConfig(
        default_model="default",
        request_timeout=10,
        max_retries=1,
        models=[
            ModelInfo(
                name="default",
                provider="test",
                model_id="test/mock-model",
                supports_tool_calling=True,
            ),
            ModelInfo(
                name="no-tools",
                provider="test",
                model_id="test/no-tools-model",
                supports_tool_calling=False,
            ),
        ],
    )


@pytest.fixture
def test_mcp_config() -> MCPConfig:
    """MCPConfig with test defaults."""
    return MCPConfig(
        transport="stdio",
        server_binary="/usr/local/bin/vault-mcp-server",
        vault_addr="http://vault:8200",
        vault_token="test-token",
        tool_timeout=5,
    )


@pytest.fixture
def sample_tools() -> list[dict[str, Any]]:
    """Sample tool definitions in OpenAI format."""
    return SAMPLE_TOOLS_OPENAI


@pytest.fixture
def secret_context() -> SecretContext:
    """Fresh SecretContext for testing."""
    return SecretContext()


@pytest.fixture
def mock_llm_simple() -> MockLLMProvider:
    """MockLLMProvider that returns a single text response (no tool calls)."""
    return MockLLMProvider(responses=[
        LLMResponse(
            content="The secret was read successfully.",
            tool_calls=None,
            model="test/mock-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        ),
    ])


@pytest.fixture
def mock_llm_with_tool_call() -> MockLLMProvider:
    """MockLLMProvider that returns one tool call, then a text response."""
    return MockLLMProvider(responses=[
        # First response: LLM requests a tool call
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_001",
                    name="vault_kv_read",
                    arguments={"path": "secret/myapp/db"},
                ),
            ],
            model="test/mock-model",
            usage=None,
        ),
        # Second response: LLM produces final text
        LLMResponse(
            content="I read the secret at secret/myapp/db. It contains keys: username, password.",
            tool_calls=None,
            model="test/mock-model",
            usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        ),
    ])


@pytest.fixture
def mock_mcp_kv_read() -> MockMCPClient:
    """MockMCPClient that returns a KV read result."""
    return MockMCPClient(
        tool_results={
            "vault_kv_read": ToolResult(
                content=json.dumps({
                    "data": {
                        "username": "admin",
                        "password": "s3cret!123",
                    },
                    "metadata": {
                        "version": 3,
                        "created_time": "2026-01-15T10:00:00Z",
                    },
                }),
                is_error=False,
            ),
        },
    )


@pytest.fixture
def make_agent(test_agent_config):
    """Factory for creating AgentCore instances with mock dependencies."""

    def _make(
        llm: MockLLMProvider | None = None,
        mcp: MockMCPClient | None = None,
        config: AgentConfig | None = None,
    ) -> AgentCore:
        return AgentCore(
            llm_provider=llm or MockLLMProvider(responses=[
                LLMResponse(
                    content="Default response.",
                    tool_calls=None,
                    model="test/mock-model",
                    usage=None,
                ),
            ]),
            mcp_client=mcp or MockMCPClient(),
            config=config or test_agent_config,
            vault_addr="http://vault-test:8200",
        )

    return _make


# ---------------------------------------------------------------------------
# Integration test fixtures (FastAPI TestClient)
# ---------------------------------------------------------------------------

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app


def _build_test_app(
    *,
    agent: AgentCore | MockLLMProvider | None = None,
    mcp_client: MCPClient | MockMCPClient | None = None,
    llm_provider: LLMProvider | MockLLMProvider | None = None,
    mcp_connected: bool = True,
    mcp_healthy: bool = True,
) -> FastAPI:
    """Create a FastAPI app with controllable mocks on app.state.

    For integration tests, the agent is typically a MagicMock so the test
    can control ``agent.execute()`` return values directly.
    """
    app = create_app()

    # Default agent mock
    if agent is None:
        mock_agent = MagicMock(spec=AgentCore)
        mock_agent.execute = AsyncMock(return_value=AgentResult(
            status="completed",
            result="Operation completed successfully.",
            tool_calls=[],
            model_used="github/gpt-4o",
            iterations=1,
        ))
        agent = mock_agent

    # Default MCP client mock
    if mcp_client is None:
        mcp_client = MockMCPClient()
        # Override is_connected via a simple object attribute trick for MagicMock
        mcp_mock = MagicMock(spec=MCPClient)
        type(mcp_mock).is_connected = property(lambda self: mcp_connected)
        mcp_mock.health_check = AsyncMock(return_value=mcp_healthy)
        mcp_mock.get_tools.return_value = SAMPLE_MCP_TOOLS
        mcp_mock.get_tools_as_openai_format.return_value = SAMPLE_TOOLS_OPENAI
        mcp_client = mcp_mock

    # Default LLM provider mock
    if llm_provider is None:
        llm_mock = MagicMock(spec=LLMProvider)
        llm_mock._config = LLMConfig(
            default_model="default",
            models=[
                ModelInfo(
                    name="default",
                    provider="github",
                    model_id="github/gpt-4o",
                    supports_tool_calling=True,
                ),
                ModelInfo(
                    name="fast",
                    provider="github",
                    model_id="github/gpt-4o-mini",
                    supports_tool_calling=True,
                ),
            ],
        )
        llm_mock.get_available_models.return_value = llm_mock._config.models
        llm_provider = llm_mock

    # Wire everything into app.state
    app.state.agent = agent
    app.state.mcp_client = mcp_client
    app.state.llm_provider = llm_provider
    app.state.start_time = time.monotonic()
    app.state.settings = MagicMock()

    return app


@pytest.fixture
def integration_app():
    """Factory fixture that returns a FastAPI app builder for integration tests.

    Usage in tests::

        def test_something(integration_app):
            app = integration_app(mcp_connected=True, mcp_healthy=True)
            client = TestClient(app)
            resp = client.get("/api/v1/health")
    """
    return _build_test_app


@pytest.fixture
def api_client(integration_app):
    """Pre-built TestClient with default healthy mocks.

    For tests that just need a working API without custom mock configuration.
    """
    app = integration_app()
    return TestClient(app)
