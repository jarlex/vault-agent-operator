"""Unit tests for AgentCore reasoning loop.

Tests cover:
- Single tool call flow (LLM returns tool call, then final text)
- Multi-step flow (multiple tool calls before final response)
- Max iterations exceeded
- LLM error handling
- MCP tool error fed back to LLM
- Secret redaction applied in the reasoning loop
- Empty LLM response handling
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.agent.core import AgentCore
from src.agent.types import AgentResult, ToolCallRecord
from src.config.models import AgentConfig
from src.llm.provider import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMServiceError,
    ToolCall,
    TokenUsage,
)
from src.mcp.client import MCPError, ToolResult

from tests.conftest import MockLLMProvider, MockMCPClient


# ===================================================================
# Single Tool Call Flow
# ===================================================================


class TestSingleToolCallFlow:
    """Test the reasoning loop with a single tool call.

    Spec scenario: "Single tool call (simple read)"
    """

    @pytest.mark.asyncio
    async def test_single_tool_call_completes_in_two_iterations(
        self, make_agent, mock_llm_with_tool_call, mock_mcp_kv_read,
    ) -> None:
        """GIVEN the agent receives a read prompt,
        WHEN the LLM responds with one tool call and then final text,
        THEN the loop completes in 2 iterations with status 'completed'."""
        agent: AgentCore = make_agent(llm=mock_llm_with_tool_call, mcp=mock_mcp_kv_read)

        result = await agent.execute(prompt="read kv/myapp/db")

        assert result.status == "completed"
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "vault_kv_read"
        assert result.model_used == "test/mock-model"

    @pytest.mark.asyncio
    async def test_single_tool_call_result_text(
        self, make_agent, mock_llm_with_tool_call, mock_mcp_kv_read,
    ) -> None:
        """GIVEN a single tool call flow, WHEN completed,
        THEN result.result contains the LLM's final text response."""
        agent: AgentCore = make_agent(llm=mock_llm_with_tool_call, mcp=mock_mcp_kv_read)
        result = await agent.execute(prompt="read kv/myapp/db")
        assert "read the secret" in result.result.lower() or "secret" in result.result.lower()

    @pytest.mark.asyncio
    async def test_tool_call_records_audit_trail(
        self, make_agent, mock_llm_with_tool_call, mock_mcp_kv_read,
    ) -> None:
        """GIVEN a tool call flow, WHEN completed,
        THEN tool_calls contains an audit trail with tool name, arguments, and redacted result."""
        agent: AgentCore = make_agent(llm=mock_llm_with_tool_call, mcp=mock_mcp_kv_read)
        result = await agent.execute(prompt="read kv/myapp/db")

        record = result.tool_calls[0]
        assert record.tool_name == "vault_kv_read"
        assert record.arguments == {"path": "secret/myapp/db"}
        assert record.is_error is False
        assert record.duration_ms >= 0


# ===================================================================
# Multi-Step Flow
# ===================================================================


class TestMultiStepFlow:
    """Test the reasoning loop with multiple sequential tool calls.

    Spec scenario: "Multi-step reasoning (compound operation)"
    """

    @pytest.mark.asyncio
    async def test_multi_step_flow(self, make_agent) -> None:
        """GIVEN a prompt requiring multiple tool calls,
        WHEN the LLM calls two tools sequentially before final text,
        THEN all tool calls are recorded and the final result is returned."""
        llm = MockLLMProvider(responses=[
            # Step 1: LLM requests first tool call
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/app"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            # Step 2: LLM requests second tool call
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_002", name="vault_kv_write", arguments={
                        "path": "secret/app/copy",
                        "data": {"key": "value"},
                    }),
                ],
                model="test/mock-model",
                usage=None,
            ),
            # Step 3: LLM produces final text
            LLMResponse(
                content="I read the secret and wrote a copy successfully.",
                tool_calls=None,
                model="test/mock-model",
                usage=None,
            ),
        ])

        mcp = MockMCPClient(tool_results={
            "vault_kv_read": ToolResult(
                content=json.dumps({"data": {"key": "value"}}),
                is_error=False,
            ),
            "vault_kv_write": ToolResult(
                content=json.dumps({"data": {"version": 1}}),
                is_error=False,
            ),
        })

        agent: AgentCore = make_agent(llm=llm, mcp=mcp)
        result = await agent.execute(prompt="read secret/app and copy to secret/app/copy")

        assert result.status == "completed"
        assert result.iterations == 3
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "vault_kv_read"
        assert result.tool_calls[1].tool_name == "vault_kv_write"


# ===================================================================
# Max Iterations Exceeded
# ===================================================================


class TestMaxIterationsExceeded:
    """Test behavior when reasoning loop hits max iterations.

    Spec scenario: "Maximum iterations exceeded"
    """

    @pytest.mark.asyncio
    async def test_max_iterations_returns_warning(self) -> None:
        """GIVEN max_iterations=2 and the LLM keeps calling tools,
        WHEN the loop hits the limit, THEN result has a warning about max iterations."""
        # LLM always returns tool calls — never produces final text
        endless_responses = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id=f"call_{i}", name="vault_kv_read", arguments={"path": "x"})],
                model="test/mock-model",
                usage=None,
            )
            for i in range(10)
        ]
        llm = MockLLMProvider(responses=endless_responses)

        mcp = MockMCPClient(tool_results={
            "vault_kv_read": ToolResult(
                content=json.dumps({"data": {"k": "v"}}),
                is_error=False,
            ),
        })

        config = AgentConfig(max_iterations=2)
        agent = AgentCore(
            llm_provider=llm,
            mcp_client=mcp,
            config=config,
            vault_addr="http://test:8200",
        )

        result = await agent.execute(prompt="read something")

        assert result.status == "completed"
        assert result.iterations == 2
        assert result.warning is not None
        assert "max iterations" in result.warning.lower()
        assert result.error_code == "max_iterations"


# ===================================================================
# LLM Error Handling
# ===================================================================


class TestLLMErrorHandling:
    """Test how the agent handles LLM errors."""

    @pytest.mark.asyncio
    async def test_llm_error_returns_error_status(self, make_agent) -> None:
        """GIVEN the LLM raises an error, WHEN execute is called,
        THEN result status is 'error' with an appropriate error message."""

        class FailingLLM(MockLLMProvider):
            async def complete(self, messages, tools=None, model=None):
                raise LLMError("Connection refused")

        agent: AgentCore = make_agent(llm=FailingLLM())
        result = await agent.execute(prompt="test prompt")

        assert result.status == "error"
        assert "LLM error" in result.result
        assert result.error_code == "llm_error"

    @pytest.mark.asyncio
    async def test_llm_auth_error_classified(self, make_agent) -> None:
        """GIVEN the LLM raises an auth error, WHEN execute is called,
        THEN the error is classified as llm_auth."""

        class AuthFailLLM(MockLLMProvider):
            async def complete(self, messages, tools=None, model=None):
                raise LLMAuthError("Invalid token")

        agent: AgentCore = make_agent(llm=AuthFailLLM())
        result = await agent.execute(prompt="test")

        assert result.status == "error"
        assert result.error_code == "llm_auth"

    @pytest.mark.asyncio
    async def test_llm_rate_limit_error_classified(self, make_agent) -> None:
        """GIVEN the LLM raises a rate limit error, WHEN execute is called,
        THEN the error is classified as llm_rate_limit."""

        class RateLimitLLM(MockLLMProvider):
            async def complete(self, messages, tools=None, model=None):
                raise LLMRateLimitError("Rate limited")

        agent: AgentCore = make_agent(llm=RateLimitLLM())
        result = await agent.execute(prompt="test")

        assert result.status == "error"
        assert result.error_code == "llm_rate_limit"

    @pytest.mark.asyncio
    async def test_llm_service_error_classified(self, make_agent) -> None:
        """GIVEN the LLM raises a service error, WHEN execute is called,
        THEN the error is classified as llm_service."""

        class ServiceFailLLM(MockLLMProvider):
            async def complete(self, messages, tools=None, model=None):
                raise LLMServiceError("Service unavailable")

        agent: AgentCore = make_agent(llm=ServiceFailLLM())
        result = await agent.execute(prompt="test")

        assert result.status == "error"
        assert result.error_code == "llm_service"


# ===================================================================
# Empty LLM Response
# ===================================================================


class TestEmptyLLMResponse:
    """Test handling of empty LLM responses.

    Spec scenario: "LLM returns no tool calls and no text"
    """

    @pytest.mark.asyncio
    async def test_empty_response_returns_error(self, make_agent) -> None:
        """GIVEN the LLM returns no content and no tool calls,
        WHEN execute is called, THEN result status is 'error' with appropriate message."""
        llm = MockLLMProvider(responses=[
            LLMResponse(content=None, tool_calls=None, model="test/mock-model", usage=None),
        ])

        agent: AgentCore = make_agent(llm=llm)
        result = await agent.execute(prompt="test")

        assert result.status == "error"
        assert "empty response" in result.result.lower()
        assert result.error_code == "empty_response"


# ===================================================================
# MCP Tool Error → Fed Back to LLM
# ===================================================================


class TestMCPToolError:
    """Test that MCP tool errors are fed back to the LLM for reasoning.

    Spec scenario: "MCP tool returns an error"
    """

    @pytest.mark.asyncio
    async def test_mcp_error_fed_to_llm_then_final_response(self, make_agent) -> None:
        """GIVEN the MCP tool raises an error,
        WHEN the agent processes it, THEN the error is fed back to the LLM
        AND the LLM can produce a final text response about the error."""
        llm = MockLLMProvider(responses=[
            # LLM requests tool call
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/missing"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            # After receiving the error, LLM produces final text
            LLMResponse(
                content="The secret at secret/missing was not found.",
                tool_calls=None,
                model="test/mock-model",
                usage=None,
            ),
        ])

        class ErrorMCPClient(MockMCPClient):
            async def call_tool(self, name: str, arguments: dict) -> ToolResult:
                self.calls.append({"name": name, "arguments": arguments})
                raise MCPError("Path not found: secret/missing")

        agent: AgentCore = make_agent(llm=llm, mcp=ErrorMCPClient())
        result = await agent.execute(prompt="read secret/missing")

        assert result.status == "completed"
        assert "not found" in result.result.lower()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].is_error is True

    @pytest.mark.asyncio
    async def test_mcp_error_message_in_tool_result(self, make_agent) -> None:
        """GIVEN an MCP error, WHEN fed back to LLM,
        THEN the tool result message in the conversation contains the error."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "x"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            LLMResponse(content="Error occurred.", tool_calls=None, model="test/mock-model", usage=None),
        ])

        class ErrorMCP(MockMCPClient):
            async def call_tool(self, name: str, arguments: dict) -> ToolResult:
                raise MCPError("permission denied")

        agent: AgentCore = make_agent(llm=llm, mcp=ErrorMCP())
        result = await agent.execute(prompt="read x")

        # Check the LLM received the error in messages
        second_call = llm.calls[1]
        messages = second_call["messages"]
        # Find the tool result message
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "permission denied" in tool_messages[0]["content"]


# ===================================================================
# Secret Redaction Applied in the Loop
# ===================================================================


class TestSecretRedactionInLoop:
    """Verify that secret redaction is applied at every stage of the reasoning loop."""

    @pytest.mark.asyncio
    async def test_tool_result_redacted_before_llm_sees_it(self, make_agent) -> None:
        """GIVEN the MCP returns a KV read with secret values,
        WHEN the redacted result is fed to the LLM (in messages),
        THEN the actual secret values do NOT appear in the LLM's messages."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/db"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            LLMResponse(
                content="Read the secret successfully.",
                tool_calls=None,
                model="test/mock-model",
                usage=None,
            ),
        ])

        secret_password = "V3ryS3cr3tP@ssw0rd!"
        mcp = MockMCPClient(tool_results={
            "vault_kv_read": ToolResult(
                content=json.dumps({
                    "data": {"password": secret_password, "username": "dbuser"},
                    "metadata": {"version": 1},
                }),
                is_error=False,
            ),
        })

        agent: AgentCore = make_agent(llm=llm, mcp=mcp)
        result = await agent.execute(prompt="read secret/db")

        # Check the messages sent to the LLM on the second call
        second_call = llm.calls[1]
        messages_json = json.dumps(second_call["messages"])

        # The secret password must NOT appear in any message sent to LLM
        assert secret_password not in messages_json, (
            f"Secret value '{secret_password}' leaked into LLM messages!"
        )

        # But key names SHOULD appear (they're safe metadata)
        assert "password" in messages_json or "keys" in messages_json

    @pytest.mark.asyncio
    async def test_unredacted_data_available_for_consumer(self, make_agent) -> None:
        """GIVEN a tool call returns secret data,
        WHEN the loop completes, THEN unredacted_responses contains the full data for the consumer."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/db"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            LLMResponse(content="Done.", tool_calls=None, model="test/mock-model", usage=None),
        ])

        mcp = MockMCPClient(tool_results={
            "vault_kv_read": ToolResult(
                content=json.dumps({"data": {"api_key": "sk-real-key-12345"}}),
                is_error=False,
            ),
        })

        agent: AgentCore = make_agent(llm=llm, mcp=mcp)
        result = await agent.execute(prompt="read secret/db")

        assert len(result.unredacted_responses) >= 1
        # The consumer should be able to find the real value
        found = any(
            "sk-real-key-12345" in json.dumps(r["response"])
            for r in result.unredacted_responses
        )
        assert found, "Consumer should get unredacted data"

    @pytest.mark.asyncio
    async def test_prompt_with_secret_data_is_sanitized(self, make_agent) -> None:
        """GIVEN a prompt with secret_data, WHEN execute is called,
        THEN the sanitized prompt sent to the LLM contains placeholders, not real values."""
        llm = MockLLMProvider(responses=[
            LLMResponse(content="Written.", tool_calls=None, model="test/mock-model", usage=None),
        ])

        agent: AgentCore = make_agent(llm=llm)
        result = await agent.execute(
            prompt="write to kv/prod/db",
            secret_data={"password": "TopS3cret!"},
        )

        # Check the first LLM call's messages
        first_call = llm.calls[0]
        user_message = next(m for m in first_call["messages"] if m["role"] == "user")

        # The real secret value must NOT be in the user message
        assert "TopS3cret!" not in user_message["content"]
        # A placeholder should be present
        assert "[SECRET_VALUE_" in user_message["content"]

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_error_result(self, make_agent) -> None:
        """GIVEN an unexpected exception during execution,
        WHEN execute is called, THEN result status is 'error' and no crash."""

        class BrokenLLM(MockLLMProvider):
            async def complete(self, messages, tools=None, model=None):
                raise RuntimeError("Unexpected internal error")

        agent: AgentCore = make_agent(llm=BrokenLLM())
        result = await agent.execute(prompt="test")

        assert result.status == "error"
        assert "unexpected error" in result.result.lower()
