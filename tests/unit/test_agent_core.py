"""Unit tests for AgentCore reasoning loop.

Tests cover:
- Single tool call flow (LLM returns tool call → raw result returned directly)
- Multi-step flow (LLM returns multiple tool calls → raw results aggregated)
- Max iterations exceeded
- LLM error handling
- MCP tool error → retry via hybrid loop
- Secret redaction applied in the reasoning loop (LLM never sees secrets)
- Empty LLM response handling
- Text-only response (LLM couldn't map to tool)
- Raw tool result structure

Hybrid Loop Architecture:
    The LLM is ONLY used for INPUT processing (understanding intent, selecting
    tools). After tool execution:
    - All tools succeed → raw results returned directly (fast path, 1 iteration)
    - Any tool fails → error fed back to LLM → LLM retries (new iteration)
    - After max_iterations with only errors → last error returned
    - LLM returns text → text returned to caller
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

    Raw-Response Architecture: The LLM selects a tool, the tool executes,
    and the raw result is returned directly. Only 1 iteration (no second
    LLM call for summarization).
    """

    @pytest.mark.asyncio
    async def test_single_tool_call_completes_in_one_iteration(
        self, make_agent, mock_llm_with_tool_call, mock_mcp_kv_read,
    ) -> None:
        """GIVEN the agent receives a read prompt,
        WHEN the LLM responds with one tool call,
        THEN the loop completes in 1 iteration with status 'completed'
        and returns raw tool results directly."""
        agent: AgentCore = make_agent(llm=mock_llm_with_tool_call, mcp=mock_mcp_kv_read)

        result = await agent.execute(prompt="read kv/myapp/db")

        assert result.status == "completed"
        assert result.iterations == 1
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "vault_kv_read"
        assert result.model_used == "test/mock-model"
        # Raw tool results should be populated
        assert len(result.raw_tool_results) == 1
        assert result.raw_tool_results[0]["tool_name"] == "vault_kv_read"
        assert result.raw_tool_results[0]["is_error"] is False

    @pytest.mark.asyncio
    async def test_single_tool_call_result_contains_raw_vault_data(
        self, make_agent, mock_llm_with_tool_call, mock_mcp_kv_read,
    ) -> None:
        """GIVEN a single tool call flow, WHEN completed,
        THEN result.result contains the raw Vault JSON data (not LLM text)."""
        agent: AgentCore = make_agent(llm=mock_llm_with_tool_call, mcp=mock_mcp_kv_read)
        result = await agent.execute(prompt="read kv/myapp/db")

        # Result should be raw JSON from Vault, containing actual data
        parsed = json.loads(result.result)
        assert "data" in parsed
        assert parsed["data"]["username"] == "admin"
        assert parsed["data"]["password"] == "s3cret!123"

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
    """Test the reasoning loop with multiple tool calls in a single LLM response.

    Raw-Response Architecture: The LLM selects multiple tools in one response,
    all execute, and all raw results are returned in a single iteration.
    """

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_response(self, make_agent) -> None:
        """GIVEN a prompt requiring multiple tool calls,
        WHEN the LLM returns multiple tool calls in one response,
        THEN all tool calls execute and raw results are aggregated."""
        llm = MockLLMProvider(responses=[
            # LLM requests two tool calls in one response
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/app"}),
                    ToolCall(id="call_002", name="vault_kv_write", arguments={
                        "path": "secret/app/copy",
                        "data": {"key": "value"},
                    }),
                ],
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
        assert result.iterations == 1
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "vault_kv_read"
        assert result.tool_calls[1].tool_name == "vault_kv_write"

        # Raw results should contain both tool outputs
        assert len(result.raw_tool_results) == 2
        assert result.raw_tool_results[0]["tool_name"] == "vault_kv_read"
        assert result.raw_tool_results[1]["tool_name"] == "vault_kv_write"

        # Result string for multiple tools should be a JSON array
        parsed = json.loads(result.result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2


# ===================================================================
# Max Iterations Exceeded
# ===================================================================


class TestMaxIterationsExceeded:
    """Test behavior when reasoning loop hits max iterations.

    Hybrid Loop Architecture: max_iterations limits error-retry iterations.
    Successful tool calls always return immediately on the first iteration.
    Max iterations is reached when every iteration results in tool errors and
    the LLM cannot recover.

    See also: TestMCPToolError.test_mcp_error_exhausts_retries for the case
    where retries are exhausted.
    """

    @pytest.mark.asyncio
    async def test_tool_call_returns_immediately_regardless_of_max_iterations(self) -> None:
        """GIVEN max_iterations=2 and the LLM returns a tool call,
        WHEN the tool executes, THEN the result is returned in iteration 1
        (the loop doesn't continue to iteration 2)."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_0", name="vault_kv_read", arguments={"path": "x"})],
                model="test/mock-model",
                usage=None,
            ),
            # This response would be used in iteration 2, but should NOT be reached
            LLMResponse(
                content="This should never be reached.",
                tool_calls=None,
                model="test/mock-model",
                usage=None,
            ),
        ])

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
        assert result.iterations == 1  # Returns immediately after tool call
        assert len(result.tool_calls) == 1
        assert result.raw_tool_results[0]["tool_name"] == "vault_kv_read"
        assert result.warning is None  # No warning — completed normally


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
# MCP Tool Error → Fed Back to LLM for Retry
# ===================================================================


class TestMCPToolError:
    """Test the hybrid loop error retry behavior.

    Hybrid Loop Architecture: When an MCP tool fails, the redacted error is
    fed back to the LLM so it can retry with different parameters or a
    different approach. The loop continues up to max_iterations. If the LLM
    eventually succeeds, the raw result is returned. If all retries fail,
    the last error is returned with status 'error'.
    """

    @pytest.mark.asyncio
    async def test_mcp_error_triggers_retry_and_succeeds(self, make_agent) -> None:
        """GIVEN the MCP tool fails on the first attempt but succeeds on retry,
        WHEN the agent processes the error and retries,
        THEN the successful result is returned with status 'completed'."""
        llm = MockLLMProvider(responses=[
            # Iteration 1: LLM requests tool call → tool fails
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/missing"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            # Iteration 2: LLM retries with a corrected path
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_002", name="vault_kv_read", arguments={"path": "secret/found"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
        ])

        call_count = 0

        class RetryMCPClient(MockMCPClient):
            async def call_tool(self, name: str, arguments: dict) -> ToolResult:
                nonlocal call_count
                call_count += 1
                self.calls.append({"name": name, "arguments": arguments})
                if call_count == 1:
                    raise MCPError("Path not found: secret/missing")
                return ToolResult(
                    content=json.dumps({"data": {"key": "value"}}),
                    is_error=False,
                )

        agent: AgentCore = make_agent(llm=llm, mcp=RetryMCPClient())
        result = await agent.execute(prompt="read secret/missing")

        assert result.status == "completed"
        assert result.iterations == 2  # Took 2 iterations
        assert len(result.tool_calls) == 2  # Both attempts recorded
        assert result.tool_calls[0].is_error is True
        assert result.tool_calls[1].is_error is False
        # Raw result should contain the successful result
        assert len(result.raw_tool_results) == 1
        assert result.raw_tool_results[0]["is_error"] is False
        assert "value" in result.raw_tool_results[0]["result"]
        # LLM was called twice (initial + retry)
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_mcp_error_exhausts_retries(self, make_agent) -> None:
        """GIVEN the MCP tool fails on every attempt,
        WHEN max_iterations is reached, THEN the last error is returned
        with status 'error' and error_code 'max_iterations'."""
        llm = MockLLMProvider(responses=[
            # Iteration 1: tool call → fails
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/missing"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            # Iteration 2: retry → fails again
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_002", name="vault_kv_read", arguments={"path": "secret/other"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
        ])

        class AlwaysErrorMCP(MockMCPClient):
            async def call_tool(self, name: str, arguments: dict) -> ToolResult:
                self.calls.append({"name": name, "arguments": arguments})
                raise MCPError("Path not found")

        config = AgentConfig(max_iterations=2)
        agent: AgentCore = make_agent(llm=llm, mcp=AlwaysErrorMCP(), config=config)
        result = await agent.execute(prompt="read secret/missing")

        assert result.status == "error"
        assert result.error_code == "max_iterations"
        assert result.iterations == 2
        assert len(result.tool_calls) == 2
        assert all(tc.is_error for tc in result.tool_calls)
        # Raw results should contain the last error
        assert len(result.raw_tool_results) == 1
        assert result.raw_tool_results[0]["is_error"] is True
        assert "not found" in result.raw_tool_results[0]["result"].lower()
        assert result.warning is not None
        assert "max iterations" in result.warning.lower()

    @pytest.mark.asyncio
    async def test_mcp_error_message_in_raw_result(self, make_agent) -> None:
        """GIVEN an MCP error with max_iterations=1 (no retries),
        WHEN the error occurs, THEN the error message is returned as the
        last error result after exhausting max_iterations."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "x"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
        ])

        class ErrorMCP(MockMCPClient):
            async def call_tool(self, name: str, arguments: dict) -> ToolResult:
                raise MCPError("permission denied")

        config = AgentConfig(max_iterations=1)
        agent: AgentCore = make_agent(llm=llm, mcp=ErrorMCP(), config=config)
        result = await agent.execute(prompt="read x")

        # The raw result should contain the error
        assert len(result.raw_tool_results) == 1
        raw = result.raw_tool_results[0]
        assert raw["is_error"] is True
        assert "permission denied" in raw["result"]

        # The LLM was called once (no retry since max_iterations=1)
        assert len(llm.calls) == 1
        assert result.status == "error"
        assert result.error_code == "max_iterations"

    @pytest.mark.asyncio
    async def test_llm_gives_up_with_text_after_error(self, make_agent) -> None:
        """GIVEN a tool error, WHEN the LLM decides it cannot retry and
        responds with text instead of a new tool call, THEN the text is
        returned to the caller."""
        llm = MockLLMProvider(responses=[
            # Iteration 1: tool call → fails
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/x"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            # Iteration 2: LLM gives up and responds with text
            LLMResponse(
                content="I could not read the secret because the path does not exist.",
                tool_calls=None,
                model="test/mock-model",
                usage=None,
            ),
        ])

        class ErrorMCP(MockMCPClient):
            async def call_tool(self, name: str, arguments: dict) -> ToolResult:
                raise MCPError("path not found")

        agent: AgentCore = make_agent(llm=llm, mcp=ErrorMCP())
        result = await agent.execute(prompt="read secret/x")

        assert result.status == "completed"
        assert result.iterations == 2
        assert "could not read" in result.result.lower()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].is_error is True

    @pytest.mark.asyncio
    async def test_hybrid_loop_mixed_results_retry_succeeds(self, make_agent) -> None:
        """GIVEN the LLM requests 2 tool calls, one succeeds and one fails,
        WHEN the loop retries and the LLM requests only the failed tool again,
        THEN the final result is successful after 2 iterations.

        Mixed Results (Retry-All Approach): When ANY tool fails in a batch,
        the entire iteration is treated as failed. The errors are fed back to
        the LLM which retries — potentially requesting a different subset of
        tools on the next iteration."""
        llm = MockLLMProvider(responses=[
            # Iteration 1: LLM requests two tool calls — one will succeed, one will fail
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/app"}),
                    ToolCall(id="call_002", name="vault_kv_read", arguments={"path": "secret/missing"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
            # Iteration 2: LLM retries only the failed tool with a corrected path
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_003", name="vault_kv_read", arguments={"path": "secret/found"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
        ])

        call_count = 0

        class MixedResultsMCP(MockMCPClient):
            async def call_tool(self, name: str, arguments: dict) -> ToolResult:
                nonlocal call_count
                call_count += 1
                self.calls.append({"name": name, "arguments": arguments})
                path = arguments.get("path", "")
                if path == "secret/missing":
                    raise MCPError("Path not found: secret/missing")
                # All other paths succeed
                return ToolResult(
                    content=json.dumps({"data": {"key": "value"}}),
                    is_error=False,
                )

        agent: AgentCore = make_agent(llm=llm, mcp=MixedResultsMCP())
        result = await agent.execute(prompt="read secret/app and secret/missing")

        # Final result should be successful (iteration 2 had no errors)
        assert result.status == "completed"
        assert result.iterations == 2

        # Iteration 1 produced 2 tool call records (1 success + 1 error),
        # iteration 2 produced 1 tool call record (success) = 3 total
        assert len(result.tool_calls) == 3
        assert result.tool_calls[0].is_error is False   # call_001: secret/app succeeded
        assert result.tool_calls[1].is_error is True     # call_002: secret/missing failed
        assert result.tool_calls[2].is_error is False    # call_003: secret/found succeeded

        # Raw tool results come from the final (successful) iteration only
        assert len(result.raw_tool_results) == 1
        assert result.raw_tool_results[0]["is_error"] is False
        assert "value" in result.raw_tool_results[0]["result"]

        # LLM was called twice (initial + retry after mixed failure)
        assert len(llm.calls) == 2

        # MCP was called 3 times total (2 in iteration 1 + 1 in iteration 2)
        assert call_count == 3


# ===================================================================
# Secret Redaction Applied in the Loop
# ===================================================================


class TestSecretRedactionInLoop:
    """Verify that secret isolation is maintained in the reasoning loop.

    Hybrid Loop Architecture: The LLM NEVER sees real tool results. On
    success, a minimal acknowledgement (no secrets) is appended to messages.
    On error, a REDACTED error message is appended. The raw Vault data is
    returned directly to the API consumer. The audit trail
    (ToolCallRecord.result) is still redacted for safe logging.
    """

    @pytest.mark.asyncio
    async def test_llm_never_sees_tool_results(self, make_agent) -> None:
        """GIVEN the MCP returns a KV read with secret values,
        WHEN the reasoning loop completes,
        THEN the LLM was only called ONCE (for tool selection) and never
        received tool results in its messages."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/db"}),
                ],
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

        # LLM should have been called exactly once (tool selection only)
        assert len(llm.calls) == 1

        # The LLM's only call should NOT contain any tool results
        first_call = llm.calls[0]
        messages_json = json.dumps(first_call["messages"])
        assert secret_password not in messages_json
        assert "dbuser" not in messages_json

        # But the raw result returned to the caller SHOULD contain real data
        assert secret_password in result.result
        assert "dbuser" in result.result

    @pytest.mark.asyncio
    async def test_raw_result_contains_actual_vault_data(self, make_agent) -> None:
        """GIVEN a tool call returns secret data,
        WHEN the loop completes, THEN raw_tool_results contains the full
        unredacted Vault data for the consumer."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_001", name="vault_kv_read", arguments={"path": "secret/db"}),
                ],
                model="test/mock-model",
                usage=None,
            ),
        ])

        mcp = MockMCPClient(tool_results={
            "vault_kv_read": ToolResult(
                content=json.dumps({"data": {"api_key": "sk-real-key-12345"}}),
                is_error=False,
            ),
        })

        agent: AgentCore = make_agent(llm=llm, mcp=mcp)
        result = await agent.execute(prompt="read secret/db")

        # Raw tool results should contain the real value
        assert len(result.raw_tool_results) == 1
        assert "sk-real-key-12345" in result.raw_tool_results[0]["result"]

        # The result string should also contain the real value
        assert "sk-real-key-12345" in result.result

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


# ===================================================================
# Secret Restoration in Final Result for API Consumer
# ===================================================================


class TestSecretRestorationInResult:
    """Verify placeholder restoration in the text-only path.

    Raw-Response Architecture: When tools are called, raw Vault results are
    returned directly — no placeholder restoration needed. Placeholder
    restoration ONLY applies when the LLM responds with text (no tool calls),
    which is the fallback/error path where the LLM couldn't map to a tool.
    """

    @pytest.mark.asyncio
    async def test_placeholders_in_text_only_result_restored_from_secret_data(self, make_agent) -> None:
        """GIVEN a prompt with secret_data causing placeholder substitution,
        WHEN the LLM responds with text only (no tool calls) and echoes placeholders,
        THEN the result text has real values restored for the API consumer."""
        # The LLM will "see" the placeholder and echo it back
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content="I wrote the password [SECRET_VALUE_1] to kv/prod/db successfully.",
                tool_calls=None,
                model="test/mock-model",
                usage=None,
            ),
        ])

        agent: AgentCore = make_agent(llm=llm)
        result = await agent.execute(
            prompt="write to kv/prod/db",
            secret_data={"password": "TopS3cret!"},
        )

        # The API consumer should see real values, not placeholders
        assert "[SECRET_VALUE_" not in result.result
        assert "TopS3cret!" in result.result

    @pytest.mark.asyncio
    async def test_multiple_placeholders_restored_in_text_only_result(self, make_agent) -> None:
        """GIVEN a prompt with multiple secret_data values,
        WHEN the LLM references multiple placeholders in text-only response,
        THEN all placeholders are restored to real values."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content=(
                    "I wrote username=[SECRET_VALUE_1] and "
                    "password=[SECRET_VALUE_2] to kv/prod/db."
                ),
                tool_calls=None,
                model="test/mock-model",
                usage=None,
            ),
        ])

        agent: AgentCore = make_agent(llm=llm)
        result = await agent.execute(
            prompt="write to kv/prod/db",
            secret_data={"username": "admin", "password": "S3cret!"},
        )

        assert "[SECRET_VALUE_" not in result.result
        assert "admin" in result.result
        assert "S3cret!" in result.result

    @pytest.mark.asyncio
    async def test_result_without_placeholders_unchanged(self, make_agent) -> None:
        """GIVEN a prompt with no secret_data (no placeholders created),
        WHEN the LLM returns plain text, THEN the result is unchanged."""
        llm = MockLLMProvider(responses=[
            LLMResponse(
                content="Listed all secrets under kv/prod/.",
                tool_calls=None,
                model="test/mock-model",
                usage=None,
            ),
        ])

        agent: AgentCore = make_agent(llm=llm)
        result = await agent.execute(prompt="list secrets under kv/prod/")

        assert result.result == "Listed all secrets under kv/prod/."

    @pytest.mark.asyncio
    async def test_tool_call_with_secret_data_returns_raw_vault_result(self, make_agent) -> None:
        """GIVEN a write flow where the prompt had secret_data and the LLM
        calls a tool, WHEN execute completes, THEN the result contains the
        raw Vault response (not LLM text with placeholders).

        Raw-Response Architecture: The tool result goes directly to the caller.
        The LLM is not called again, so there's no text with placeholders to
        restore. The secret_data placeholders were only used in the prompt
        sanitization and tool argument restoration."""
        llm = MockLLMProvider(responses=[
            # LLM calls tool with placeholders in args
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_001",
                        name="vault_kv_write",
                        arguments={
                            "path": "secret/app",
                            "data": {
                                "password": "[SECRET_VALUE_1]",
                            },
                        },
                    ),
                ],
                model="test/mock-model",
                usage=None,
            ),
        ])

        mcp = MockMCPClient(tool_results={
            "vault_kv_write": ToolResult(
                content=json.dumps({"data": {"version": 1}}),
                is_error=False,
            ),
        })

        agent: AgentCore = make_agent(llm=llm, mcp=mcp)
        result = await agent.execute(
            prompt="write to secret/app",
            secret_data={"password": "MyR3alP@ss!"},
        )

        # Result should be raw Vault response (write confirmation)
        parsed = json.loads(result.result)
        assert parsed == {"data": {"version": 1}}

        # The LLM should only have been called once
        assert len(llm.calls) == 1

        # The MCP client should have received the real password
        assert mcp.calls[0]["arguments"]["data"]["password"] == "MyR3alP@ss!"

        # The LLM never saw the real password
        first_call = llm.calls[0]
        messages_json = json.dumps(first_call["messages"])
        assert "MyR3alP@ss!" not in messages_json
