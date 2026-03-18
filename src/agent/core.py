"""Agent Core — reasoning loop and tool dispatch for vault-operator-agent.

This module implements the central agent logic: it takes a natural-language
prompt, runs an iterative reasoning loop against an LLM, dispatches tool calls
to the MCP server, and returns a structured result.

Security Model (CRITICAL):
    The agent enforces a strict secret-value isolation protocol at every stage
    of the reasoning loop:

    1. **User prompt → PromptSanitizer → LLM**: Secret values in the consumer's
       prompt are replaced with opaque placeholders BEFORE the LLM sees them.

    2. **LLM tool call args → SecretRedactor.restore_placeholders → MCP server**:
       When the LLM generates a tool call containing placeholders, real values
       are substituted back BEFORE the call reaches the MCP server.

    3. **MCP result → SecretRedactor.redact_tool_result → LLM**: Secret values
       in MCP responses are stripped BEFORE the result enters the LLM conversation.
       Only metadata, key names, and paths are shown to the LLM.

    4. **MCP result (unredacted) → SecretContext → API consumer**: The full,
       unredacted tool response is stored in the SecretContext and included in
       the AgentResult for the API consumer who explicitly requested the data.

    The LLM NEVER sees real secret values at ANY point in the loop.

Usage::

    from src.agent.core import AgentCore
    from src.llm.provider import LLMProvider
    from src.mcp.client import MCPClient
    from src.config.models import AgentConfig

    agent = AgentCore(
        llm_provider=llm_provider,
        mcp_client=mcp_client,
        config=AgentConfig(),
    )
    result = await agent.execute(
        prompt="read the secret at kv/myapp/database",
        secret_data={"password": "SuperS3cret!"},
    )
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.agent.prompts import load_system_prompt
from src.agent.redaction import PromptSanitizer, SecretContext, SecretRedactor
from src.agent.types import AgentResult, ToolCallRecord
from src.config.models import AgentConfig
from src.llm.provider import (
    LLMError,
    LLMProvider,
    LLMResponse,
    ToolCall,
)
from src.logging import get_logger
from src.mcp.client import MCPClient, MCPError, ToolResult

logger = get_logger(__name__)


class AgentCore:
    """Vault Operator Agent — reasoning loop with secret-safe tool dispatch.

    The AgentCore orchestrates a multi-turn conversation with an LLM, forwarding
    tool calls to the vault-mcp-server via the MCPClient, and enforcing secret
    value isolation at every stage.

    Parameters
    ----------
    llm_provider:
        The LLM provider for chat completions (wraps LiteLLM).
    mcp_client:
        The MCP client connected to the vault-mcp-server.
    config:
        Agent configuration (max_iterations, system_prompt_path).
    vault_addr:
        Vault server address for system prompt templating.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        mcp_client: MCPClient,
        config: AgentConfig,
        vault_addr: str = "",
    ) -> None:
        self._llm = llm_provider
        self._mcp = mcp_client
        self._config = config
        self._vault_addr = vault_addr
        self._sanitizer = PromptSanitizer()
        self._redactor = SecretRedactor()

    async def execute(
        self,
        prompt: str,
        model: str | None = None,
        secret_data: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Execute the agent reasoning loop for a given prompt.

        This is the main entry point for processing a consumer request. It:

        1. Creates a request-scoped SecretContext.
        2. Sanitizes the prompt (replaces secret values with placeholders).
        3. Builds the initial message list (system prompt + sanitized user prompt).
        4. Retrieves available tools from the MCP client.
        5. Runs the reasoning loop: LLM completion → tool dispatch → repeat.
        6. Returns a structured AgentResult with redacted audit trail and
           unredacted data for the API consumer.

        Parameters
        ----------
        prompt:
            The consumer's natural language prompt.
        model:
            Optional model alias override. When ``None``, the LLM provider's
            default model is used.
        secret_data:
            Optional structured secret data from the consumer. When provided,
            ALL values are treated as secrets and replaced with placeholders.

        Returns
        -------
        AgentResult
            The complete execution result including status, text response,
            tool call audit trail, and unredacted secret data.
        """
        start_time = time.monotonic()

        logger.info(
            "agent.execute.start",
            prompt_length=len(prompt),
            has_secret_data=secret_data is not None,
            model_override=model,
        )

        # Use SecretContext as a context manager to ensure cleanup
        with SecretContext() as ctx:
            try:
                result = await self._run_loop(prompt, model, secret_data, ctx)
            except Exception as exc:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                logger.error(
                    "agent.execute.unhandled_error",
                    error=str(exc),
                    exc_type=type(exc).__name__,
                    duration_ms=duration_ms,
                )
                return AgentResult(
                    status="error",
                    result=f"Agent encountered an unexpected error: {type(exc).__name__}",
                    model_used=model or "",
                    iterations=0,
                    error_code="internal_error",
                )

        duration_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(
            "agent.execute.done",
            status=result.status,
            iterations=result.iterations,
            tool_call_count=len(result.tool_calls),
            model_used=result.model_used,
            duration_ms=duration_ms,
            warning=result.warning,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: Reasoning Loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        prompt: str,
        model: str | None,
        secret_data: dict[str, Any] | None,
        ctx: SecretContext,
    ) -> AgentResult:
        """Core reasoning loop implementation.

        This method contains the actual loop logic, separated from ``execute()``
        for cleaner error handling and SecretContext lifecycle management.
        """
        # Step 1: Sanitize the user prompt
        sanitized_prompt = self._sanitizer.sanitize_prompt(
            prompt=prompt,
            secret_data=secret_data,
            context=ctx,
        )

        # Step 2: Get tools from MCP client in OpenAI format
        tools = self._mcp.get_tools_as_openai_format()
        if not tools:
            logger.warning("agent.no_tools", message="No MCP tools available")

        # Step 3: Build system prompt with dynamic context
        try:
            system_prompt = load_system_prompt(
                prompt_path=self._config.system_prompt_path,
                vault_addr=self._vault_addr,
                available_tools=tools,
            )
        except FileNotFoundError:
            logger.warning(
                "agent.system_prompt_missing",
                path=self._config.system_prompt_path,
            )
            system_prompt = (
                "You are a Vault operator agent. Use the available tools to "
                "execute Vault operations. Secret values in tool results are "
                "redacted — reference secrets by their key names and paths only."
            )

        # Step 4: Build initial messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sanitized_prompt},
        ]

        # Step 5: Reasoning loop
        tool_call_log: list[ToolCallRecord] = []
        model_used = ""
        max_iterations = self._config.max_iterations
        llm_response: LLMResponse | None = None

        for iteration in range(1, max_iterations + 1):
            logger.info(
                "agent.loop.iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                message_count=len(messages),
            )

            # Call LLM
            try:
                llm_response = await self._llm.complete(
                    messages=messages,
                    tools=tools if tools else None,
                    model=model,
                )
            except LLMError as exc:
                logger.error(
                    "agent.llm_error",
                    iteration=iteration,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
                return AgentResult(
                    status="error",
                    result=f"LLM error: {exc}",
                    tool_calls=tool_call_log,
                    model_used=model_used or (model or ""),
                    iterations=iteration,
                    unredacted_responses=ctx.get_unredacted_responses(),
                    error_code=self._classify_llm_error(exc),
                )

            model_used = llm_response.model

            # Check for tool calls
            if llm_response.tool_calls:
                # Process each tool call
                new_records = await self._dispatch_tool_calls(
                    llm_response=llm_response,
                    messages=messages,
                    ctx=ctx,
                )
                tool_call_log.extend(new_records)

            elif llm_response.content:
                # Final text response — reasoning loop is done
                return AgentResult(
                    status="completed",
                    result=llm_response.content,
                    tool_calls=tool_call_log,
                    model_used=model_used,
                    iterations=iteration,
                    unredacted_responses=ctx.get_unredacted_responses(),
                )

            else:
                # No tool calls AND no content — unexpected LLM behavior
                logger.warning(
                    "agent.empty_response",
                    iteration=iteration,
                    model=model_used,
                )
                return AgentResult(
                    status="error",
                    result="LLM returned an empty response (no tool calls and no text content).",
                    tool_calls=tool_call_log,
                    model_used=model_used,
                    iterations=iteration,
                    unredacted_responses=ctx.get_unredacted_responses(),
                    error_code="empty_response",
                )

        # Max iterations reached without a final text response
        logger.warning(
            "agent.max_iterations",
            max_iterations=max_iterations,
            tool_call_count=len(tool_call_log),
        )

        # Try to extract any partial content from the last response
        fallback_msg = (
            "The agent reached the maximum number of reasoning iterations "
            f"({max_iterations}) without producing a final response."
        )
        last_content = (
            llm_response.content
            if llm_response is not None and llm_response.content
            else fallback_msg
        )

        return AgentResult(
            status="completed",
            result=last_content,
            tool_calls=tool_call_log,
            model_used=model_used,
            iterations=max_iterations,
            unredacted_responses=ctx.get_unredacted_responses(),
            warning=f"Max iterations ({max_iterations}) reached",
            error_code="max_iterations",
        )

    # ------------------------------------------------------------------
    # Internal: Tool Call Dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool_calls(
        self,
        llm_response: LLMResponse,
        messages: list[dict[str, Any]],
        ctx: SecretContext,
    ) -> list[ToolCallRecord]:
        """Process tool calls from an LLM response.

        For each tool call:
        1. Restore placeholder tokens to real values in arguments.
        2. Call the MCP tool with real arguments.
        3. Redact the tool result for the LLM.
        4. Store the unredacted result in SecretContext.
        5. Append the tool call and redacted result to the message list.
        6. Record the call in the audit trail (with redacted arguments).

        Parameters
        ----------
        llm_response:
            The LLM response containing tool calls.
        messages:
            The mutable message list — tool calls and results are appended.
        ctx:
            The request-scoped SecretContext.

        Returns
        -------
        list[ToolCallRecord]
            Audit trail entries for each tool invocation.
        """
        records: list[ToolCallRecord] = []

        if not llm_response.tool_calls:
            return records

        # Append the assistant message with tool calls to the conversation
        # This is required by the OpenAI API format: the assistant message
        # must contain the tool_calls, and each tool result must reference
        # the tool call ID.
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": llm_response.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in llm_response.tool_calls
            ],
        }
        messages.append(assistant_message)

        for tool_call in llm_response.tool_calls:
            record = await self._execute_single_tool(tool_call, messages, ctx)
            records.append(record)

        return records

    async def _execute_single_tool(
        self,
        tool_call: ToolCall,
        messages: list[dict[str, Any]],
        ctx: SecretContext,
    ) -> ToolCallRecord:
        """Execute a single MCP tool call with full security flow.

        Security flow:
            1. LLM tool call args (may contain placeholders)
            2. → restore_placeholders → real values
            3. → MCP server (receives real values)
            4. → MCP result (may contain secrets)
            5. → redact_tool_result → redacted for LLM (stored in messages)
            6. → unredacted stored in SecretContext (for API consumer)

        Parameters
        ----------
        tool_call:
            The tool call from the LLM response.
        messages:
            The mutable message list — the tool result is appended.
        ctx:
            The request-scoped SecretContext.

        Returns
        -------
        ToolCallRecord
            Audit trail entry with redacted arguments and result.
        """
        tool_name = tool_call.name
        redacted_arguments = tool_call.arguments  # These are what the LLM produced (safe for logging)
        start_time = time.monotonic()

        logger.info(
            "agent.tool_call.start",
            tool_name=tool_name,
            tool_call_id=tool_call.id,
        )

        # Step 1: Restore placeholders → real values in arguments
        real_arguments = self._redactor.restore_placeholders(
            arguments=tool_call.arguments,
            context=ctx,
        )

        # Step 2: Call MCP tool with real arguments
        try:
            mcp_result: ToolResult = await self._mcp.call_tool(
                name=tool_name,
                arguments=real_arguments,
            )
        except MCPError as exc:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "agent.tool_call.mcp_error",
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                error=str(exc),
                exc_type=type(exc).__name__,
                duration_ms=duration_ms,
            )

            # Report the error to the LLM for reasoning
            # Redact any secret values that may appear in the error message
            safe_error = self._redactor.redact_error_message(str(exc), ctx)
            error_result = json.dumps({
                "error": True,
                "message": safe_error,
                "tool_name": tool_name,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": error_result,
            })

            return ToolCallRecord(
                tool_name=tool_name,
                arguments=redacted_arguments,
                result=error_result,
                is_error=True,
                duration_ms=duration_ms,
            )

        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Step 3: Redact the tool result for the LLM
        # This also stores the unredacted result in SecretContext
        redacted_result = self._redactor.redact_tool_result(
            tool_name=tool_name,
            result=mcp_result.content,
            context=ctx,
        )

        # Step 4: Append redacted result to messages for the LLM
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": redacted_result,
        })

        logger.info(
            "agent.tool_call.done",
            tool_name=tool_name,
            tool_call_id=tool_call.id,
            is_error=mcp_result.is_error,
            duration_ms=duration_ms,
        )

        return ToolCallRecord(
            tool_name=tool_name,
            arguments=redacted_arguments,
            result=redacted_result,
            is_error=mcp_result.is_error,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Internal: Error Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_llm_error(exc: LLMError) -> str:
        """Classify an LLM error into a machine-readable error code.

        Parameters
        ----------
        exc:
            The LLM exception.

        Returns
        -------
        str
            One of: ``"llm_auth"``, ``"llm_rate_limit"``, ``"llm_service"``,
            ``"llm_tool_unsupported"``, ``"llm_error"``.
        """
        from src.llm.provider import (
            LLMAuthError,
            LLMRateLimitError,
            LLMServiceError,
            LLMToolCallUnsupportedError,
        )

        if isinstance(exc, LLMAuthError):
            return "llm_auth"
        if isinstance(exc, LLMRateLimitError):
            return "llm_rate_limit"
        if isinstance(exc, LLMServiceError):
            return "llm_service"
        if isinstance(exc, LLMToolCallUnsupportedError):
            return "llm_tool_unsupported"
        return "llm_error"
