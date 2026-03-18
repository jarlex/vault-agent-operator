"""Agent Core — hybrid reasoning loop and tool dispatch for vault-operator-agent.

This module implements the central agent logic: it takes a natural-language
prompt, runs a hybrid reasoning loop against an LLM, dispatches tool calls
to the MCP server, and returns a structured result.

**Hybrid Loop Architecture**:
    The loop combines a fast single-pass path with error-retry capability:

    - **All tools succeed** → Return raw Vault results directly to the API
      caller WITHOUT a second LLM call (fast/happy path).
    - **Any tool fails** → Feed the redacted error back to the LLM → LLM
      retries with different parameters or a different approach → loop
      continues up to ``max_iterations``.
    - **After max_iterations with only errors** → Return the last error to
      the caller.
    - **LLM returns text (no tool calls)** → Return text to caller directly.
    - **LLM returns empty** → Return error to caller.

Security Model (CRITICAL):
    The agent enforces a strict secret-value isolation protocol at every stage
    of the reasoning loop:

    1. **User prompt → PromptSanitizer → LLM**: Secret values in the consumer's
       prompt are replaced with opaque placeholders BEFORE the LLM sees them.

    2. **LLM tool call args → SecretRedactor.restore_placeholders → MCP server**:
       When the LLM generates a tool call containing placeholders, real values
       are substituted back BEFORE the call reaches the MCP server.

    3. **MCP result → API consumer**: Raw MCP tool results are returned directly
       to the API consumer. The LLM NEVER sees successful tool results.

    4. **MCP error → redacted → LLM**: When a tool fails, the error message
       is REDACTED before being fed back to the LLM for retry reasoning.

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

    **Stateless Design (CRITICAL)**:
        Each call to ``execute()`` is completely independent — no conversation
        history, tool results, or user context persists between calls. The
        instance holds only reusable infrastructure (LLM provider, MCP client,
        config, sanitizer, redactor — all of which are stateless). Per-request
        state (messages, SecretContext, tool call log) is created fresh in each
        ``execute()`` call and destroyed at the end.

    **API-Executor Behavior**:
        The agent operates as an API executor, not a chatbot. The system prompt
        instructs the LLM to never ask clarifying questions and to always
        attempt to execute operations immediately. This is enforced via the
        system prompt at ``config/prompts/system.md``.

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
        5. Calls the LLM to interpret the prompt and select tool(s).
        6. Executes the selected tool(s) against Vault.
        7. Returns the raw tool results directly (no LLM summarization).

        **Raw-Response Architecture**:
            The LLM is only used for INPUT processing (steps 2-5). After tools
            execute, their raw results go directly to the API caller without
            passing through the LLM again. If the LLM responds with text instead
            of tool calls (meaning it couldn't map the request to a tool), that
            text is returned as-is.

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
            The complete execution result including status, raw tool results,
            and tool call audit trail.
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

            # Restore placeholder tokens in the result text ONLY for the
            # text-only path (when the LLM responded with text instead of
            # tool calls).  In the raw-response path, raw_tool_results
            # already contain the actual Vault data (they were never
            # redacted).  The text path is the fallback when no tools were
            # called — the LLM may have echoed placeholders in its text.
            if not result.raw_tool_results and ctx.has_placeholders():
                result.result = ctx.resolve_all_placeholders(result.result)

        duration_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(
            "agent.execute.done",
            status=result.status,
            iterations=result.iterations,
            tool_call_count=len(result.tool_calls),
            raw_tool_result_count=len(result.raw_tool_results),
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

        **Raw-Response Architecture**:
            The loop calls the LLM once to interpret the prompt and select
            tool(s). After tools execute, their raw results are returned
            directly to the caller — the LLM is NOT called again for
            summarization.

            If the LLM responds with text instead of tool calls (it couldn't
            map the request to a tool), that text is returned as-is.

            The loop supports multiple iterations ONLY for the case where
            tool execution fails and the error is fed back to the LLM for
            a retry with different parameters.
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
                "You are a Vault operator agent that acts as an API executor. "
                "Use the available tools to execute Vault operations immediately. "
                "NEVER ask clarifying questions — always attempt to execute. "
                "Secret values in tool results are redacted — reference secrets "
                "by their key names and paths only. If you cannot fulfill the "
                "request, return a clear error message."
            )

        # Step 4: Build initial messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sanitized_prompt},
        ]

        # Step 5: Reasoning loop — LLM selects tool(s), we execute and return raw results
        tool_call_log: list[ToolCallRecord] = []
        model_used = ""
        max_iterations = self._config.max_iterations
        llm_response: LLMResponse | None = None
        last_error_results: list[dict[str, Any]] | None = None

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
                    error_code=self._classify_llm_error(exc),
                )

            model_used = llm_response.model

            # Check for tool calls
            if llm_response.tool_calls:
                # Process each tool call
                new_records, raw_results = await self._dispatch_tool_calls(
                    llm_response=llm_response,
                    messages=messages,
                    ctx=ctx,
                )
                tool_call_log.extend(new_records)

                # HYBRID LOOP: Check if ALL tool calls succeeded.
                has_errors = any(r["is_error"] for r in raw_results)

                if not has_errors:
                    # FAST PATH: All tools succeeded → return raw results
                    # directly to the caller without a second LLM call.
                    if len(raw_results) == 1:
                        result_str = raw_results[0]["result"]
                    else:
                        result_str = json.dumps(raw_results)

                    return AgentResult(
                        status="completed",
                        result=result_str,
                        tool_calls=tool_call_log,
                        model_used=model_used,
                        iterations=iteration,
                        raw_tool_results=raw_results,
                    )

                # ERROR RETRY PATH: At least one tool failed. The redacted
                # error has already been appended to messages by
                # _execute_single_tool(). Continue the loop so the LLM
                # sees the error and can retry with different parameters
                # or a different approach.
                logger.info(
                    "agent.loop.error_retry",
                    iteration=iteration,
                    total_tools=len(raw_results),
                    failed_tools=sum(1 for r in raw_results if r["is_error"]),
                )
                last_error_results = raw_results
                continue

            elif llm_response.content:
                # LLM responded with text instead of tool calls — this means
                # it couldn't map the request to a tool. Return the text as-is
                # (it's an error/explanation from the LLM).
                return AgentResult(
                    status="completed",
                    result=llm_response.content,
                    tool_calls=tool_call_log,
                    model_used=model_used,
                    iterations=iteration,
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
                    error_code="empty_response",
                )

        # Max iterations reached without a successful tool execution.
        # This means every iteration resulted in tool errors and the LLM
        # could not recover. Return the last error to the caller.
        logger.warning(
            "agent.max_iterations",
            max_iterations=max_iterations,
            tool_call_count=len(tool_call_log),
        )

        # If we have error results from the last failed tool dispatch,
        # return them so the caller sees the actual error.
        if last_error_results is not None:
            if len(last_error_results) == 1:
                result_str = last_error_results[0]["result"]
            else:
                result_str = json.dumps(last_error_results)

            return AgentResult(
                status="error",
                result=result_str,
                tool_calls=tool_call_log,
                model_used=model_used,
                iterations=max_iterations,
                raw_tool_results=last_error_results,
                warning=f"Max iterations ({max_iterations}) reached — tool errors not resolved",
                error_code="max_iterations",
            )

        # Fallback: no tool errors captured (shouldn't happen in normal flow,
        # but defensive coding for edge cases like repeated empty responses).
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
            status="error",
            result=last_content,
            tool_calls=tool_call_log,
            model_used=model_used,
            iterations=max_iterations,
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
    ) -> tuple[list[ToolCallRecord], list[dict[str, Any]]]:
        """Process tool calls from an LLM response.

        For each tool call:
        1. Restore placeholder tokens to real values in arguments.
        2. Call the MCP tool with real arguments.
        3. Store the raw (unredacted) result for direct return to API consumer.
        4. On ERROR: append the redacted error to messages (for LLM retry).
        5. On SUCCESS: append a minimal acknowledgement to messages (required
           by the OpenAI API format, but contains no secrets).
        6. Record the call in the audit trail (with redacted arguments).

        Parameters
        ----------
        llm_response:
            The LLM response containing tool calls.
        messages:
            The mutable message list — error results are appended for retry;
            successful results get a minimal ack appended.
        ctx:
            The request-scoped SecretContext.

        Returns
        -------
        tuple[list[ToolCallRecord], list[dict[str, Any]]]
            A tuple of (audit trail entries, raw tool results). Each raw result
            is a dict with ``"tool_name"``, ``"result"`` (raw MCP response string),
            and ``"is_error"`` keys.
        """
        records: list[ToolCallRecord] = []
        raw_results: list[dict[str, Any]] = []

        if not llm_response.tool_calls:
            return records, raw_results

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
            record, raw_result = await self._execute_single_tool(tool_call, messages, ctx)
            records.append(record)
            raw_results.append(raw_result)

        return records, raw_results

    async def _execute_single_tool(
        self,
        tool_call: ToolCall,
        messages: list[dict[str, Any]],
        ctx: SecretContext,
    ) -> tuple[ToolCallRecord, dict[str, Any]]:
        """Execute a single MCP tool call with full security flow.

        Security flow:
            1. LLM tool call args (may contain placeholders)
            2. → restore_placeholders → real values
            3. → MCP server (receives real values)
            4. → Raw MCP result returned directly to API consumer
            5. On ERROR: redacted error appended to messages (for LLM retry)
            6. On SUCCESS: result is NOT appended to messages (LLM never sees it)

        Parameters
        ----------
        tool_call:
            The tool call from the LLM response.
        messages:
            The mutable message list — only error results are appended (so the
            LLM can see them for retry). Successful results are NOT appended
            because the LLM never needs to see them (raw-response architecture).
        ctx:
            The request-scoped SecretContext.

        Returns
        -------
        tuple[ToolCallRecord, dict[str, Any]]
            A tuple of (audit trail entry, raw tool result dict). The raw result
            has keys ``"tool_name"``, ``"result"`` (raw MCP content), and
            ``"is_error"``.
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

        # WARNING: DEBUG logging of real_arguments may expose secret values
        # (after placeholder restoration).  Only enable DEBUG level in
        # development environments.
        logger.debug(
            "agent.tool_call.arguments",
            tool_name=tool_name,
            tool_call_id=tool_call.id,
            llm_arguments=redacted_arguments,
            resolved_arguments=real_arguments,
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

            # Report the error to the LLM for reasoning (retry path)
            # Redact any secret values that may appear in the error message
            safe_error = self._redactor.redact_error_message(str(exc), ctx)
            error_result = json.dumps({
                "error": True,
                "message": safe_error,
                "tool_name": tool_name,
            })

            # Append error to messages so the LLM can see it on the next
            # iteration and retry with different parameters/approach.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": error_result,
            })

            raw_result = {
                "tool_name": tool_name,
                "result": error_result,
                "is_error": True,
            }

            return ToolCallRecord(
                tool_name=tool_name,
                arguments=redacted_arguments,
                result=error_result,
                is_error=True,
                duration_ms=duration_ms,
            ), raw_result

        duration_ms = int((time.monotonic() - start_time) * 1000)

        # WARNING: DEBUG logging of raw MCP result.  This will contain
        # secret values (passwords, tokens, keys, etc.).  Only enable
        # DEBUG level in development environments.
        logger.debug(
            "agent.tool_call.raw_result",
            tool_name=tool_name,
            tool_call_id=tool_call.id,
            is_error=mcp_result.is_error,
            duration_ms=duration_ms,
            raw_content=mcp_result.content,
        )

        # Step 3: Redact the tool result for the audit trail only.
        # In the raw-response architecture, successful tool results are NOT
        # appended to messages — the LLM never sees them.  We only redact
        # for the ToolCallRecord audit log.
        redacted_result = self._redactor.redact_tool_result(
            tool_name=tool_name,
            result=mcp_result.content,
            context=ctx,
        )

        # Step 4: Append a minimal tool result message.  The OpenAI API
        # requires a tool-result message for every tool_call_id in the
        # assistant message.  We must satisfy this contract so that, if
        # the loop continues (due to another tool failing), the message
        # list is valid.  The content is a short acknowledgement — it
        # does NOT contain secret values.
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({"status": "ok", "tool_name": tool_name}),
        })

        logger.info(
            "agent.tool_call.done",
            tool_name=tool_name,
            tool_call_id=tool_call.id,
            is_error=mcp_result.is_error,
            duration_ms=duration_ms,
        )

        # Build raw result for direct return to API consumer
        raw_result = {
            "tool_name": tool_name,
            "result": mcp_result.content,
            "is_error": mcp_result.is_error,
        }

        return ToolCallRecord(
            tool_name=tool_name,
            arguments=redacted_arguments,
            result=redacted_result,
            is_error=mcp_result.is_error,
            duration_ms=duration_ms,
        ), raw_result

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
