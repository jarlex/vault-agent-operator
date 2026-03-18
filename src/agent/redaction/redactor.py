"""SecretRedactor — intercepts MCP tool results and strips secret values.

This module sits between the MCP client and the Agent Core's LLM conversation.
Every tool result passes through the redactor BEFORE being added to the LLM
message history. The redactor:

1. **Redacts tool results**: Removes secret values from MCP responses,
   keeping only metadata and key names for the LLM to reason about.
   The full unredacted response is stored in the SecretContext for the
   API consumer.

2. **Restores placeholders**: When the LLM produces tool-call arguments
   containing placeholder tokens (from the PromptSanitizer), the redactor
   replaces them with real secret values before the call reaches the MCP server.

Security contract:
    - ``redact_tool_result()`` is called BEFORE every tool result enters the LLM.
    - ``restore_placeholders()`` is called BEFORE every tool call reaches MCP.
    - The redactor uses policies from ``policies.py`` for tool-specific rules.
    - Unknown tools use conservative redaction (all values stripped).

Usage in the Agent Core reasoning loop::

    # After MCP tool call returns:
    redacted_result = redactor.redact_tool_result(tool_name, raw_result, context)
    messages.append({"role": "tool", "content": redacted_result})

    # Before MCP tool call:
    real_arguments = redactor.restore_placeholders(arguments, context)
    mcp_result = await mcp_client.call_tool(tool_name, real_arguments)
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.redaction.context import SecretContext
from src.agent.redaction.policies import get_policy_for_tool
from src.logging import get_logger

logger = get_logger(__name__)

# Pattern to detect placeholder tokens in strings
_PLACEHOLDER_PATTERN = re.compile(r"\[SECRET_VALUE_\d+\]")


class SecretRedactor:
    """Redacts secret values from MCP tool results and restores placeholders in tool arguments.

    This class is stateless — all secret state lives in the ``SecretContext``
    passed to each method. A single ``SecretRedactor`` instance can be shared
    across requests.

    The redactor delegates to per-tool-type ``RedactionPolicy`` instances
    for the actual redaction logic. See ``policies.py``.
    """

    def redact_tool_result(
        self,
        tool_name: str,
        result: Any,
        context: SecretContext,
    ) -> str:
        """Redact secret values from an MCP tool result before it enters the LLM conversation.

        The full unredacted result is stored in ``context`` for inclusion in
        the API consumer's response.

        Parameters
        ----------
        tool_name:
            The MCP tool that produced this result (e.g. ``"vault_kv_read"``).
        result:
            The raw tool result from the MCP server. May be a string (JSON),
            dict, or other type.
        context:
            The request-scoped ``SecretContext`` for storing unredacted data.

        Returns
        -------
        str
            A JSON string of the redacted result, safe to include in the LLM
            conversation as a tool result message.
        """
        policy = get_policy_for_tool(tool_name)
        policy_name = type(policy).__name__

        try:
            redaction_result = policy.redact(tool_name, result)
            redacted = redaction_result["redacted"]
            unredacted = redaction_result["unredacted"]
        except Exception:
            # If redaction fails, use ultra-conservative fallback:
            # redact EVERYTHING, store nothing.
            logger.exception(
                "redaction.policy_error",
                tool_name=tool_name,
                policy=policy_name,
            )
            redacted = {
                "error": "Redaction processing failed. Tool result fully redacted for safety.",
                "tool_name": tool_name,
            }
            unredacted = result

        # Store unredacted response for the API consumer
        context.store_unredacted_response(tool_name, unredacted)

        # Also scan the redacted result for any accidentally leaked secret values.
        # This is a defense-in-depth check — policies should already handle this,
        # but we double-check here.
        redacted = self._scrub_known_secrets(redacted, context)

        logger.info(
            "redaction.tool_result_redacted",
            tool_name=tool_name,
            policy=policy_name,
            # NEVER log the result content
        )

        # Return as JSON string for inclusion in LLM messages
        if isinstance(redacted, str):
            return redacted
        return json.dumps(redacted, indent=2, default=str)

    def restore_placeholders(
        self,
        arguments: dict[str, Any],
        context: SecretContext,
    ) -> dict[str, Any]:
        """Replace placeholder tokens in tool-call arguments with real secret values.

        Called BEFORE a tool call is forwarded to the MCP server. The LLM may
        have included placeholder tokens like ``[SECRET_VALUE_1]`` in its
        generated arguments (because the PromptSanitizer replaced real values
        in the prompt). This method restores the real values so the MCP server
        receives correct data.

        Parameters
        ----------
        arguments:
            The tool-call arguments generated by the LLM, potentially containing
            placeholder tokens.
        context:
            The request-scoped ``SecretContext`` with placeholder → value mappings.

        Returns
        -------
        dict[str, Any]
            A new dict with all placeholder tokens replaced by their real values.
            The original dict is not modified.
        """
        if not context.has_placeholders():
            return arguments

        restored = self._deep_restore(arguments, context)

        # Log that we performed restoration (without logging the values)
        placeholder_count = self._count_placeholders_in(arguments)
        if placeholder_count > 0:
            logger.info(
                "redaction.placeholders_restored",
                placeholder_count=placeholder_count,
                # NEVER log the actual values
            )

        return restored

    def redact_error_message(
        self,
        error_message: str,
        context: SecretContext,
    ) -> str:
        """Redact any secret values that may appear in error messages.

        Vault or MCP server error messages might accidentally include secret
        values (e.g. "cannot write value 's3cret' to path ..."). This method
        replaces any known secret values with ``[REDACTED]``.

        Parameters
        ----------
        error_message:
            The raw error message from the MCP server or Vault.
        context:
            The request-scoped ``SecretContext`` with known secret values.

        Returns
        -------
        str
            The error message with any known secret values replaced.
        """
        if not context.has_placeholders():
            return error_message

        redacted = error_message
        # We need to check if any known secret values appear in the error
        # Access internal mapping through resolve — iterate placeholders
        # Note: We use the context's resolve to check each known value
        for response_entry in context.get_unredacted_responses():
            response = response_entry.get("response")
            if isinstance(response, dict):
                redacted = self._redact_dict_values_from_string(redacted, response)
            elif isinstance(response, str):
                # Try to parse as JSON for value extraction
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        redacted = self._redact_dict_values_from_string(redacted, parsed)
                except (json.JSONDecodeError, TypeError):
                    pass

        return redacted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deep_restore(self, data: Any, context: SecretContext) -> Any:
        """Recursively replace placeholder tokens with real values in a data structure."""
        if isinstance(data, dict):
            return {k: self._deep_restore(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deep_restore(item, context) for item in data]
        elif isinstance(data, str):
            if _PLACEHOLDER_PATTERN.search(data):
                return context.resolve_all_placeholders(data)
            return data
        else:
            return data

    def _count_placeholders_in(self, data: Any) -> int:
        """Count placeholder tokens in a data structure."""
        count = 0
        if isinstance(data, dict):
            for v in data.values():
                count += self._count_placeholders_in(v)
        elif isinstance(data, list):
            for item in data:
                count += self._count_placeholders_in(item)
        elif isinstance(data, str):
            count += len(_PLACEHOLDER_PATTERN.findall(data))
        return count

    def _scrub_known_secrets(self, data: Any, context: SecretContext) -> Any:
        """Defense-in-depth: scan redacted output for any leaked secret values.

        If any value in the redacted output matches a known secret value from
        the context's unredacted responses, replace it with ``[REDACTED]``.

        This is a safety net — well-written policies should never leak secrets,
        but this catches edge cases and policy bugs.
        """
        # Collect all known secret values from unredacted responses
        known_secrets: set[str] = set()
        for entry in context.get_unredacted_responses():
            response = entry.get("response")
            self._extract_string_values(response, known_secrets)

        if not known_secrets:
            return data

        return self._replace_known_secrets(data, known_secrets)

    def _extract_string_values(self, data: Any, values: set[str]) -> None:
        """Recursively extract all string values from a data structure."""
        if isinstance(data, dict):
            for v in data.values():
                self._extract_string_values(v, values)
        elif isinstance(data, list):
            for item in data:
                self._extract_string_values(item, values)
        elif isinstance(data, str):
            # Only consider values that look like they could be secrets
            # (skip very short strings, common words, paths, etc.)
            stripped = data.strip()
            if len(stripped) >= 4 and not stripped.startswith("/"):
                values.add(stripped)

    def _replace_known_secrets(self, data: Any, known_secrets: set[str]) -> Any:
        """Replace any occurrence of known secret values in the data."""
        if isinstance(data, dict):
            return {k: self._replace_known_secrets(v, known_secrets) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_known_secrets(item, known_secrets) for item in data]
        elif isinstance(data, str):
            result = data
            for secret in known_secrets:
                if secret in result:
                    result = result.replace(secret, "[REDACTED]")
                    logger.warning(
                        "redaction.defense_in_depth_triggered",
                        message="Secret value found in supposedly redacted output — replaced.",
                        # NEVER log the actual secret value
                    )
            return result
        else:
            return data

    def _redact_dict_values_from_string(self, text: str, data: dict[str, Any]) -> str:
        """Replace any dict values that appear in a string with [REDACTED]."""
        result = text
        for value in data.values():
            if isinstance(value, str) and len(value) >= 4 and value in result:
                result = result.replace(value, "[REDACTED]")
            elif isinstance(value, dict):
                result = self._redact_dict_values_from_string(result, value)
        return result
