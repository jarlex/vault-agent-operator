"""PromptSanitizer — extracts secret values from consumer prompts before LLM processing.

When a consumer sends a prompt like::

    "write username=admin and password=SuperS3cret! to kv/prod/database"

the sanitizer extracts the secret values (``admin``, ``SuperS3cret!``),
replaces them with opaque placeholder tokens (``[SECRET_VALUE_1]``,
``[SECRET_VALUE_2]``), and returns the sanitized prompt::

    "write username=[SECRET_VALUE_1] and password=[SECRET_VALUE_2] to kv/prod/database"

The placeholder ↔ real value mappings are stored in the ``SecretContext``
for later restoration by the ``SecretRedactor`` when the LLM produces tool
calls.

The sanitizer handles two input modes:

1. **Natural language prompts with inline values**: Pattern detection for
   formats like ``key=value``, ``"password is X"``, ``JSON-like {...}`` etc.

2. **Structured secret_data field**: When the consumer provides an explicit
   ``secret_data`` dict, ALL values in that dict are treated as secrets and
   replaced with placeholders.

Security contract:
    - After sanitization, the returned prompt contains ZERO real secret values.
    - Only placeholder tokens and non-secret context (paths, key names, instructions)
      remain in the sanitized prompt.
    - The SecretContext holds the only reference to the real values.

Usage::

    sanitizer = PromptSanitizer()
    ctx = SecretContext()

    sanitized = sanitizer.sanitize_prompt(
        prompt="write password=MyS3cret to kv/prod/db",
        secret_data=None,
        context=ctx,
    )
    # sanitized == "write password=[SECRET_VALUE_1] to kv/prod/db"
    # ctx.resolve_placeholder("[SECRET_VALUE_1]") == "MyS3cret"
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.redaction.context import SecretContext
from src.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pattern definitions for secret value detection in natural language prompts.
# ---------------------------------------------------------------------------

# Pattern: key=value (no spaces in value) or key="value with spaces"
# Examples: password=SuperS3cret!, username="my user", api_key='abc123'
_KV_ASSIGNMENT_PATTERN = re.compile(
    r"""
    (?:^|[\s,;])                    # Start of string or separator
    ([\w.-]+)                       # Key name (captured for reference, group 1)
    \s*=\s*                         # Equals sign with optional whitespace
    (?:
        "([^"]+)"                   # Double-quoted value (group 2)
        |'([^']+)'                  # Single-quoted value (group 3)
        |(\S+)                      # Unquoted value (group 4)
    )
    """,
    re.VERBOSE,
)

# Pattern: "key is VALUE" or "key: VALUE" (common natural language formats)
# Examples: "password is MyS3cret", "the token is ghp_abc123"
_NATURAL_LANGUAGE_PATTERN = re.compile(
    r"""
    (?:^|[\s,;])                    # Start or separator
    (?:the\s+)?                     # Optional "the"
    (password|secret|token|key|value|credential|api[_-]?key|access[_-]?key)  # Known secret field names
    \s+(?:is|:)\s+                  # "is" or ":"
    (?:
        "([^"]+)"                   # Double-quoted value (group 2)
        |'([^']+)'                  # Single-quoted value (group 3)
        |(\S+)                      # Unquoted value (group 4)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Pattern: JSON-like inline object {"key": "value", ...}
_JSON_INLINE_PATTERN = re.compile(
    r"\{[^}]*\}",
)

# Vault paths — these are NOT secrets and should be preserved
_VAULT_PATH_PATTERN = re.compile(
    r"""
    (?:^|[\s"'])                    # Start or delimiter
    (                               # Group 1: the path
        (?:secret|kv|pki|auth|sys)  # Known Vault mount prefixes
        /[\w./-]+                   # Path segments
    )
    (?:[\s"',;]|$)                  # End delimiter
    """,
    re.VERBOSE,
)


class PromptSanitizer:
    """Extracts and replaces secret values in consumer prompts before LLM processing.

    The sanitizer is stateless — all secret state lives in the ``SecretContext``
    passed to ``sanitize_prompt()``. A single instance can be shared across
    requests.
    """

    def sanitize_prompt(
        self,
        prompt: str,
        secret_data: dict[str, Any] | None,
        context: SecretContext,
    ) -> str:
        """Sanitize a consumer prompt by replacing secret values with placeholders.

        This method handles two modes:

        1. **Structured secret_data**: If ``secret_data`` is provided, ALL
           values in the dict are treated as secrets. The sanitized prompt
           includes a reference to the data with placeholder tokens.

        2. **Inline values in natural language**: If no ``secret_data`` is
           provided, the sanitizer scans the prompt for common value patterns
           (``key=value``, ``"password is X"``, inline JSON) and replaces
           detected values with placeholders.

        Parameters
        ----------
        prompt:
            The consumer's natural language prompt.
        secret_data:
            Optional structured secret data (e.g. from the ``secret_data``
            field of a ``TaskRequest``). If provided, all values are treated
            as secrets.
        context:
            The request-scoped ``SecretContext`` for storing placeholder mappings.

        Returns
        -------
        str
            The sanitized prompt with all secret values replaced by placeholder
            tokens. Safe to send to the LLM.
        """
        sanitized = prompt

        if secret_data:
            sanitized = self._sanitize_structured_data(sanitized, secret_data, context)
        else:
            sanitized = self._sanitize_inline_values(sanitized, context)

        placeholder_count = context.placeholder_count
        if placeholder_count > 0:
            logger.info(
                "sanitizer.prompt_sanitized",
                placeholder_count=placeholder_count,
                prompt_length=len(sanitized),
                mode="structured" if secret_data else "inline",
                # NEVER log the original prompt or secret values
            )

        return sanitized

    # ------------------------------------------------------------------
    # Structured secret_data handling
    # ------------------------------------------------------------------

    def _sanitize_structured_data(
        self,
        prompt: str,
        secret_data: dict[str, Any],
        context: SecretContext,
    ) -> str:
        """Handle the case where the consumer provides an explicit ``secret_data`` dict.

        ALL values in ``secret_data`` are treated as secrets and replaced with
        placeholders. The prompt is augmented with a reference to the
        placeholder-substituted data.

        Example:
            prompt = "write to kv/prod/db"
            secret_data = {"username": "admin", "password": "S3cret"}

            Returns: "write to kv/prod/db with data: {username: [SECRET_VALUE_1], password: [SECRET_VALUE_2]}"
        """
        sanitized_prompt = prompt

        # Replace any secret_data values that appear literally in the prompt
        for key, value in secret_data.items():
            str_value = str(value)
            if str_value in sanitized_prompt:
                placeholder = context.create_placeholder(str_value)
                sanitized_prompt = sanitized_prompt.replace(str_value, placeholder)

        # Build a sanitized representation of the secret data with placeholders
        sanitized_data: dict[str, str] = {}
        for key, value in secret_data.items():
            str_value = str(value)
            placeholder = context.create_placeholder(str_value)
            sanitized_data[key] = placeholder

        # Append the sanitized data reference to the prompt
        data_repr = ", ".join(f"{k}: {v}" for k, v in sanitized_data.items())
        sanitized_prompt = f"{sanitized_prompt} with data: {{{data_repr}}}"

        return sanitized_prompt

    # ------------------------------------------------------------------
    # Inline value detection and replacement
    # ------------------------------------------------------------------

    def _sanitize_inline_values(
        self,
        prompt: str,
        context: SecretContext,
    ) -> str:
        """Scan the prompt for inline secret values and replace them with placeholders.

        Detects common patterns:
        - ``key=value`` assignments
        - ``"password is X"`` natural language patterns
        - Inline JSON objects with values

        Preserves Vault paths and key names as non-secret context.
        """
        sanitized = prompt

        # Collect Vault paths first — these should NOT be treated as secrets
        vault_paths: set[str] = set()
        for match in _VAULT_PATH_PATTERN.finditer(prompt):
            vault_paths.add(match.group(1))

        # 1. Process inline JSON objects
        sanitized = self._sanitize_inline_json(sanitized, context, vault_paths)

        # 2. Process key=value assignments
        sanitized = self._sanitize_kv_assignments(sanitized, context, vault_paths)

        # 3. Process natural language patterns ("password is X")
        sanitized = self._sanitize_natural_language(sanitized, context, vault_paths)

        return sanitized

    def _sanitize_inline_json(
        self,
        prompt: str,
        context: SecretContext,
        vault_paths: set[str],
    ) -> str:
        """Find and sanitize inline JSON objects in the prompt."""
        result = prompt

        for match in _JSON_INLINE_PATTERN.finditer(prompt):
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    sanitized_parts: list[str] = []
                    for key, value in data.items():
                        str_value = str(value)
                        if str_value in vault_paths:
                            # Vault paths are not secrets
                            sanitized_parts.append(f'"{key}": "{str_value}"')
                        else:
                            placeholder = context.create_placeholder(str_value)
                            sanitized_parts.append(f'"{key}": "{placeholder}"')

                    sanitized_json = "{" + ", ".join(sanitized_parts) + "}"
                    result = result.replace(json_str, sanitized_json, 1)
            except (json.JSONDecodeError, TypeError):
                # Not valid JSON — skip
                continue

        return result

    def _sanitize_kv_assignments(
        self,
        prompt: str,
        context: SecretContext,
        vault_paths: set[str],
    ) -> str:
        """Find and sanitize ``key=value`` assignments in the prompt."""
        result = prompt

        # Process matches in reverse order to preserve string positions
        matches = list(_KV_ASSIGNMENT_PATTERN.finditer(prompt))
        for match in reversed(matches):
            # Extract the value from whichever group matched
            value = match.group(2) or match.group(3) or match.group(4)
            if not value:
                continue

            # Skip if the value looks like a Vault path
            if value in vault_paths or value.startswith(("secret/", "kv/", "pki/", "auth/", "sys/")):
                continue

            # Skip if the value is very short (likely a flag or boolean)
            if len(value) <= 1:
                continue

            # Skip common non-secret values
            if value.lower() in ("true", "false", "null", "none", "yes", "no"):
                continue

            placeholder = context.create_placeholder(value)

            # Replace just the value portion in the original match
            full_match = match.group(0)
            # Rebuild the match with the value replaced
            key = match.group(1)
            # Find the value in the full match and replace it
            replacement = full_match.replace(value, placeholder, 1)
            result = result[:match.start()] + replacement + result[match.end():]

        return result

    def _sanitize_natural_language(
        self,
        prompt: str,
        context: SecretContext,
        vault_paths: set[str],
    ) -> str:
        """Find and sanitize natural language patterns like ``"password is X"``."""
        result = prompt

        matches = list(_NATURAL_LANGUAGE_PATTERN.finditer(prompt))
        for match in reversed(matches):
            value = match.group(2) or match.group(3) or match.group(4)
            if not value:
                continue

            # Skip Vault paths
            if value in vault_paths:
                continue

            # Skip very short values
            if len(value) <= 1:
                continue

            placeholder = context.create_placeholder(value)

            full_match = match.group(0)
            replacement = full_match.replace(value, placeholder, 1)
            result = result[:match.start()] + replacement + result[match.end():]

        return result
