"""SecretContext — request-scoped mapping of placeholder tokens to real secret values.

This is a SECURITY-CRITICAL component. The SecretContext holds the only copy
of real secret values during a request lifecycle. It is:

- **Request-scoped**: Created when a request starts, destroyed when it ends.
- **Non-serializable**: Intentionally prevents JSON/pickle serialization.
- **Non-loggable**: ``__repr__`` and ``__str__`` never expose secret values.
- **Ephemeral**: All secret values are cleared on ``destroy()``.

The SecretContext serves two purposes:

1. **Placeholder ↔ real value mapping**: When the PromptSanitizer extracts
   secret values from a consumer's prompt, it stores them here keyed by opaque
   placeholder tokens (e.g. ``[SECRET_VALUE_1]``). When the LLM produces a
   tool call containing those placeholders, the SecretRedactor uses this
   context to substitute the real values before forwarding to MCP.

2. **Unredacted tool responses**: When the SecretRedactor strips secret values
   from an MCP tool result before passing it to the LLM, it stores the full
   unredacted result here. The API layer retrieves these to include in the
   consumer-facing response.

Usage::

    ctx = SecretContext()
    token = ctx.create_placeholder("SuperS3cret!")
    # token == "[SECRET_VALUE_1]"
    ctx.resolve_placeholder(token)  # → "SuperS3cret!"

    ctx.store_unredacted_response("vault_kv_read", {"password": "SuperS3cret!"})
    ctx.get_unredacted_responses()  # → [{"tool_name": "vault_kv_read", "response": {...}}]

    ctx.destroy()  # Clears all data from memory
"""

from __future__ import annotations

import threading
from typing import Any

from src.logging import get_logger

logger = get_logger(__name__)

# Counter prefix — deterministic, opaque, non-guessable (no part of real value)
_PLACEHOLDER_PREFIX = "[SECRET_VALUE_"
_PLACEHOLDER_SUFFIX = "]"


class SecretContext:
    """Request-scoped container for secret values and their placeholder mappings.

    This class is the ONLY place where real secret values live during request
    processing. It is intentionally NOT serializable and NOT loggable.

    Thread Safety
    -------------
    The context uses a threading lock to protect concurrent access. While the
    agent is primarily async (single-threaded event loop), the lock provides
    safety if the context is ever accessed from background threads or in tests.

    Security Model
    --------------
    - Placeholders are sequential: ``[SECRET_VALUE_1]``, ``[SECRET_VALUE_2]``, etc.
    - Placeholders contain NO part of the actual secret value.
    - ``__repr__`` and ``__str__`` never expose mapping contents.
    - ``__getstate__`` / ``__reduce__`` raise errors to prevent serialization.
    - ``destroy()`` clears all internal data structures.
    """

    __slots__ = (
        "_placeholder_to_value",
        "_value_to_placeholder",
        "_unredacted_responses",
        "_counter",
        "_lock",
        "_destroyed",
    )

    def __init__(self) -> None:
        self._placeholder_to_value: dict[str, str] = {}
        self._value_to_placeholder: dict[str, str] = {}
        self._unredacted_responses: list[dict[str, Any]] = []
        self._counter: int = 0
        self._lock = threading.Lock()
        self._destroyed: bool = False

    # ------------------------------------------------------------------
    # Placeholder Management
    # ------------------------------------------------------------------

    def create_placeholder(self, real_value: str) -> str:
        """Create an opaque placeholder token for a secret value.

        If the same ``real_value`` was already registered, the existing
        placeholder is returned (idempotent for the same value within one
        request).

        Parameters
        ----------
        real_value:
            The actual secret value to protect.

        Returns
        -------
        str
            An opaque token like ``[SECRET_VALUE_1]``.

        Raises
        ------
        RuntimeError
            If the context has been destroyed.
        """
        self._assert_alive()
        with self._lock:
            # Idempotent: return existing placeholder if value already registered
            if real_value in self._value_to_placeholder:
                return self._value_to_placeholder[real_value]

            self._counter += 1
            placeholder = f"{_PLACEHOLDER_PREFIX}{self._counter}{_PLACEHOLDER_SUFFIX}"

            self._placeholder_to_value[placeholder] = real_value
            self._value_to_placeholder[real_value] = placeholder

            logger.debug(
                "secret_context.placeholder_created",
                placeholder=placeholder,
                # NEVER log the real value
            )
            return placeholder

    def resolve_placeholder(self, placeholder: str) -> str | None:
        """Resolve a placeholder token back to its real secret value.

        Parameters
        ----------
        placeholder:
            The placeholder token (e.g. ``[SECRET_VALUE_1]``).

        Returns
        -------
        str | None
            The real secret value, or ``None`` if the placeholder is unknown.

        Raises
        ------
        RuntimeError
            If the context has been destroyed.
        """
        self._assert_alive()
        with self._lock:
            return self._placeholder_to_value.get(placeholder)

    def resolve_all_placeholders(self, text: str) -> str:
        """Replace ALL placeholder tokens in a string with their real values.

        This is used by the SecretRedactor to restore real values in tool-call
        arguments before forwarding to the MCP server.

        Parameters
        ----------
        text:
            The string potentially containing placeholder tokens.

        Returns
        -------
        str
            The string with all known placeholders replaced by real values.

        Raises
        ------
        RuntimeError
            If the context has been destroyed.
        """
        self._assert_alive()
        with self._lock:
            result = text
            for placeholder, real_value in self._placeholder_to_value.items():
                result = result.replace(placeholder, real_value)
            return result

    def has_placeholders(self) -> bool:
        """Return True if any placeholder mappings have been registered."""
        self._assert_alive()
        with self._lock:
            return len(self._placeholder_to_value) > 0

    @property
    def placeholder_count(self) -> int:
        """Number of placeholder ↔ value pairs currently registered."""
        self._assert_alive()
        with self._lock:
            return len(self._placeholder_to_value)

    # ------------------------------------------------------------------
    # Unredacted Response Storage
    # ------------------------------------------------------------------

    def store_unredacted_response(self, tool_name: str, response: Any) -> None:
        """Store the full, unredacted MCP tool response for the API consumer.

        The redacted version goes to the LLM; this full version is kept here
        so the API layer can include it in the consumer-facing ``TaskResponse``.

        Parameters
        ----------
        tool_name:
            The MCP tool that produced this response.
        response:
            The complete, unredacted tool result.

        Raises
        ------
        RuntimeError
            If the context has been destroyed.
        """
        self._assert_alive()
        with self._lock:
            self._unredacted_responses.append({
                "tool_name": tool_name,
                "response": response,
            })

        logger.debug(
            "secret_context.unredacted_response_stored",
            tool_name=tool_name,
            # NEVER log the response content
        )

    def get_unredacted_responses(self) -> list[dict[str, Any]]:
        """Retrieve all stored unredacted tool responses.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has ``"tool_name"`` and ``"response"`` keys.

        Raises
        ------
        RuntimeError
            If the context has been destroyed.
        """
        self._assert_alive()
        with self._lock:
            # Return a copy to prevent external mutation
            return list(self._unredacted_responses)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Clear all secret data from memory and mark the context as destroyed.

        After calling this method, any further method calls will raise
        ``RuntimeError``. This is called at the end of each request to ensure
        no secret values persist in memory.
        """
        with self._lock:
            # Overwrite values before clearing to reduce memory remnant risk
            for key in self._placeholder_to_value:
                self._placeholder_to_value[key] = ""
            for key in self._value_to_placeholder:
                self._value_to_placeholder[key] = ""

            self._placeholder_to_value.clear()
            self._value_to_placeholder.clear()
            self._unredacted_responses.clear()
            self._counter = 0
            self._destroyed = True

        logger.debug("secret_context.destroyed")

    @property
    def is_destroyed(self) -> bool:
        """Whether this context has been destroyed."""
        return self._destroyed

    # ------------------------------------------------------------------
    # Security: prevent serialization and logging of secrets
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Safe repr — never exposes secret values."""
        return (
            f"<SecretContext placeholders={self._counter} "
            f"unredacted_responses={len(self._unredacted_responses)} "
            f"destroyed={self._destroyed}>"
        )

    def __str__(self) -> str:
        """Safe str — never exposes secret values."""
        return self.__repr__()

    def __getstate__(self) -> None:
        """Prevent pickling — SecretContext MUST NOT be serialized."""
        raise TypeError(
            "SecretContext cannot be serialized. It contains real secret values "
            "and is designed to be ephemeral and request-scoped."
        )

    def __reduce__(self) -> None:
        """Prevent pickle reduce protocol."""
        raise TypeError(
            "SecretContext cannot be serialized. It contains real secret values "
            "and is designed to be ephemeral and request-scoped."
        )

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self) -> SecretContext:
        """Support ``with SecretContext() as ctx:`` pattern for automatic cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Destroy the context on exit, even if an exception occurred."""
        self.destroy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_alive(self) -> None:
        """Raise if the context has been destroyed."""
        if self._destroyed:
            raise RuntimeError(
                "SecretContext has been destroyed. Cannot access secret data "
                "after request completion."
            )
