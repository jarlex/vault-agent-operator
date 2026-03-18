"""LLM Provider — wraps LiteLLM for model-agnostic async completions with tool calling.

Provides:
- Async ``complete()`` method using ``litellm.acompletion()``
- Model routing from YAML config (alias → provider/model_id)
- Retry with exponential backoff for transient errors (429, 5xx)
- No retry for authentication failures (401, 403)
- Tool-calling compatibility checks
- ``get_available_models()`` returning configured ``ModelInfo`` list

Usage::

    from src.config import LLMConfig
    from src.llm.provider import LLMProvider

    provider = LLMProvider(config=llm_config, api_key="ghp_...")
    response = await provider.complete(messages=[{"role": "user", "content": "hello"}])
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)

from src.config.models import LLMConfig, ModelInfo
from src.logging import get_logger

logger = get_logger(__name__)

# Suppress LiteLLM's own verbose logging — we handle logging ourselves
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# Response data classes (matching design doc interfaces exactly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool/function call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token consumption for a single completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalised response from an LLM completion."""

    content: str | None
    tool_calls: list[ToolCall] | None
    model: str
    usage: TokenUsage | None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for LLM provider errors."""


class LLMAuthError(LLMError):
    """Authentication/authorization failure (401/403) — not retryable."""


class LLMRateLimitError(LLMError):
    """Rate-limited (429) — retryable."""


class LLMServiceError(LLMError):
    """Transient server error (5xx) — retryable."""


class LLMToolCallUnsupportedError(LLMError):
    """Selected model does not support tool/function calling."""


# ---------------------------------------------------------------------------
# LLMProvider
# ---------------------------------------------------------------------------


class LLMProvider:
    """Abstraction over LiteLLM for model-agnostic async completions.

    Parameters
    ----------
    config:
        ``LLMConfig`` with model list, default model, timeout and retry settings.
    api_key:
        API key passed to LiteLLM (e.g. GitHub PAT for ``github/`` models).
    """

    def __init__(self, config: LLMConfig, api_key: str) -> None:
        self._config = config
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Run a chat completion, optionally with tool definitions.

        Parameters
        ----------
        messages:
            OpenAI-format message list (role + content).
        tools:
            OpenAI-format tool definitions.  ``None`` disables tool calling.
        model:
            Model *alias* (e.g. ``"default"``, ``"fast"``).  When ``None``,
            the configured ``default_model`` is used.

        Returns
        -------
        LLMResponse
            Normalised response with content and/or tool calls.

        Raises
        ------
        LLMAuthError
            On 401/403 — never retried.
        LLMRateLimitError
            When retries for 429 are exhausted.
        LLMServiceError
            When retries for 5xx are exhausted.
        LLMToolCallUnsupportedError
            If the resolved model does not support tool calling and tools were
            provided.
        LLMError
            Catch-all for unexpected errors.
        """
        model_info = self._resolve_model(model)
        model_id = model_info.model_id

        # Guard: reject tool calling if model doesn't support it
        if tools and not model_info.supports_tool_calling:
            raise LLMToolCallUnsupportedError(
                f"Model '{model_info.name}' ({model_id}) does not support tool calling"
            )

        logger.info(
            "llm.completion.start",
            model=model_id,
            model_alias=model_info.name,
            message_count=len(messages),
            tool_count=len(tools) if tools else 0,
        )

        last_exc: Exception | None = None
        max_retries = self._config.max_retries

        for attempt in range(1, max_retries + 2):  # attempts = retries + 1
            try:
                response = await self._call_litellm(model_id, messages, tools)
                parsed = self._parse_response(response, model_id)

                logger.info(
                    "llm.completion.done",
                    model=model_id,
                    has_content=parsed.content is not None,
                    tool_call_count=len(parsed.tool_calls) if parsed.tool_calls else 0,
                    prompt_tokens=parsed.usage.prompt_tokens if parsed.usage else None,
                    completion_tokens=parsed.usage.completion_tokens if parsed.usage else None,
                )
                return parsed

            except AuthenticationError as exc:
                # 401/403 — never retry
                logger.error("llm.auth_error", model=model_id, error=str(exc))
                raise LLMAuthError(f"Authentication failed for model {model_id}: {exc}") from exc

            except RateLimitError as exc:
                last_exc = exc
                if attempt > max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "llm.rate_limit",
                    model=model_id,
                    attempt=attempt,
                    retry_after_s=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)

            except (ServiceUnavailableError, litellm.exceptions.APIConnectionError) as exc:
                last_exc = exc
                if attempt > max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "llm.transient_error",
                    model=model_id,
                    attempt=attempt,
                    retry_after_s=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)

            except Exception as exc:
                # Unknown errors — also attempt retry (may be transient)
                last_exc = exc
                if attempt > max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "llm.unexpected_error",
                    model=model_id,
                    attempt=attempt,
                    retry_after_s=delay,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            "llm.retries_exhausted",
            model=model_id,
            max_retries=max_retries,
            error=str(last_exc),
        )

        if isinstance(last_exc, RateLimitError):
            raise LLMRateLimitError(
                f"Rate limited after {max_retries} retries for model {model_id}: {last_exc}"
            ) from last_exc

        if isinstance(last_exc, (ServiceUnavailableError, litellm.exceptions.APIConnectionError)):
            raise LLMServiceError(
                f"Service unavailable after {max_retries} retries for model {model_id}: {last_exc}"
            ) from last_exc

        raise LLMError(
            f"LLM call failed after {max_retries} retries for model {model_id}: {last_exc}"
        ) from last_exc

    def get_available_models(self) -> list[ModelInfo]:
        """Return the list of configured models."""
        return list(self._config.models)

    @staticmethod
    def supports_tool_calling(model_name: str, config: LLMConfig) -> bool:
        """Check if a named model supports function/tool calling."""
        info = config.get_model(model_name)
        if info is None:
            return False
        return info.supports_tool_calling

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, alias: str | None) -> ModelInfo:
        """Resolve a model alias to its ``ModelInfo``.  Falls back to default."""
        info = self._config.get_model(alias)
        if info is not None:
            return info

        # If caller passed a name that doesn't match any alias, try default
        if alias is not None:
            logger.warning(
                "llm.model_not_found",
                requested=alias,
                fallback=self._config.default_model,
            )
        default = self._config.get_model(self._config.default_model)
        if default is None:
            raise LLMError(
                f"Default model '{self._config.default_model}' is not configured"
            )
        return default

    async def _call_litellm(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        """Call ``litellm.acompletion()`` with the configured parameters."""
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "api_key": self._api_key,
            "timeout": self._config.request_timeout,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return await litellm.acompletion(**kwargs)

    @staticmethod
    def _parse_response(response: Any, model_id: str) -> LLMResponse:
        """Parse a LiteLLM response into our normalised ``LLMResponse``."""
        choice = response.choices[0]
        message = choice.message

        # --- Content ---
        content: str | None = getattr(message, "content", None)

        # --- Tool calls ---
        raw_tool_calls = getattr(message, "tool_calls", None)
        tool_calls: list[ToolCall] | None = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                fn = tc.function
                try:
                    args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": fn.arguments}
                tool_calls.append(
                    ToolCall(id=tc.id, name=fn.name, arguments=args)
                )

        # --- Usage ---
        usage: TokenUsage | None = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            usage = TokenUsage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", 0),
                completion_tokens=getattr(raw_usage, "completion_tokens", 0),
                total_tokens=getattr(raw_usage, "total_tokens", 0),
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            model=model_id,
            usage=usage,
        )

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff: 1s, 2s, 4s, ..., capped at 30s."""
        return min(2 ** (attempt - 1), 30.0)
