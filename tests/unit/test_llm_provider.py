"""Unit tests for LLMProvider — mock litellm, test model routing, retry logic, error handling.

Tests cover:
- Model resolution (alias → model_id)
- Tool calling compatibility checks
- Retry logic with exponential backoff for transient errors (429, 5xx)
- No retry for authentication failures (401, 403)
- Response parsing (content, tool calls, usage)
- get_available_models()
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.models import LLMConfig, ModelInfo
from src.llm.provider import (
    LLMAuthError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMServiceError,
    LLMToolCallUnsupportedError,
    ToolCall,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_litellm_response(
    content: str | None = "Hello!",
    tool_calls: list[dict] | None = None,
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Build a mock LiteLLM response object matching the actual shape."""
    response = MagicMock()

    message = MagicMock()
    message.content = content

    if tool_calls:
        mock_tool_calls = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function = MagicMock()
            mock_tc.function.name = tc["name"]
            mock_tc.function.arguments = json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"]
            mock_tool_calls.append(mock_tc)
        message.tool_calls = mock_tool_calls
    else:
        message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    response.choices = [choice]

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    response.usage = usage

    return response


def _make_provider(
    config: LLMConfig | None = None,
    api_key: str = "test-key",
) -> LLMProvider:
    """Create an LLMProvider with test config."""
    if config is None:
        config = LLMConfig(
            default_model="default",
            request_timeout=10,
            max_retries=2,
            models=[
                ModelInfo(name="default", provider="test", model_id="test/gpt-4o", supports_tool_calling=True),
                ModelInfo(name="no-tools", provider="test", model_id="test/no-tools", supports_tool_calling=False),
                ModelInfo(name="fast", provider="test", model_id="test/fast", supports_tool_calling=True),
            ],
        )
    return LLMProvider(config=config, api_key=api_key)


# ===================================================================
# Model Resolution
# ===================================================================


class TestModelResolution:
    """Test model alias resolution to model_id."""

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_default_model_used_when_none(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN no model specified, WHEN complete() is called,
        THEN the default model (test/gpt-4o) is used."""
        mock_acompletion.return_value = _make_litellm_response()

        provider = _make_provider()
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}])

        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "test/gpt-4o"
        assert result.model == "test/gpt-4o"

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_specific_model_alias(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN model alias 'fast', WHEN complete() is called,
        THEN the corresponding model_id is used."""
        mock_acompletion.return_value = _make_litellm_response()

        provider = _make_provider()
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="fast",
        )

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "test/fast"

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_unknown_alias_falls_back_to_default(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN an unknown model alias, WHEN complete() is called,
        THEN the default model is used as fallback."""
        mock_acompletion.return_value = _make_litellm_response()

        provider = _make_provider()
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="nonexistent-model",
        )

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "test/gpt-4o"

    def test_get_available_models(self) -> None:
        """GIVEN configured models, WHEN get_available_models() is called,
        THEN all configured models are returned."""
        provider = _make_provider()
        models = provider.get_available_models()
        assert len(models) == 3
        names = {m.name for m in models}
        assert names == {"default", "no-tools", "fast"}


# ===================================================================
# Tool Calling Compatibility
# ===================================================================


class TestToolCallingCompatibility:
    """Test tool calling compatibility checks."""

    @pytest.mark.asyncio
    async def test_model_without_tool_support_raises(self) -> None:
        """GIVEN a model that doesn't support tool calling,
        WHEN complete() is called with tools, THEN LLMToolCallUnsupportedError is raised."""
        provider = _make_provider()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]

        with pytest.raises(LLMToolCallUnsupportedError, match="does not support tool calling"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                model="no-tools",
            )

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_model_with_tool_support_works(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN a model that supports tool calling,
        WHEN complete() is called with tools, THEN the call succeeds."""
        mock_acompletion.return_value = _make_litellm_response()

        provider = _make_provider()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]

        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            model="default",
        )
        assert result.content == "Hello!"

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_no_tools_means_no_tool_choice(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN tools=None, WHEN complete() is called,
        THEN tool_choice is NOT included in the litellm call."""
        mock_acompletion.return_value = _make_litellm_response()

        provider = _make_provider()
        await provider.complete(messages=[{"role": "user", "content": "hi"}], tools=None)

        call_kwargs = mock_acompletion.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_supports_tool_calling_static(self) -> None:
        """Test the static supports_tool_calling helper."""
        config = LLMConfig(models=[
            ModelInfo(name="a", provider="t", model_id="t/a", supports_tool_calling=True),
            ModelInfo(name="b", provider="t", model_id="t/b", supports_tool_calling=False),
        ])
        assert LLMProvider.supports_tool_calling("a", config) is True
        assert LLMProvider.supports_tool_calling("b", config) is False
        assert LLMProvider.supports_tool_calling("unknown", config) is False


# ===================================================================
# Retry Logic
# ===================================================================


class TestRetryLogic:
    """Test retry behavior for transient errors."""

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_rate_limit_retried_then_succeeds(
        self, mock_sleep: AsyncMock, mock_acompletion: AsyncMock,
    ) -> None:
        """GIVEN the LLM returns 429 once then succeeds,
        WHEN complete() is called, THEN it retries and returns success."""
        from litellm.exceptions import RateLimitError

        mock_acompletion.side_effect = [
            RateLimitError("rate limited", "test", "test"),
            _make_litellm_response(content="OK after retry"),
        ]

        provider = _make_provider()
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "OK after retry"
        assert mock_acompletion.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_rate_limit_exhausts_retries(
        self, mock_sleep: AsyncMock, mock_acompletion: AsyncMock,
    ) -> None:
        """GIVEN the LLM always returns 429,
        WHEN retries are exhausted, THEN LLMRateLimitError is raised."""
        from litellm.exceptions import RateLimitError

        mock_acompletion.side_effect = RateLimitError("rate limited", "test", "test")

        provider = _make_provider()  # max_retries=2

        with pytest.raises(LLMRateLimitError, match="Rate limited"):
            await provider.complete(messages=[{"role": "user", "content": "hi"}])

        # 1 initial + 2 retries = 3 calls
        assert mock_acompletion.call_count == 3

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_auth_error_not_retried(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN the LLM returns 401 (auth error),
        WHEN complete() is called, THEN LLMAuthError is raised immediately (no retries)."""
        from litellm.exceptions import AuthenticationError

        mock_acompletion.side_effect = AuthenticationError("bad token", "test", "test")

        provider = _make_provider()

        with pytest.raises(LLMAuthError, match="Authentication failed"):
            await provider.complete(messages=[{"role": "user", "content": "hi"}])

        # Only one call — no retries
        assert mock_acompletion.call_count == 1

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_service_error_retried(
        self, mock_sleep: AsyncMock, mock_acompletion: AsyncMock,
    ) -> None:
        """GIVEN the LLM returns 5xx then succeeds, WHEN retried, THEN succeeds."""
        from litellm.exceptions import ServiceUnavailableError

        mock_acompletion.side_effect = [
            ServiceUnavailableError("down", "test", "test"),
            _make_litellm_response(content="Back up"),
        ]

        provider = _make_provider()
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "Back up"
        assert mock_acompletion.call_count == 2


# ===================================================================
# Response Parsing
# ===================================================================


class TestResponseParsing:
    """Test LLM response parsing into normalized LLMResponse."""

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_parse_text_response(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN a text response, WHEN parsed, THEN content is extracted."""
        mock_acompletion.return_value = _make_litellm_response(content="Hello world")

        provider = _make_provider()
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "Hello world"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_parse_tool_call_response(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN a response with tool calls, WHEN parsed, THEN tool calls are extracted."""
        mock_acompletion.return_value = _make_litellm_response(
            content=None,
            tool_calls=[
                {"id": "call_1", "name": "vault_kv_read", "arguments": {"path": "secret/app"}},
            ],
        )

        provider = _make_provider()
        result = await provider.complete(
            messages=[{"role": "user", "content": "read secret"}],
            tools=[{"type": "function", "function": {"name": "vault_kv_read", "parameters": {}}}],
        )

        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "vault_kv_read"
        assert result.tool_calls[0].arguments == {"path": "secret/app"}

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_parse_usage(self, mock_acompletion: AsyncMock) -> None:
        """GIVEN a response with usage info, WHEN parsed, THEN TokenUsage is populated."""
        mock_acompletion.return_value = _make_litellm_response(
            prompt_tokens=100, completion_tokens=50,
        )

        provider = _make_provider()
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}])

        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150


# ===================================================================
# Backoff Delay
# ===================================================================


class TestBackoffDelay:
    """Test the exponential backoff calculation."""

    def test_backoff_values(self) -> None:
        """GIVEN attempt numbers, WHEN _backoff_delay is called,
        THEN it returns exponential values capped at 30s."""
        assert LLMProvider._backoff_delay(1) == 1.0
        assert LLMProvider._backoff_delay(2) == 2.0
        assert LLMProvider._backoff_delay(3) == 4.0
        assert LLMProvider._backoff_delay(4) == 8.0
        assert LLMProvider._backoff_delay(5) == 16.0
        assert LLMProvider._backoff_delay(6) == 30.0  # capped
        assert LLMProvider._backoff_delay(10) == 30.0  # still capped
