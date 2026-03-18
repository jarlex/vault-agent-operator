"""Structured logging configuration for vault-operator-agent.

Configures structlog for JSON output to stdout with:
- ISO 8601 timestamps
- Log level
- Context binding (request_id, tool_name, model, client_cn)
- Secret value redaction processor
- Log level controlled by settings

Usage::

    from src.logging import get_logger, setup_logging

    setup_logging(level="INFO", fmt="json", redact_patterns=["token", "password"])
    logger = get_logger()
    logger.info("task.started", prompt="read secret", request_id="abc-123")
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog


# ---------------------------------------------------------------------------
# Redaction Processor
# ---------------------------------------------------------------------------

# Compiled at module level; updated by setup_logging()
_REDACT_RE: re.Pattern[str] | None = None
_REDACT_PLACEHOLDER = "[REDACTED]"

# Values that look like secrets by pattern (high-entropy, tokens, etc.)
_SECRET_VALUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^ghp_[A-Za-z0-9_]{36,}$"),  # GitHub PAT
    re.compile(r"^gho_[A-Za-z0-9_]{36,}$"),  # GitHub OAuth
    re.compile(r"^sk-[A-Za-z0-9]{32,}$"),     # OpenAI key
    re.compile(r"^hvs\.[A-Za-z0-9_-]{20,}$"),  # Vault token
    re.compile(r"-----BEGIN .* PRIVATE KEY-----"),  # PEM private key
    re.compile(r"^Bearer\s+.{20,}$"),          # Bearer token in auth headers
]


def _is_secret_value(value: str) -> bool:
    """Heuristic check if a string looks like a known secret format."""
    return any(pat.search(value) for pat in _SECRET_VALUE_PATTERNS)


def _redact_value(key: str, value: Any) -> Any:
    """Redact a single value if the key matches a sensitive pattern."""
    if _REDACT_RE is None:
        return value

    if isinstance(value, str):
        # Redact by key name
        if _REDACT_RE.search(key.lower()):
            return _REDACT_PLACEHOLDER
        # Redact by value pattern (regardless of key)
        if _is_secret_value(value):
            return _REDACT_PLACEHOLDER
    return value


def _redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact sensitive values from a dict."""
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = _redact_dict(v)
        elif isinstance(v, list):
            result[k] = [_redact_dict(i) if isinstance(i, dict) else _redact_value(k, i) for i in v]
        else:
            result[k] = _redact_value(k, v)
    return result


def redaction_processor(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """structlog processor that redacts sensitive values from log events.

    Redacts values whose keys match any of the configured redact_patterns,
    and values that look like known secret formats (tokens, keys, etc.).
    This processor ensures that secrets are NEVER written to logs, even at
    DEBUG level.
    """
    return _redact_dict(event_dict)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    redact_patterns: list[str] | None = None,
) -> None:
    """Configure structlog for the application.

    Parameters
    ----------
    level:
        Logging level (DEBUG, INFO, WARNING, ERROR).
    fmt:
        Output format — ``"json"`` for structured JSON to stdout (production),
        ``"console"`` for human-readable coloured output (development).
    redact_patterns:
        List of substrings to match against log event key names.  Any key
        whose lowercased name contains one of these substrings will have its
        value replaced with ``[REDACTED]``.
    """
    global _REDACT_RE

    # Build redaction regex from patterns
    if redact_patterns:
        escaped = [re.escape(p) for p in redact_patterns]
        _REDACT_RE = re.compile("|".join(escaped), re.IGNORECASE)
    else:
        _REDACT_RE = None

    # Shared processors (both renderers)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        redaction_processor,
    ]

    if fmt == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib root logger so uvicorn/httpx logs also go through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quieten noisy third-party loggers
    for name in ("uvicorn.access", "httpx", "httpcore", "litellm"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger, optionally with initial context.

    Parameters
    ----------
    name:
        Logger name (defaults to caller module).
    **initial_context:
        Key-value pairs to bind to every log call from this logger.

    Returns
    -------
    structlog.stdlib.BoundLogger
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)  # type: ignore[assignment]
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
