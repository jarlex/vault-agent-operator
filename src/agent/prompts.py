"""System prompt builder for the Vault Operator Agent.

Loads the system prompt from a Markdown template file and injects dynamic
context: available tools, Vault address, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.logging import get_logger

logger = get_logger(__name__)


def load_system_prompt(
    *,
    prompt_path: str = "config/prompts/system.md",
    vault_addr: str = "",
    available_tools: list[dict[str, Any]] | None = None,
) -> str:
    """Load and render the system prompt template.

    Parameters
    ----------
    prompt_path:
        Path to the Markdown prompt template file.
    vault_addr:
        The Vault server address to inject into the prompt.
    available_tools:
        List of tool definitions (OpenAI function-calling format). The prompt
        will receive a human-readable summary of tool names and descriptions.

    Returns
    -------
    str
        The fully rendered system prompt.

    Raises
    ------
    FileNotFoundError
        If the prompt template file is missing.
    """
    path = Path(prompt_path)
    if not path.is_file():
        raise FileNotFoundError(f"System prompt template not found: {prompt_path}")

    template = path.read_text(encoding="utf-8")

    # Build tool summary
    tools_summary = _format_tools_summary(available_tools or [])

    # Simple template substitution (avoids Jinja2 dependency for MVP)
    rendered = template.replace("{{ vault_addr }}", vault_addr or "not configured")
    rendered = rendered.replace("{{ available_tools }}", tools_summary or "none discovered yet")

    logger.debug(
        "system_prompt.loaded",
        prompt_path=prompt_path,
        vault_addr=vault_addr,
        tool_count=len(available_tools or []),
        prompt_length=len(rendered),
    )

    return rendered


def _format_tools_summary(tools: list[dict[str, Any]]) -> str:
    """Format tool definitions into a human-readable summary for the prompt.

    Parameters
    ----------
    tools:
        List of tool dicts in OpenAI function-calling format::

            [{"type": "function", "function": {"name": ..., "description": ...}}]

    Returns
    -------
    str
        Newline-separated list of ``- tool_name: description`` entries.
    """
    if not tools:
        return "none discovered yet"

    lines: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "no description")
        lines.append(f"- **{name}**: {desc}")

    return "\n".join(lines)
