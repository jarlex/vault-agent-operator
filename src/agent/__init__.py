"""Agent package for vault-operator-agent.

Exports:
    - **AgentCore**: The main agent class with the reasoning loop.
    - **AgentResult**: Structured result from agent execution.
    - **ToolCallRecord**: Audit trail entry for a single tool invocation.
    - **load_system_prompt**: System prompt builder.
    - Redaction layer components via ``src.agent.redaction``.
"""

from src.agent.core import AgentCore
from src.agent.prompts import load_system_prompt
from src.agent.types import AgentResult, ToolCallRecord

__all__ = [
    "AgentCore",
    "AgentResult",
    "ToolCallRecord",
    "load_system_prompt",
]
