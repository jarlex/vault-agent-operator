"""Secret Redaction Layer for vault-operator-agent.

This package is the **security core** of the agent. It ensures that secret
values retrieved from Vault NEVER reach the LLM, while still allowing the
API consumer to receive the full unredacted data they requested.

Architecture:

    Consumer ──prompt──▶ PromptSanitizer ──sanitized prompt──▶ AgentCore
                                                                    │
                              SecretContext (request-scoped)         │
                              ┌──────────────────────────┐         │
                              │ placeholder ↔ real value  │◀────────┤
                              │ unredacted tool responses │         │
                              └──────────────────────────┘         │
                                                                    │
    MCP Result ──▶ SecretRedactor ──redacted result──▶ LLM conversation
                       │
                       └── policies.py (per-tool redaction rules)

Components:

- **SecretContext**: Request-scoped mapping of placeholder tokens to real secret
  values. Created at request start, destroyed at request end. Not serializable,
  not loggable.

- **SecretRedactor**: Intercepts MCP tool results and removes secret values
  before they enter the LLM conversation. Stores unredacted responses in the
  SecretContext for the API consumer. Also handles restoring placeholders to
  real values before MCP tool calls.

- **RedactionPolicy / policies**: Per-tool-type rules defining which fields
  are secret and which are safe metadata. Conservative default for unknown tools.

- **PromptSanitizer**: Extracts secret values from consumer prompts and
  ``secret_data`` fields, replacing them with opaque placeholder tokens before
  the prompt reaches the LLM.

Security guarantees:
    1. Secret values are NEVER included in any LLM request payload.
    2. Key names, paths, and metadata ARE passed to the LLM for reasoning.
    3. The API consumer receives full unredacted data.
    4. SecretContext is destroyed after each request.
    5. Unknown tools use conservative redaction (all values redacted).
"""

from src.agent.redaction.context import SecretContext
from src.agent.redaction.policies import (
    DefaultRedactionPolicy,
    KVReadRedactionPolicy,
    KVWriteRedactionPolicy,
    PKIRedactionPolicy,
    RedactionPolicy,
    get_policy_for_tool,
)
from src.agent.redaction.redactor import SecretRedactor
from src.agent.redaction.sanitizer import PromptSanitizer

__all__ = [
    "SecretContext",
    "SecretRedactor",
    "PromptSanitizer",
    "RedactionPolicy",
    "DefaultRedactionPolicy",
    "KVReadRedactionPolicy",
    "KVWriteRedactionPolicy",
    "PKIRedactionPolicy",
    "get_policy_for_tool",
]
