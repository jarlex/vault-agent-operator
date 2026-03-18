"""Redaction policies — per-tool-type rules for what constitutes a secret value.

Each policy defines which fields in an MCP tool result are **safe** (metadata,
paths, key names, timestamps) and which are **secret** (actual values, private
keys, PEM content). The redactor uses these policies to strip secrets before
the tool result enters the LLM conversation.

Security principle: **allowlist-based**. Only explicitly marked safe fields
pass through. Everything else is redacted. This is safer than a blocklist
approach because new/unknown fields default to redacted.

Policy hierarchy:
    1. Tool-specific policy (e.g. KVReadRedactionPolicy for ``vault_kv_read``)
    2. Category policy (e.g. PKIRedactionPolicy for all ``vault_pki_*`` tools)
    3. DefaultRedactionPolicy (conservative: redact ALL values, keep keys)

Adding new policies:
    1. Subclass ``RedactionPolicy``
    2. Implement ``redact(tool_name, result)`` → redacted result
    3. Register the tool name(s) in ``_TOOL_POLICY_MAP``
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Any

from src.logging import get_logger

logger = get_logger(__name__)


class RedactionPolicy(ABC):
    """Base class for tool-specific redaction policies.

    Each policy knows how to transform a raw MCP tool result into a
    redacted version safe for the LLM, plus an unredacted version for
    the API consumer.
    """

    @abstractmethod
    def redact(self, tool_name: str, result: Any) -> dict[str, Any]:
        """Redact secret values from a tool result.

        Parameters
        ----------
        tool_name:
            The MCP tool that produced this result.
        result:
            The raw tool result (typically a string or dict from MCP).

        Returns
        -------
        dict[str, Any]
            A dict with two keys:
            - ``"redacted"``: The redacted result safe for the LLM.
            - ``"unredacted"``: The original, full result for the API consumer.
        """
        ...


class KVReadRedactionPolicy(RedactionPolicy):
    """Redaction policy for KV secret read operations.

    **Safe for LLM** (metadata):
        - Secret path
        - Key names (the keys of the data dict, NOT the values)
        - Metadata: version, created_time, deletion_time, destroyed, custom_metadata
        - Mount point

    **Redacted** (secret values):
        - All values in the ``data`` dict
        - Any field not explicitly in the safe list
    """

    # Fields in KV metadata that are safe to pass to the LLM
    SAFE_METADATA_FIELDS: frozenset[str] = frozenset({
        "version",
        "created_time",
        "deletion_time",
        "destroyed",
        "custom_metadata",
        "cas_required",
        "delete_version_after",
        "max_versions",
        "oldest_version",
        "current_version",
    })

    def redact(self, tool_name: str, result: Any) -> dict[str, Any]:
        """Redact KV read result: keep path, key names, metadata; remove values."""
        parsed = _parse_result(result)
        unredacted = copy.deepcopy(parsed)

        redacted: dict[str, Any] = {}

        # Preserve path if present
        for path_key in ("path", "mount", "mount_path"):
            if path_key in parsed:
                redacted[path_key] = parsed[path_key]

        # Extract key names from data, redact values
        if "data" in parsed and isinstance(parsed["data"], dict):
            # KV v2 wraps in data.data
            inner_data = parsed["data"]
            if "data" in inner_data and isinstance(inner_data["data"], dict):
                # KV v2 format: {data: {data: {key: val}, metadata: {...}}}
                redacted["keys"] = list(inner_data["data"].keys())
                redacted["key_count"] = len(inner_data["data"])
                if "metadata" in inner_data and isinstance(inner_data["metadata"], dict):
                    redacted["metadata"] = _filter_safe_fields(
                        inner_data["metadata"], self.SAFE_METADATA_FIELDS
                    )
            else:
                # KV v1 format or flat: {data: {key: val}}
                redacted["keys"] = list(inner_data.keys())
                redacted["key_count"] = len(inner_data)
        elif "keys" in parsed and isinstance(parsed["keys"], list):
            # Already a list response (e.g. from list operation)
            redacted["keys"] = parsed["keys"]

        # Preserve metadata at top level if present
        if "metadata" in parsed and isinstance(parsed["metadata"], dict):
            redacted["metadata"] = _filter_safe_fields(
                parsed["metadata"], self.SAFE_METADATA_FIELDS
            )

        redacted["note"] = (
            "Secret values retrieved successfully. Values are available in the "
            "API response but are not shown here for security."
        )

        return {"redacted": redacted, "unredacted": unredacted}


class KVWriteRedactionPolicy(RedactionPolicy):
    """Redaction policy for KV secret write operations.

    Write operations typically return metadata about the write (version,
    created_time) without echoing back the values. Still, we redact
    conservatively to catch any unexpected value echo.

    **Safe for LLM**: path, version, created_time, status.
    **Redacted**: Any field that might echo back written values.
    """

    SAFE_FIELDS: frozenset[str] = frozenset({
        "path",
        "mount",
        "mount_path",
        "version",
        "created_time",
        "deletion_time",
        "destroyed",
        "request_id",
        "lease_id",
        "lease_duration",
        "renewable",
    })

    def redact(self, tool_name: str, result: Any) -> dict[str, Any]:
        """Redact KV write result: keep write metadata, strip any echoed values."""
        parsed = _parse_result(result)
        unredacted = copy.deepcopy(parsed)

        redacted: dict[str, Any] = {}

        # Preserve safe top-level fields
        for key in self.SAFE_FIELDS:
            if key in parsed:
                redacted[key] = parsed[key]

        # If there's a data section with metadata (KV v2 write response)
        if "data" in parsed and isinstance(parsed["data"], dict):
            safe_data: dict[str, Any] = {}
            for key in ("version", "created_time", "deletion_time", "destroyed",
                        "custom_metadata"):
                if key in parsed["data"]:
                    safe_data[key] = parsed["data"][key]
            if safe_data:
                redacted["write_metadata"] = safe_data

        redacted["note"] = "Write operation completed. Result metadata shown above."

        return {"redacted": redacted, "unredacted": unredacted}


class PKIRedactionPolicy(RedactionPolicy):
    """Redaction policy for PKI certificate operations.

    **Safe for LLM** (certificate metadata):
        - Serial number
        - Expiry / not_before / not_after
        - Common Name (CN)
        - Subject Alternative Names (SANs)
        - Issuer
        - Certificate fingerprint
        - Revocation status

    **Redacted** (secret material):
        - Private key (PEM content)
        - Certificate PEM content
        - CA chain PEM content
        - Any ``-----BEGIN`` PEM blocks
    """

    SAFE_FIELDS: frozenset[str] = frozenset({
        "serial_number",
        "serial",
        "expiration",
        "not_before",
        "not_after",
        "issuing_ca",
        "ca_chain",  # NOTE: only names/paths, not PEM content — handled below
        "common_name",
        "alt_names",
        "ip_sans",
        "uri_sans",
        "ou",
        "organization",
        "country",
        "locality",
        "province",
        "street_address",
        "postal_code",
        "revocation_time",
        "revocation_time_rfc3339",
    })

    # Fields that contain PEM content — always redact
    PEM_FIELDS: frozenset[str] = frozenset({
        "private_key",
        "private_key_type",
        "certificate",
        "issuing_ca",
        "ca_chain",
        "csr",
    })

    def redact(self, tool_name: str, result: Any) -> dict[str, Any]:
        """Redact PKI result: keep metadata (serial, CN, expiry); remove keys and PEM."""
        parsed = _parse_result(result)
        unredacted = copy.deepcopy(parsed)

        redacted: dict[str, Any] = {}

        # Walk all fields
        data = parsed.get("data", parsed) if isinstance(parsed, dict) else parsed
        if isinstance(data, dict):
            for key, value in data.items():
                if key in self.PEM_FIELDS:
                    # Check if it contains PEM content
                    if isinstance(value, str) and "-----BEGIN" in value:
                        redacted[key] = "[REDACTED_PEM_CONTENT]"
                    elif key == "private_key_type":
                        redacted[key] = value  # type name is safe (e.g. "rsa")
                    else:
                        redacted[key] = "[REDACTED]"
                elif key in self.SAFE_FIELDS:
                    # Safe metadata — but still check for embedded PEM
                    if isinstance(value, str) and "-----BEGIN" in value:
                        redacted[key] = "[REDACTED_PEM_CONTENT]"
                    elif isinstance(value, list):
                        # e.g. ca_chain could be a list of PEM strings
                        redacted[key] = [
                            "[REDACTED_PEM_CONTENT]" if isinstance(v, str) and "-----BEGIN" in v
                            else v
                            for v in value
                        ]
                    else:
                        redacted[key] = value
                else:
                    # Unknown field — conservative redaction
                    redacted[key] = _conservative_redact_value(value)

        redacted["note"] = (
            "PKI certificate metadata shown above. Private key and PEM content "
            "are available in the API response but are not shown here for security."
        )

        return {"redacted": redacted, "unredacted": unredacted}


class DefaultRedactionPolicy(RedactionPolicy):
    """Conservative default policy for unknown/unrecognized tools.

    Security principle: when in doubt, redact EVERYTHING except key names and
    structural metadata. This ensures that new tools added to vault-mcp-server
    don't accidentally leak secrets through the LLM.

    **Safe**: Key names (dict keys), list lengths, data types.
    **Redacted**: ALL values.
    """

    def redact(self, tool_name: str, result: Any) -> dict[str, Any]:
        """Apply conservative redaction: keep all keys, redact all values."""
        parsed = _parse_result(result)
        unredacted = copy.deepcopy(parsed)

        redacted = _conservative_redact(parsed)

        logger.warning(
            "redaction.default_policy_used",
            tool_name=tool_name,
            message=(
                f"Tool '{tool_name}' has no specific redaction policy. "
                "Conservative redaction applied — all values redacted."
            ),
        )

        return {"redacted": redacted, "unredacted": unredacted}


# ---------------------------------------------------------------------------
# Tool → Policy Mapping
# ---------------------------------------------------------------------------

# Map of tool name patterns to policy instances.
# Keys are matched as prefixes or exact names.
_TOOL_POLICY_MAP: dict[str, RedactionPolicy] = {}

# Category mappings: tool name prefix → policy
_CATEGORY_POLICY_MAP: dict[str, RedactionPolicy] = {
    "vault_kv_read": KVReadRedactionPolicy(),
    "vault_kv_get": KVReadRedactionPolicy(),
    "vault_kv_list": KVReadRedactionPolicy(),  # list may return paths (safe) but we handle it
    "vault_kv_write": KVWriteRedactionPolicy(),
    "vault_kv_put": KVWriteRedactionPolicy(),
    "vault_kv_delete": KVWriteRedactionPolicy(),
    "vault_kv_patch": KVWriteRedactionPolicy(),
    "vault_pki_issue": PKIRedactionPolicy(),
    "vault_pki_read": PKIRedactionPolicy(),
    "vault_pki_list": PKIRedactionPolicy(),
    "vault_pki_revoke": PKIRedactionPolicy(),
    "vault_pki_sign": PKIRedactionPolicy(),
    "vault_pki_generate": PKIRedactionPolicy(),
    # Mount operations typically don't return secrets — but use conservative anyway
    # since they could theoretically contain config values
}

# Singleton default policy
_DEFAULT_POLICY = DefaultRedactionPolicy()


def get_policy_for_tool(tool_name: str) -> RedactionPolicy:
    """Look up the redaction policy for a given MCP tool name.

    Resolution order:
    1. Exact match in ``_TOOL_POLICY_MAP`` (custom overrides)
    2. Exact match in ``_CATEGORY_POLICY_MAP``
    3. Prefix match in ``_CATEGORY_POLICY_MAP`` (e.g. ``vault_kv_`` prefix)
    4. ``DefaultRedactionPolicy`` (conservative: redact ALL values)

    Parameters
    ----------
    tool_name:
        The MCP tool name (e.g. ``"vault_kv_read"``, ``"vault_pki_issue"``).

    Returns
    -------
    RedactionPolicy
        The appropriate policy for this tool.
    """
    # 1. Custom override
    if tool_name in _TOOL_POLICY_MAP:
        return _TOOL_POLICY_MAP[tool_name]

    # 2. Exact category match
    if tool_name in _CATEGORY_POLICY_MAP:
        return _CATEGORY_POLICY_MAP[tool_name]

    # 3. Prefix match — find the longest matching prefix
    best_match: str | None = None
    for prefix in _CATEGORY_POLICY_MAP:
        if tool_name.startswith(prefix) and (
            best_match is None or len(prefix) > len(best_match)
        ):
            best_match = prefix

    if best_match is not None:
        return _CATEGORY_POLICY_MAP[best_match]

    # 4. Default conservative policy
    return _DEFAULT_POLICY


def register_policy(tool_name: str, policy: RedactionPolicy) -> None:
    """Register a custom redaction policy for a specific tool.

    This allows extending the redaction layer for new or custom MCP tools
    without modifying this module.

    Parameters
    ----------
    tool_name:
        The exact MCP tool name.
    policy:
        The policy instance to use for this tool.
    """
    _TOOL_POLICY_MAP[tool_name] = policy
    logger.info("redaction.policy_registered", tool_name=tool_name, policy=type(policy).__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_result(result: Any) -> Any:
    """Parse an MCP tool result into a Python object.

    MCP tool results may be strings (JSON-encoded), dicts, or other types.
    This normalizes them to a usable Python structure.
    """
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result


def _filter_safe_fields(data: dict[str, Any], safe_fields: frozenset[str]) -> dict[str, Any]:
    """Return only the fields from ``data`` that are in ``safe_fields``."""
    return {k: v for k, v in data.items() if k in safe_fields}


def _conservative_redact(data: Any) -> Any:
    """Recursively redact ALL values while preserving structure and keys.

    - Dicts: keep keys, redact all values
    - Lists: show length and redact all elements
    - Strings/numbers/bools: redact entirely
    """
    if isinstance(data, dict):
        return {k: _conservative_redact(v) for k, v in data.items()}
    elif isinstance(data, list):
        return f"[list of {len(data)} items — values redacted]"
    elif isinstance(data, str):
        if data.startswith("-----BEGIN"):
            return "[REDACTED_PEM_CONTENT]"
        return "[REDACTED]"
    elif isinstance(data, (int, float)):
        # Numbers could be safe metadata (versions, counts) or secret
        # Conservative: redact unless it's clearly a version/count
        return "[REDACTED]"
    elif isinstance(data, bool):
        # Booleans are generally safe metadata
        return data
    elif data is None:
        return None
    else:
        return "[REDACTED]"


def _conservative_redact_value(value: Any) -> Any:
    """Redact a single value conservatively.

    Same as ``_conservative_redact`` but for a single value, not recursive.
    """
    return _conservative_redact(value)
