#!/usr/bin/env bash
# =============================================================================
# healthcheck.sh — Docker HEALTHCHECK script for vault-operator-agent
#
# Checks the /api/v1/health endpoint and exits with appropriate code.
# Supports both mTLS and non-mTLS modes.
#
# For distroless containers, use the Go binary's --healthcheck flag instead:
#   /app/vault-agent-operator --healthcheck
#
# This script is for non-distroless images or host-side health checks
# that have curl and jq available.
#
# Usage (outside container or in non-distroless image):
#   ./scripts/healthcheck.sh
#
# Environment:
#   MTLS_ENABLED — "true" to use mTLS certs, anything else for plain HTTP
#   HEALTH_URL  — override the health endpoint URL
# =============================================================================
set -euo pipefail

# Defaults
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/api/v1/health}"
MTLS_ENABLED="${MTLS__ENABLED:-false}"

if [ "${MTLS_ENABLED}" = "true" ]; then
    HEALTH_URL="https://localhost:8000/api/v1/health"
    response=$(curl -sf \
        --cacert /certs/ca.pem \
        --cert /certs/client.pem \
        --key /certs/client-key.pem \
        --max-time 5 \
        "${HEALTH_URL}" 2>/dev/null) || exit 1
else
    response=$(curl -sf --max-time 5 "${HEALTH_URL}" 2>/dev/null) || exit 1
fi

# Parse status from JSON response.
# Try jq first (preferred), fall back to grep pattern match.
if command -v jq &>/dev/null; then
    status=$(echo "${response}" | jq -r '.status // ""' 2>/dev/null || echo "")
else
    # Fallback: simple pattern extraction without jq or python
    status=$(echo "${response}" | grep -oP '"status"\s*:\s*"\K[^"]+' 2>/dev/null || echo "")
fi

case "${status}" in
    healthy|degraded)
        exit 0
        ;;
    *)
        exit 1
        ;;
esac
