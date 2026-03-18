#!/usr/bin/env bash
# =============================================================================
# healthcheck.sh — Docker HEALTHCHECK script for vault-operator-agent
#
# Checks the /api/v1/health endpoint and exits with appropriate code.
# Supports both mTLS and non-mTLS modes.
#
# Usage (inside container):
#   /app/scripts/healthcheck.sh
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

# Check if status is "healthy" or "degraded" (both are acceptable)
status=$(echo "${response}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")

case "${status}" in
    healthy|degraded)
        exit 0
        ;;
    *)
        exit 1
        ;;
esac
