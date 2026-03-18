#!/usr/bin/env bash
# =============================================================================
# test-mvp.sh — Automated smoke test for vault-operator-agent MVP
#
# Tests the agent API endpoints against a running docker-compose stack.
#
# Usage:
#   ./scripts/test-mvp.sh [base_url]
#
# Environment:
#   BASE_URL — Agent base URL (default: http://localhost:8000)
#   TIMEOUT  — Max seconds to wait for services (default: 120)
# =============================================================================
set -euo pipefail

BASE_URL="${1:-${BASE_URL:-http://localhost:8000}}"
TIMEOUT="${TIMEOUT:-120}"
API_PREFIX="/api/v1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
SKIP=0
TOTAL=0

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log_test() {
    TOTAL=$((TOTAL + 1))
    echo -e "\n${YELLOW}[TEST ${TOTAL}]${NC} $1"
}

log_pass() {
    PASS=$((PASS + 1))
    echo -e "  ${GREEN}PASS${NC}: $1"
}

log_fail() {
    FAIL=$((FAIL + 1))
    echo -e "  ${RED}FAIL${NC}: $1"
}

log_skip() {
    SKIP=$((SKIP + 1))
    echo -e "  ${YELLOW}SKIP${NC}: $1"
}

# Make an API request and capture response + status code
# Usage: api_call METHOD PATH [DATA]
# Sets: RESPONSE, HTTP_CODE
api_call() {
    local method="$1"
    local path="$2"
    local data="${3:-}"
    local url="${BASE_URL}${API_PREFIX}${path}"

    local tmp_file
    tmp_file=$(mktemp)

    if [ -n "${data}" ]; then
        HTTP_CODE=$(curl -s -o "${tmp_file}" -w "%{http_code}" \
            -X "${method}" \
            -H "Content-Type: application/json" \
            -d "${data}" \
            --max-time 120 \
            "${url}" 2>/dev/null) || HTTP_CODE="000"
    else
        HTTP_CODE=$(curl -s -o "${tmp_file}" -w "%{http_code}" \
            -X "${method}" \
            --max-time 30 \
            "${url}" 2>/dev/null) || HTTP_CODE="000"
    fi

    RESPONSE=$(cat "${tmp_file}" 2>/dev/null || echo "")
    rm -f "${tmp_file}"
}

# Check if response is valid JSON
is_json() {
    echo "$1" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null
}

# Extract a field from JSON response
json_field() {
    echo "${RESPONSE}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    keys = '$1'.split('.')
    val = data
    for k in keys:
        val = val[k]
    print(val)
except:
    print('')
" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Wait for services to be healthy
# ---------------------------------------------------------------------------
echo "============================================================"
echo " vault-operator-agent MVP Smoke Tests"
echo "============================================================"
echo ""
echo "Base URL: ${BASE_URL}"
echo "Timeout:  ${TIMEOUT}s"
echo ""

echo "--- Waiting for agent to be healthy..."
SECONDS_WAITED=0
HEALTH_OK=false

while [ "${SECONDS_WAITED}" -lt "${TIMEOUT}" ]; do
    health_response=$(curl -sf --max-time 5 "${BASE_URL}${API_PREFIX}/health" 2>/dev/null) || health_response=""

    if [ -n "${health_response}" ]; then
        health_status=$(echo "${health_response}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
        if [ "${health_status}" = "healthy" ] || [ "${health_status}" = "degraded" ]; then
            echo "    Agent is ${health_status} (waited ${SECONDS_WAITED}s)"
            HEALTH_OK=true
            break
        fi
    fi

    sleep 3
    SECONDS_WAITED=$((SECONDS_WAITED + 3))
    echo "    Waiting... (${SECONDS_WAITED}s / ${TIMEOUT}s)"
done

if [ "${HEALTH_OK}" != "true" ]; then
    echo ""
    echo -e "${RED}ERROR: Agent not healthy after ${TIMEOUT}s. Aborting tests.${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo " Running Tests"
echo "============================================================"

# ---------------------------------------------------------------------------
# Test 1: GET /api/v1/health
# ---------------------------------------------------------------------------
log_test "GET /health — returns healthy status"
api_call GET "/health"

if [ "${HTTP_CODE}" = "200" ]; then
    if is_json "${RESPONSE}"; then
        status=$(json_field "status")
        if [ "${status}" = "healthy" ] || [ "${status}" = "degraded" ]; then
            log_pass "HTTP 200, status=${status}"
        else
            log_fail "Unexpected status: ${status}"
        fi
    else
        log_fail "Response is not valid JSON"
    fi
else
    log_fail "Expected HTTP 200, got ${HTTP_CODE}"
fi

# ---------------------------------------------------------------------------
# Test 2: GET /api/v1/health — valid JSON structure
# ---------------------------------------------------------------------------
log_test "GET /health — response has required fields"
api_call GET "/health"

if [ "${HTTP_CODE}" = "200" ] && is_json "${RESPONSE}"; then
    has_fields=$(echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
required = ['status']
missing = [f for f in required if f not in data]
print('ok' if not missing else ','.join(missing))
" 2>/dev/null)
    if [ "${has_fields}" = "ok" ]; then
        log_pass "All required fields present"
    else
        log_fail "Missing fields: ${has_fields}"
    fi
else
    log_fail "Could not verify fields (HTTP ${HTTP_CODE})"
fi

# ---------------------------------------------------------------------------
# Test 3: GET /api/v1/models
# ---------------------------------------------------------------------------
log_test "GET /models — returns available models"
api_call GET "/models"

if [ "${HTTP_CODE}" = "200" ]; then
    if is_json "${RESPONSE}"; then
        model_count=$(echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('available_models', data.get('models', []))
print(len(models))
" 2>/dev/null || echo "0")
        if [ "${model_count}" -gt "0" ]; then
            log_pass "HTTP 200, ${model_count} model(s) available"
        else
            log_fail "No models returned"
        fi
    else
        log_fail "Response is not valid JSON"
    fi
else
    log_fail "Expected HTTP 200, got ${HTTP_CODE}"
fi

# ---------------------------------------------------------------------------
# Test 4: POST /api/v1/tasks — "list secrets"
# ---------------------------------------------------------------------------
log_test "POST /tasks — list secrets (may require LLM)"
api_call POST "/tasks" '{"prompt": "list secrets under the path secret/"}'

if [ "${HTTP_CODE}" = "200" ]; then
    if is_json "${RESPONSE}"; then
        status=$(json_field "status")
        if [ "${status}" = "completed" ]; then
            log_pass "HTTP 200, status=completed"
        elif [ "${status}" = "error" ]; then
            error=$(json_field "error")
            log_fail "Agent returned error: ${error}"
        else
            log_fail "Unexpected status: ${status}"
        fi
    else
        log_fail "Response is not valid JSON"
    fi
elif [ "${HTTP_CODE}" = "503" ]; then
    log_skip "LLM/MCP not available (HTTP 503) — infrastructure dependency"
elif [ "${HTTP_CODE}" = "000" ]; then
    log_fail "Connection failed or timeout"
else
    log_fail "Expected HTTP 200, got ${HTTP_CODE}"
fi

# ---------------------------------------------------------------------------
# Test 5: POST /api/v1/tasks — "read specific secret"
# ---------------------------------------------------------------------------
log_test "POST /tasks — read secret at secret/myapp/database"
api_call POST "/tasks" '{"prompt": "read the secret at secret/myapp/database"}'

if [ "${HTTP_CODE}" = "200" ]; then
    if is_json "${RESPONSE}"; then
        status=$(json_field "status")
        if [ "${status}" = "completed" ]; then
            result=$(json_field "result")
            if [ -n "${result}" ]; then
                log_pass "HTTP 200, status=completed, result has content"
            else
                log_fail "Result is empty"
            fi
        elif [ "${status}" = "error" ]; then
            error=$(json_field "error")
            log_fail "Agent returned error: ${error}"
        else
            log_fail "Unexpected status: ${status}"
        fi
    else
        log_fail "Response is not valid JSON"
    fi
elif [ "${HTTP_CODE}" = "503" ]; then
    log_skip "LLM/MCP not available (HTTP 503) — infrastructure dependency"
elif [ "${HTTP_CODE}" = "000" ]; then
    log_fail "Connection failed or timeout"
else
    log_fail "Expected HTTP 200, got ${HTTP_CODE}"
fi

# ---------------------------------------------------------------------------
# Test 6: POST /api/v1/tasks — missing prompt (validation)
# ---------------------------------------------------------------------------
log_test "POST /tasks — missing prompt returns 422"
api_call POST "/tasks" '{}'

if [ "${HTTP_CODE}" = "422" ] || [ "${HTTP_CODE}" = "400" ]; then
    log_pass "Validation error returned (HTTP ${HTTP_CODE})"
else
    log_fail "Expected HTTP 422 or 400, got ${HTTP_CODE}"
fi

# ---------------------------------------------------------------------------
# Test 7: Invalid endpoint returns 404
# ---------------------------------------------------------------------------
log_test "GET /nonexistent — returns 404"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    "${BASE_URL}${API_PREFIX}/nonexistent" 2>/dev/null) || HTTP_CODE="000"

if [ "${HTTP_CODE}" = "404" ]; then
    log_pass "HTTP 404 for unknown endpoint"
else
    log_fail "Expected HTTP 404, got ${HTTP_CODE}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Test Results"
echo "============================================================"
echo ""
echo -e "  ${GREEN}PASSED${NC}: ${PASS}"
echo -e "  ${RED}FAILED${NC}: ${FAIL}"
echo -e "  ${YELLOW}SKIPPED${NC}: ${SKIP}"
echo -e "  TOTAL:   ${TOTAL}"
echo ""

if [ "${FAIL}" -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
