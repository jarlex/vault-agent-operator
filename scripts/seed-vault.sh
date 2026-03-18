#!/usr/bin/env bash
# =============================================================================
# seed-vault.sh — Pre-seed Vault with test data for MVP testing
#
# This script populates a Vault dev server with:
#   - KV v2 secrets (myapp/database, myapp/api-key)
#   - PKI engine with root CA and role
#   - Test policies
#
# Usage:
#   ./scripts/seed-vault.sh
#
# Environment:
#   VAULT_ADDR  — Vault server address (default: http://127.0.0.1:8200)
#   VAULT_TOKEN — Vault authentication token (default: dev-token)
#
# Idempotent: safe to run multiple times.
# =============================================================================
set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-dev-token}"

export VAULT_ADDR VAULT_TOKEN

echo "==> Seeding Vault at ${VAULT_ADDR}"

# ---------------------------------------------------------------------------
# Wait for Vault to be ready
# ---------------------------------------------------------------------------
echo "--- Waiting for Vault to be ready..."
MAX_RETRIES=30
RETRY_INTERVAL=2

for i in $(seq 1 ${MAX_RETRIES}); do
    if vault status >/dev/null 2>&1; then
        echo "    Vault is ready (attempt ${i}/${MAX_RETRIES})"
        break
    fi
    if [ "${i}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: Vault not ready after ${MAX_RETRIES} attempts"
        exit 1
    fi
    echo "    Waiting for Vault... (attempt ${i}/${MAX_RETRIES})"
    sleep "${RETRY_INTERVAL}"
done

# ---------------------------------------------------------------------------
# 1. Enable KV v2 at secret/ (idempotent — ignore error if already enabled)
# ---------------------------------------------------------------------------
echo "--- Enabling KV v2 secrets engine at secret/..."
vault secrets enable -path=secret -version=2 kv 2>/dev/null || \
    echo "    KV v2 already enabled at secret/"

# ---------------------------------------------------------------------------
# 2. Write test KV secrets
# ---------------------------------------------------------------------------
echo "--- Writing test secrets..."

# Database credentials
vault kv put secret/myapp/database \
    username="dbadmin" \
    password="SuperS3cret!2026" \
    host="db.example.com" \
    port="5432" \
    database="myapp_production"
echo "    Written: secret/myapp/database"

# API key
vault kv put secret/myapp/api-key \
    key="ak_live_x7k9m2p4q8r1w6t3y5n0" \
    provider="stripe" \
    environment="production"
echo "    Written: secret/myapp/api-key"

# Additional test secrets for listing
vault kv put secret/myapp/redis \
    url="redis://redis.example.com:6379" \
    password="r3d1sP@ss"
echo "    Written: secret/myapp/redis"

vault kv put secret/team-alpha/config \
    feature_flag="enabled" \
    max_retries="5"
echo "    Written: secret/team-alpha/config"

# ---------------------------------------------------------------------------
# 3. Enable PKI engine (idempotent)
# ---------------------------------------------------------------------------
echo "--- Enabling PKI secrets engine..."
vault secrets enable -path=pki pki 2>/dev/null || \
    echo "    PKI already enabled at pki/"

# Set max TTL to 1 year
vault secrets tune -max-lease-ttl=8760h pki 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Generate Root CA (idempotent — check if already exists)
# ---------------------------------------------------------------------------
echo "--- Configuring PKI Root CA..."
existing_ca=$(vault read -format=json pki/cert/ca 2>/dev/null || echo "")

if [ -z "${existing_ca}" ] || echo "${existing_ca}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    cert = data.get('data', {}).get('certificate', '')
    if not cert:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null; then
    echo "    Root CA already exists, skipping generation"
else
    vault write pki/root/generate/internal \
        common_name="vault-operator-agent-test-ca" \
        ttl=8760h \
        >/dev/null
    echo "    Root CA generated"
fi

# Generate root CA (overwrite is safe for dev)
vault write pki/root/generate/internal \
    common_name="vault-operator-agent-test-ca" \
    ttl=8760h \
    >/dev/null 2>&1 || true
echo "    Root CA configured"

# Configure CA and CRL URLs
vault write pki/config/urls \
    issuing_certificates="${VAULT_ADDR}/v1/pki/ca" \
    crl_distribution_points="${VAULT_ADDR}/v1/pki/crl" \
    >/dev/null
echo "    PKI URLs configured"

# ---------------------------------------------------------------------------
# 5. Create PKI role for certificate issuance
# ---------------------------------------------------------------------------
echo "--- Creating PKI role 'test-role'..."
vault write pki/roles/test-role \
    allowed_domains="example.com" \
    allow_subdomains=true \
    max_ttl=720h \
    >/dev/null
echo "    PKI role 'test-role' created"

# ---------------------------------------------------------------------------
# 6. Create test policies
# ---------------------------------------------------------------------------
echo "--- Creating test policies..."

# Read-only policy for KV secrets
vault policy write kv-readonly - <<'POLICY'
# Read-only access to KV v2 secrets under secret/
path "secret/data/*" {
  capabilities = ["read", "list"]
}
path "secret/metadata/*" {
  capabilities = ["read", "list"]
}
POLICY
echo "    Policy 'kv-readonly' created"

# Full access policy for KV secrets
vault policy write kv-admin - <<'POLICY'
# Full access to KV v2 secrets under secret/
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
POLICY
echo "    Policy 'kv-admin' created"

# PKI certificate issuance policy
vault policy write pki-issue - <<'POLICY'
# Issue certificates from the PKI engine
path "pki/issue/*" {
  capabilities = ["create", "update"]
}
path "pki/cert/*" {
  capabilities = ["read"]
}
path "pki/roles/*" {
  capabilities = ["read", "list"]
}
POLICY
echo "    Policy 'pki-issue' created"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==> Vault seeding complete!"
echo ""
echo "  KV Secrets:"
echo "    - secret/myapp/database    (username, password, host, port, database)"
echo "    - secret/myapp/api-key     (key, provider, environment)"
echo "    - secret/myapp/redis       (url, password)"
echo "    - secret/team-alpha/config (feature_flag, max_retries)"
echo ""
echo "  PKI Engine:"
echo "    - Root CA: vault-operator-agent-test-ca"
echo "    - Role: test-role (*.example.com, max TTL 720h)"
echo ""
echo "  Policies:"
echo "    - kv-readonly  (read/list on secret/*)"
echo "    - kv-admin     (full CRUD on secret/*)"
echo "    - pki-issue    (issue certs from pki/)"
echo ""
echo "  Test commands:"
echo "    vault kv get secret/myapp/database"
echo "    vault kv list secret/myapp"
echo "    vault write pki/issue/test-role common_name=test.example.com ttl=24h"
