#!/usr/bin/env bash
# =============================================================================
# generate-certs.sh — Generate self-signed CA + server + client certificates
# for vault-operator-agent development/testing.
#
# Usage:
#   ./scripts/generate-certs.sh [output_dir]
#
# Default output: ./certs/
# =============================================================================
set -euo pipefail

CERT_DIR="${1:-./certs}"
DAYS_VALID=365
CA_SUBJECT="/CN=vault-operator-agent-ca/O=vault-operator-agent/C=US"
SERVER_SUBJECT="/CN=vault-operator-agent/O=vault-operator-agent/C=US"
CLIENT_SUBJECT="/CN=dev-client/O=vault-operator-agent/C=US"

echo "==> Generating certificates in ${CERT_DIR}"
mkdir -p "${CERT_DIR}"

# ---- Step 1: Root CA --------------------------------------------------------
echo "--- Generating Root CA..."
openssl genrsa -out "${CERT_DIR}/ca-key.pem" 4096 2>/dev/null
openssl req -new -x509 \
    -key "${CERT_DIR}/ca-key.pem" \
    -out "${CERT_DIR}/ca.pem" \
    -days "${DAYS_VALID}" \
    -subj "${CA_SUBJECT}" \
    2>/dev/null

# ---- Step 2: Server Certificate ---------------------------------------------
echo "--- Generating Server Certificate..."
openssl genrsa -out "${CERT_DIR}/server-key.pem" 4096 2>/dev/null

# Create a SAN config for the server cert
cat > "${CERT_DIR}/server-ext.cnf" <<EOF
[req]
req_extensions = v3_req
distinguished_name = req_distinguished_name

[req_distinguished_name]

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = vault-operator-agent
DNS.3 = 127.0.0.1
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
EOF

openssl req -new \
    -key "${CERT_DIR}/server-key.pem" \
    -out "${CERT_DIR}/server.csr" \
    -subj "${SERVER_SUBJECT}" \
    -config "${CERT_DIR}/server-ext.cnf" \
    2>/dev/null

openssl x509 -req \
    -in "${CERT_DIR}/server.csr" \
    -CA "${CERT_DIR}/ca.pem" \
    -CAkey "${CERT_DIR}/ca-key.pem" \
    -CAcreateserial \
    -out "${CERT_DIR}/server.pem" \
    -days "${DAYS_VALID}" \
    -extensions v3_req \
    -extfile "${CERT_DIR}/server-ext.cnf" \
    2>/dev/null

# ---- Step 3: Client Certificate ---------------------------------------------
echo "--- Generating Client Certificate..."
openssl genrsa -out "${CERT_DIR}/client-key.pem" 4096 2>/dev/null

openssl req -new \
    -key "${CERT_DIR}/client-key.pem" \
    -out "${CERT_DIR}/client.csr" \
    -subj "${CLIENT_SUBJECT}" \
    2>/dev/null

openssl x509 -req \
    -in "${CERT_DIR}/client.csr" \
    -CA "${CERT_DIR}/ca.pem" \
    -CAkey "${CERT_DIR}/ca-key.pem" \
    -CAcreateserial \
    -out "${CERT_DIR}/client.pem" \
    -days "${DAYS_VALID}" \
    2>/dev/null

# ---- Cleanup temporary files ------------------------------------------------
rm -f "${CERT_DIR}"/*.csr "${CERT_DIR}"/*.cnf "${CERT_DIR}"/*.srl

# ---- Summary ----------------------------------------------------------------
echo ""
echo "==> Certificates generated successfully in ${CERT_DIR}/"
echo ""
echo "  CA certificate:      ${CERT_DIR}/ca.pem"
echo "  CA private key:      ${CERT_DIR}/ca-key.pem"
echo "  Server certificate:  ${CERT_DIR}/server.pem"
echo "  Server private key:  ${CERT_DIR}/server-key.pem"
echo "  Client certificate:  ${CERT_DIR}/client.pem"
echo "  Client private key:  ${CERT_DIR}/client-key.pem"
echo ""
echo "Usage with curl (mTLS):"
echo "  curl --cacert ${CERT_DIR}/ca.pem \\"
echo "       --cert ${CERT_DIR}/client.pem \\"
echo "       --key ${CERT_DIR}/client-key.pem \\"
echo "       https://localhost:8000/api/v1/health"
