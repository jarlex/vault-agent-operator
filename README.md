# vault-operator-agent

AI agent that accepts natural-language prompts via a REST API and executes
HashiCorp Vault operations through the official `vault-mcp-server`.

## Overview

`vault-operator-agent` is a stateless FastAPI service that:

- Receives natural-language instructions via `POST /api/v1/tasks`
- Runs an LLM reasoning loop (via LiteLLM) with tool-calling capabilities
- Executes Vault operations by forwarding tool calls to the official HashiCorp
  `vault-mcp-server` (MCP protocol)
- Enforces mTLS authentication for API consumers
- Redacts secret values from the LLM context (secrets never reach the model)
- Supports scheduled periodic tasks (health checks, rotation reminders)

## Supported Operations

- **KV Secrets**: read, write, list, delete (via KV v2 engine)
- **Mount Management**: create, list, delete secret engines
- **PKI**: issue, list, revoke TLS certificates

## Quick Start (MVP)

Get the agent running with a single command. Only requires Docker and a
GitHub PAT with `models:read` scope.

```bash
# 1. Set your GitHub token
export GITHUB_TOKEN=ghp_your_token_here

# 2. Launch everything (Vault + seed data + agent)
docker-compose -f docker-compose.mvp.yaml up --build
```

Wait for the output to show the agent is healthy, then try:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List available LLM models
curl http://localhost:8000/api/v1/models

# List secrets in Vault
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt": "list secrets under the path secret/"}'

# Read a specific secret
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt": "read the secret at secret/myapp/database"}'

# Read an API key
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt": "read the secret at secret/myapp/api-key"}'
```

Run the automated smoke test:

```bash
./scripts/test-mvp.sh
```

To stop everything:

```bash
docker-compose -f docker-compose.mvp.yaml down
```

## Requirements

- Python 3.12+
- Docker & Docker Compose (for deployment)
- HashiCorp Vault instance (or use the bundled dev server)
- GitHub PAT with `models:read` scope (for GitHub Models LLM access)

## Installation (Development)

```bash
pip install -e ".[dev]"
```

## Configuration

Configuration is loaded from `config/default.yaml` with environment variable
overrides. Required environment variables:

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | GitHub PAT for LLM access (models:read scope) |
| `VAULT_ADDR` | Vault server address |
| `VAULT_TOKEN` | Vault authentication token |

See `config/default.yaml` for all available settings.

## Usage

### Start the agent

```bash
# Development (no mTLS)
MTLS_ENABLED=false VAULT_ADDR=http://localhost:8200 VAULT_TOKEN=dev-token \
    GITHUB_TOKEN=ghp_... python -m uvicorn src.main:app --reload

# Production (with mTLS)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List available models
curl http://localhost:8000/api/v1/models

# Execute a Vault operation
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Read the secret at secret/myapp/db"}'
```

## Docker

### Production-like stack

```bash
# Generate mTLS certificates
./scripts/generate-certs.sh

# Set tokens
export GITHUB_TOKEN=ghp_...

# Start with mTLS enabled
docker-compose up --build
```

### MVP stack (no mTLS, pre-seeded Vault)

```bash
export GITHUB_TOKEN=ghp_...
docker-compose -f docker-compose.mvp.yaml up --build
```

### Scripts

| Script | Description |
|---|---|
| `scripts/generate-certs.sh` | Generate self-signed CA + server + client certs for mTLS |
| `scripts/healthcheck.sh` | Docker HEALTHCHECK script (supports mTLS and plain HTTP) |
| `scripts/seed-vault.sh` | Pre-seed Vault with test KV secrets, PKI engine, and policies |
| `scripts/test-mvp.sh` | Automated smoke test against a running agent |

## Architecture

```
Consumer -> [mTLS] -> FastAPI -> Agent Core -> LLM (LiteLLM)
                                     |
                                     +-> MCP Client -> vault-mcp-server -> Vault
```

## License

MIT
