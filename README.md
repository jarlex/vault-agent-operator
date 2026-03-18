# vault-agent-operator

AI agent that accepts natural-language prompts via a REST API and executes
HashiCorp Vault operations through the official `vault-mcp-server`. Written in
Go as a single static binary.

## Overview

`vault-agent-operator` is a stateless HTTP service that:

- Receives natural-language instructions via `POST /api/v1/tasks`
- Runs a hybrid LLM reasoning loop (via go-openai) with tool-calling capabilities
- Executes Vault operations by forwarding tool calls to the official HashiCorp
  `vault-mcp-server` (MCP protocol over stdio)
- Enforces mTLS authentication for API consumers
- Redacts secret values from the LLM context (secrets never reach the model)
- Supports scheduled periodic tasks (health checks, rotation reminders)
- Ships as a single binary in a distroless Docker image (<50 MB)

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
docker compose -f docker-compose.mvp.yaml up --build
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
docker compose -f docker-compose.mvp.yaml down
```

## Requirements

- Go 1.26.1+ (for local development)
- Docker & Docker Compose (for deployment)
- HashiCorp Vault instance (or use the bundled dev server)
- GitHub PAT with `models:read` scope (for GitHub Models LLM access)

## Project Structure

```
vault-agent-operator/
├── cmd/
│   └── vault-agent-operator/
│       └── main.go                    # Entry point, DI wiring, graceful shutdown
├── internal/
│   ├── agent/                         # AgentCore, hybrid reasoning loop, prompts
│   ├── api/                           # chi v5 HTTP server, handlers, middleware
│   ├── config/                        # Viper-based config loading + validation
│   ├── llm/                           # go-openai LLM provider, retry, error types
│   ├── logging/                       # zerolog setup, secret redaction
│   ├── mcp/                           # MCP client (mcp-go, stdio transport)
│   ├── redaction/                     # SecretContext, redactor, sanitizer, policies
│   └── scheduler/                     # Cron engine (robfig/cron v3)
├── config/
│   ├── default.yaml                   # Base configuration
│   └── prompts/
│       └── system.md                  # System prompt template
├── scripts/
│   ├── generate-certs.sh              # mTLS certificate generation
│   ├── healthcheck.sh                 # Docker HEALTHCHECK (curl-based)
│   ├── seed-vault.sh                  # Seed Vault with test data
│   └── test-mvp.sh                    # E2E smoke test
├── CONTRIBUTING.md                    # Contribution guidelines
├── Dockerfile                         # Multi-stage: golang → distroless
├── LICENSE                            # MIT License
├── Makefile                           # Build, test, lint, docker targets
├── docker-compose.yaml                # Production-like stack
├── docker-compose.mvp.yaml            # MVP stack (no mTLS, pre-seeded Vault)
├── go.mod
└── go.sum
```

## Dependencies

| Concern | Library |
|---------|---------|
| HTTP router | [chi v5](https://github.com/go-chi/chi) |
| LLM client | [go-openai](https://github.com/sashabaranov/go-openai) |
| MCP client | [mcp-go](https://github.com/mark3labs/mcp-go) |
| Configuration | [Viper](https://github.com/spf13/viper) |
| Logging | [zerolog](https://github.com/rs/zerolog) |
| Scheduler | [cron v3](https://github.com/robfig/cron) |

## Installation (Development)

```bash
# Clone and build
git clone https://github.com/jarlex/vault-agent-operator.git
cd vault-agent-operator

# Build the binary (with version info via ldflags)
make build

# Run tests with race detector
make test

# Run static analysis
make lint
```

Or without the Makefile:

```bash
# Build the binary
go build -o vault-agent-operator ./cmd/vault-agent-operator/

# Run tests
go test ./... -v -race

# Run vet
go vet ./...
```

## Configuration

Configuration is loaded from `config/default.yaml` with environment variable
overrides. The double-underscore `__` delimiter maps to nested YAML keys
(e.g. `LLM__DEFAULT_MODEL` overrides `llm.default_model`).

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub PAT for LLM access (`models:read` scope) |
| `VAULT_ADDR` | Vault server address (e.g. `http://vault:8200`) |
| `VAULT_TOKEN` | Vault authentication token |

### Complete Configuration Reference

All options from `config/default.yaml` with their environment variable overrides:

#### API Server

| YAML Path | Env Override | Default | Description |
|-----------|-------------|---------|-------------|
| `api.host` | `API__HOST` | `0.0.0.0` | Bind address |
| `api.port` | `API__PORT` | `8000` | HTTP listen port |
| `api.request_timeout` | `API__REQUEST_TIMEOUT` | `120` | Per-request timeout (seconds) |

#### mTLS Authentication

| YAML Path | Env Override | Default | Description |
|-----------|-------------|---------|-------------|
| `mtls.enabled` | `MTLS__ENABLED` | `true` | Enable mTLS for API consumers |
| `mtls.ca_cert_path` | `MTLS__CA_CERT_PATH` | `/certs/ca.pem` | CA certificate path |
| `mtls.server_cert_path` | `MTLS__SERVER_CERT_PATH` | `/certs/server.pem` | Server certificate path |
| `mtls.server_key_path` | `MTLS__SERVER_KEY_PATH` | `/certs/server-key.pem` | Server private key path |

#### MCP (Model Context Protocol)

| YAML Path | Env Override | Default | Description |
|-----------|-------------|---------|-------------|
| `mcp.transport` | `MCP__TRANSPORT` | `stdio` | Transport type: `stdio` (subprocess) or `http` (sidecar SSE) |
| `mcp.server_binary` | `MCP__SERVER_BINARY` | `/usr/local/bin/vault-mcp-server` | Path to vault-mcp-server binary (stdio mode) |
| `mcp.server_url` | `MCP__SERVER_URL` | `http://vault-mcp-server:3000` | Server URL (http/SSE mode) |
| `mcp.vault_addr` | `VAULT_ADDR` | `http://vault:8200` | Vault server address |
| `mcp.vault_token` | `VAULT_TOKEN` | *(empty)* | Vault authentication token |
| `mcp.tool_timeout` | `MCP__TOOL_TIMEOUT` | `30` | Per-tool invocation timeout (seconds) |
| `mcp.reconnect_initial_delay` | `MCP__RECONNECT_INITIAL_DELAY` | `1.0` | Reconnect backoff initial delay (seconds) |
| `mcp.reconnect_max_delay` | `MCP__RECONNECT_MAX_DELAY` | `60.0` | Reconnect backoff ceiling (seconds) |

#### LLM Provider

| YAML Path | Env Override | Default | Description |
|-----------|-------------|---------|-------------|
| `llm.default_model` | `LLM__DEFAULT_MODEL` | `default` | Default model alias (see model list below) |
| `llm.request_timeout` | `LLM__REQUEST_TIMEOUT` | `60` | LLM request timeout (seconds) |
| `llm.max_retries` | `LLM__MAX_RETRIES` | `3` | Max retry attempts for failed LLM requests |

Pre-configured model aliases:

| Alias | Provider | Model ID | Tool Calling |
|-------|----------|----------|--------------|
| `default` | GitHub Models | `gpt-4o` | Yes |
| `fast` | GitHub Models | `gpt-4o-mini` | Yes |
| `llama` | GitHub Models | `meta-llama-3.1-70b-instruct` | Yes |

#### Agent

| YAML Path | Env Override | Default | Description |
|-----------|-------------|---------|-------------|
| `agent.max_iterations` | `AGENT__MAX_ITERATIONS` | `10` | Max reasoning loop iterations per task |
| `agent.system_prompt_path` | `AGENT__SYSTEM_PROMPT_PATH` | `config/prompts/system.md` | Path to the system prompt template |

#### Scheduler

| YAML Path | Env Override | Default | Description |
|-----------|-------------|---------|-------------|
| `scheduler.enabled` | `SCHEDULER__ENABLED` | `true` | Enable the cron task scheduler |

Pre-configured scheduled tasks:

| Task ID | Cron | Description | Default |
|---------|------|-------------|---------|
| `health_check` | `*/5 * * * *` | Check Vault health and report status | Enabled |
| `rotation_reminder` | `0 9 * * 1` | Identify secrets that may need rotation | Disabled |

#### Logging

| YAML Path | Env Override | Default | Description |
|-----------|-------------|---------|-------------|
| `logging.level` | `LOGGING__LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARN`, `ERROR`) |
| `logging.format` | `LOGGING__FORMAT` | `json` | Log format (`json` or `console`) |
| `logging.redact_patterns` | `LOGGING__REDACT_PATTERNS` | *(see below)* | Field name patterns to redact from logs |

Default redact patterns: `token`, `password`, `secret`, `key`, `authorization`,
`credential`, `private`.

## Usage

### Start the agent (local development)

```bash
# Development (no mTLS)
MTLS__ENABLED=false \
  VAULT_ADDR=http://localhost:8200 \
  VAULT_TOKEN=dev-token \
  GITHUB_TOKEN=ghp_... \
  go run ./cmd/vault-agent-operator/

# Or run the compiled binary
MTLS__ENABLED=false \
  VAULT_ADDR=http://localhost:8200 \
  VAULT_TOKEN=dev-token \
  GITHUB_TOKEN=ghp_... \
  ./vault-agent-operator
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check (agent, MCP, Vault status) |
| `GET` | `/api/v1/models` | List available LLM models |
| `POST` | `/api/v1/tasks` | Execute a Vault operation via natural language |

#### POST /api/v1/tasks

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Read the secret at secret/myapp/db"}'
```

Request body:

```json
{
  "prompt": "Read the secret at secret/myapp/db",
  "model": "default",
  "max_iterations": 10,
  "secret_data": {}
}
```

Response:

```json
{
  "status": "completed",
  "result": "...",
  "data": [...],
  "model_used": "default",
  "duration_ms": 1234,
  "error": null
}
```

## Docker

### Production-like stack (with mTLS)

```bash
# Generate mTLS certificates
./scripts/generate-certs.sh

# Set tokens
export GITHUB_TOKEN=ghp_...

# Start with mTLS enabled
docker compose up --build
```

### MVP stack (no mTLS, pre-seeded Vault)

```bash
export GITHUB_TOKEN=ghp_...
docker compose -f docker-compose.mvp.yaml up --build
```

### Build with version info

Using the Makefile:

```bash
make docker-build
```

Or manually:

```bash
docker compose build \
  --build-arg VERSION=$(git describe --tags --always) \
  --build-arg COMMIT=$(git rev-parse --short HEAD) \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
```

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/generate-certs.sh` | Generate self-signed CA + server + client certs for mTLS |
| `scripts/healthcheck.sh` | Docker HEALTHCHECK script (supports mTLS and plain HTTP) |
| `scripts/seed-vault.sh` | Pre-seed Vault with test KV secrets, PKI engine, and policies |
| `scripts/test-mvp.sh` | Automated smoke test against a running agent |

## Architecture

```
Consumer -> [mTLS] -> chi router -> Agent Core -> LLM (go-openai)
                                       |
                                       +-> MCP Client (mcp-go) -> vault-mcp-server -> Vault
```

### Security Model

The LLM never sees actual secret values:

1. **Inbound**: User prompt is sanitized — detected secrets are replaced with
   placeholders before reaching the LLM.
2. **Tool arguments**: LLM produces tool calls with placeholders. The agent
   restores real values before forwarding to vault-mcp-server.
3. **Success path**: Raw tool results are returned directly to the API consumer.
   The LLM receives only a `{"status":"ok"}` acknowledgment.
4. **Error path**: Tool errors are redacted before being fed back to the LLM
   for retry reasoning.

### Hybrid Reasoning Loop

The agent uses a hybrid reasoning loop that optimizes for the common case:

- **Fast path**: When all tool calls succeed, raw results are returned
  immediately to the consumer. No second LLM call is needed.
- **Error retry**: When a tool call fails, the redacted error is fed back to
  the LLM, which can adjust arguments and retry.
- **Text response**: If the LLM responds with text (no tool calls), the text
  is returned as the result.

## Development

```bash
# Run all tests with race detector
make test

# Run tests with coverage
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out

# Build for production (static binary with version info)
make build

# Clean build artifacts
make clean

# Static analysis
make lint
go vet ./...
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## License

[MIT](LICENSE)
