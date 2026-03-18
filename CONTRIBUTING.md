# Contributing to vault-agent-operator

Thank you for your interest in contributing! This guide will help you get
started with the development workflow.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Go](https://go.dev/dl/) | 1.26.1+ | Build and test |
| [Docker](https://docs.docker.com/get-docker/) | 20+ | Container builds and integration testing |
| [Docker Compose](https://docs.docker.com/compose/) | v2+ | Multi-service orchestration |
| [golangci-lint](https://golangci-lint.run/welcome/install-locally/) | latest | Static analysis and linting |
| [git](https://git-scm.com/) | 2.x+ | Version control |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/jarlex/vault-agent-operator.git
cd vault-agent-operator

# Build the binary
make build

# Run the full test suite
make test

# Run the linter
make lint
```

## Development Workflow

1. **Fork and branch** — create a feature branch from `main`:

   ```bash
   git checkout -b feature/my-change
   ```

2. **Write code** — follow the existing patterns in the codebase.

3. **Test** — run the test suite with the race detector:

   ```bash
   make test
   ```

4. **Lint** — ensure your code passes static analysis:

   ```bash
   make lint
   ```

5. **Commit** — write a clear, descriptive commit message.

6. **Open a PR** — see [PR Guidelines](#pr-guidelines) below.

## Testing

The project uses Go's standard `testing` package. Tests run across 8 packages
with the race detector enabled.

```bash
# Run all tests with race detector
go test ./... -v -race -count=1

# Run tests for a specific package
go test ./internal/agent/ -v -race

# Run tests with coverage report
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out

# Run a specific test by name
go test ./internal/api/ -run TestHealthHandler -v
```

### Testing Guidelines

- Write table-driven tests where possible.
- Use the race detector (`-race` flag) — it's enabled in `make test` by default.
- Add tests for new public functions and methods.
- Use `t.Helper()` in test helper functions.
- Use `t.Parallel()` where tests are independent.
- Mock external dependencies (LLM, MCP, Vault) — see existing test patterns
  in `internal/agent/` and `internal/mcp/`.

## Code Style

- **Formatting**: Use `gofmt` / `goimports` (enforced by `golangci-lint`).
- **Linting**: Run `make lint` which uses `golangci-lint run ./...`.
- **Vetting**: Run `go vet ./...` to catch common issues.
- **Godoc**: Add doc comments on all exported types, functions, and methods.
  Each package should have a `doc.go` with a package-level comment.
- **Naming**: Follow [Go naming conventions](https://go.dev/doc/effective_go#names).
- **Error handling**: Always check and wrap errors with context. Prefer
  `fmt.Errorf("doing X: %w", err)`.

## Docker

Build and test with Docker using the Makefile:

```bash
# Build the Docker image with version info
make docker-build

# Run the full MVP stack (Vault + agent)
export GITHUB_TOKEN=ghp_...
docker compose -f docker-compose.mvp.yaml up --build

# Run the automated smoke test against the running stack
./scripts/test-mvp.sh
```

The Dockerfile uses a multi-stage build: `golang` builder stage followed by a
`distroless` runtime image (<50 MB).

## PR Guidelines

Before submitting a pull request:

- [ ] All tests pass: `make test`
- [ ] Linter is clean: `make lint`
- [ ] New code has test coverage
- [ ] Commit messages are clear and descriptive
- [ ] Documentation is updated if behavior changes (README, godoc, etc.)
- [ ] No secrets, tokens, or credentials are committed

### PR Description

- Use a descriptive title summarizing the change.
- Explain **what** changed and **why**.
- Reference related issues if applicable.
- Include any testing steps for reviewers.

## Project Layout

```
internal/           # Private application packages
  agent/            # Hybrid reasoning loop
  api/              # HTTP handlers and middleware (chi v5)
  config/           # Viper-based configuration loading
  llm/              # LLM provider (go-openai)
  logging/          # Structured logging (zerolog)
  mcp/              # MCP client (vault-mcp-server communication)
  redaction/        # Secret redaction engine
  scheduler/        # Cron scheduler (robfig/cron v3)
cmd/                # Application entry points
config/             # Configuration files and prompt templates
scripts/            # Shell scripts (certs, health, seeding, testing)
```

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
