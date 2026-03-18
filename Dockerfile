# =============================================================================
# vault-agent-operator — Multi-stage Docker build (Go)
# =============================================================================
# Stage 1 (builder):    Compile Go binary with static linking
# Stage 2 (mcp-server): Extract vault-mcp-server binary
# Stage 3 (runtime):    Minimal image with binary + config
#
# Target: < 50MB final image
# Optimisation strategy:
#   - Static Go binary (CGO_DISABLED=1) for scratch/distroless
#   - Multi-stage: only the binary + config copied to runtime
#   - distroless base (includes ca-certificates and tzdata)
#   - .dockerignore excludes tests, caches, VCS, docs
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build — compile Go binary
# ---------------------------------------------------------------------------
FROM golang:1.26-alpine AS builder

WORKDIR /build

# Install git (needed for go mod download with VCS)
RUN apk add --no-cache git

# Cache dependencies — copy go.mod/go.sum first
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY cmd/ ./cmd/
COPY internal/ ./internal/

# Build static binary with version info
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w -X main.version=${VERSION} -X main.commit=${COMMIT} -X main.date=${BUILD_DATE}" \
    -o /build/vault-agent-operator \
    ./cmd/vault-agent-operator/

# ---------------------------------------------------------------------------
# Stage 2: Extract vault-mcp-server binary from official image
# ---------------------------------------------------------------------------
FROM hashicorp/vault-mcp-server:latest AS mcp-server

# The binary is at /bin/vault-mcp-server in the official release image

# ---------------------------------------------------------------------------
# Stage 3: Runtime — minimal production image
# ---------------------------------------------------------------------------
FROM gcr.io/distroless/static-debian12:nonroot AS runtime

LABEL maintainer="vault-operator-agent contributors"
LABEL description="AI agent for HashiCorp Vault operations via natural language"

WORKDIR /app

# Copy Go binary
COPY --from=builder /build/vault-agent-operator /app/vault-agent-operator

# Copy vault-mcp-server binary from official image
COPY --from=mcp-server /bin/vault-mcp-server /usr/local/bin/vault-mcp-server

# Copy configuration files and prompts
COPY config/ ./config/

# Expose API port
EXPOSE 8000

# Health check — uses the Go binary's /api/v1/health endpoint
# Note: distroless has no shell or curl; use docker-compose healthcheck
# or an orchestrator (k8s) liveness probe instead.

# Run as non-root user (UID 1000 per spec)
USER 1000:1000

# Default command
ENTRYPOINT ["/app/vault-agent-operator"]
