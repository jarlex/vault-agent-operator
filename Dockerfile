# =============================================================================
# vault-operator-agent — Multi-stage Docker build
# =============================================================================
# Stage 1 (builder):  Install Python dependencies
# Stage 2 (mcp-server): Extract vault-mcp-server binary
# Stage 3 (runtime):  Minimal image with deps + binary + source + config
#
# Target: < 200MB final image
# Optimisation strategy:
#   - python:3.12-slim base (~120MB)
#   - Multi-stage: only runtime deps copied (no build tools)
#   - No pip cache (--no-cache-dir)
#   - Minimal apt packages (only curl for healthcheck)
#   - .dockerignore excludes tests, caches, VCS, docs
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build — install Python dependencies into a prefix
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build essentials for any compiled dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy only dependency manifest first (layer caching)
COPY pyproject.toml ./

# Install dependencies into a separate prefix for clean copy
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: Extract vault-mcp-server binary from official image
# ---------------------------------------------------------------------------
FROM hashicorp/vault-mcp-server:latest AS mcp-server

# The binary is at /vault-mcp-server in the official image
# We just need this stage to COPY FROM it

# ---------------------------------------------------------------------------
# Stage 3: Runtime — minimal production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="vault-operator-agent contributors"
LABEL description="AI agent for HashiCorp Vault operations via natural language"

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --shell /bin/bash --create-home agent

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy vault-mcp-server binary from official image
COPY --from=mcp-server /vault-mcp-server /usr/local/bin/vault-mcp-server
RUN chmod +x /usr/local/bin/vault-mcp-server

# Copy application source and config
COPY src/ ./src/
COPY config/ ./config/

# Ensure config directory is readable
RUN chown -R agent:agent /app

# Switch to non-root user
USER agent

# Expose API port
EXPOSE 8000

# Health check — uses the healthcheck script or direct curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
