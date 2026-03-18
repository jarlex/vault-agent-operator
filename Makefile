BINARY     := vault-agent-operator
PKG        := ./cmd/vault-agent-operator/
VERSION    := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT     := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
LDFLAGS    := -s -w -X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.date=$(BUILD_DATE)

.PHONY: build test lint docker-build clean

build: ## Build static binary
	CGO_ENABLED=0 go build -ldflags="$(LDFLAGS)" -o $(BINARY) $(PKG)

test: ## Run tests with race detector
	go test ./... -v -race -count=1

lint: ## Run golangci-lint
	golangci-lint run ./...

docker-build: ## Build Docker image
	docker compose build --build-arg VERSION=$(VERSION) --build-arg COMMIT=$(COMMIT) --build-arg BUILD_DATE=$(BUILD_DATE)

clean: ## Remove build artifacts
	rm -f $(BINARY) coverage.out mcp_coverage.out
