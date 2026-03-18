package mcp

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	mcpclient "github.com/mark3labs/mcp-go/client"
	mcptypes "github.com/mark3labs/mcp-go/mcp"
	"github.com/rs/zerolog"
)

// TransportConfig holds configuration for creating an MCP transport.
type TransportConfig struct {
	// Transport is the transport type: "stdio" or "sse".
	Transport string
	// ServerBinary is the path to the vault-mcp-server binary (stdio mode).
	ServerBinary string
	// ServerURL is the HTTP URL for the MCP server (sse mode).
	ServerURL string
	// VaultAddr is the VAULT_ADDR to pass to the subprocess.
	VaultAddr string
	// VaultToken is the VAULT_TOKEN to pass to the subprocess.
	VaultToken string
	// ToolTimeout is the per-tool invocation timeout in seconds.
	ToolTimeout int
	// ReconnectInitialDelay is the initial reconnection delay in seconds.
	ReconnectInitialDelay float64
	// ReconnectMaxDelay is the maximum reconnection delay in seconds.
	ReconnectMaxDelay float64
}

// StdioTransport manages an mcp-go stdio client connected to a
// vault-mcp-server subprocess. It handles lifecycle (start/stop),
// environment variable passthrough, and initialization.
type StdioTransport struct {
	mu     sync.Mutex
	cfg    TransportConfig
	client *mcpclient.Client
	logger zerolog.Logger
	closed bool
}

// NewStdioTransport creates a new StdioTransport with the given config.
func NewStdioTransport(cfg TransportConfig, logger zerolog.Logger) *StdioTransport {
	return &StdioTransport{
		cfg:    cfg,
		logger: logger.With().Str("component", "mcp.transport").Logger(),
	}
}

// Start spawns the vault-mcp-server subprocess and establishes the MCP connection.
// It performs the MCP initialization handshake and returns the connected client.
func (t *StdioTransport) Start(ctx context.Context) (*mcpclient.Client, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.client != nil {
		return t.client, nil
	}

	t.logger.Info().
		Str("binary", t.cfg.ServerBinary).
		Str("vault_addr", t.cfg.VaultAddr).
		Msg("mcp.transport.starting")

	// Build environment variables for the subprocess.
	env := t.buildEnv()

	// Create the stdio client — this spawns the subprocess.
	client, err := mcpclient.NewStdioMCPClient(t.cfg.ServerBinary, env)
	if err != nil {
		return nil, &MCPConnectionError{
			MCPError: MCPError{
				Code:    "connection",
				Message: fmt.Sprintf("failed to start vault-mcp-server subprocess: %s", t.cfg.ServerBinary),
				Err:     err,
			},
		}
	}

	// Perform the MCP initialization handshake.
	initReq := mcptypes.InitializeRequest{}
	initReq.Params.ClientInfo = mcptypes.Implementation{
		Name:    "vault-agent-operator",
		Version: "1.0.0",
	}
	initReq.Params.ProtocolVersion = mcptypes.LATEST_PROTOCOL_VERSION

	initCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	_, err = client.Initialize(initCtx, initReq)
	if err != nil {
		// Close the client if initialization fails.
		_ = client.Close()
		return nil, &MCPConnectionError{
			MCPError: MCPError{
				Code:    "connection",
				Message: "MCP initialization handshake failed",
				Err:     err,
			},
		}
	}

	t.client = client
	t.closed = false

	t.logger.Info().Msg("mcp.transport.started")
	return client, nil
}

// Stop gracefully shuts down the subprocess and closes the MCP connection.
func (t *StdioTransport) Stop() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.client == nil || t.closed {
		return nil
	}

	t.logger.Info().Msg("mcp.transport.stopping")

	err := t.client.Close()
	t.client = nil
	t.closed = true

	if err != nil {
		t.logger.Warn().Err(err).Msg("mcp.transport.close_error")
		return &MCPError{
			Code:    "close",
			Message: "error closing MCP transport",
			Err:     err,
		}
	}

	t.logger.Info().Msg("mcp.transport.stopped")
	return nil
}

// Client returns the underlying mcp-go client, or nil if not started.
func (t *StdioTransport) Client() *mcpclient.Client {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.client
}

// IsRunning returns true if the transport has an active client.
func (t *StdioTransport) IsRunning() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.client != nil && !t.closed
}

// buildEnv constructs the environment variables for the subprocess.
// It passes through VAULT_ADDR, VAULT_TOKEN, and inherits the rest
// from the current process.
func (t *StdioTransport) buildEnv() []string {
	// Start from current process environment.
	env := os.Environ()

	// Override/add vault-specific variables.
	env = setEnvVar(env, "VAULT_ADDR", t.cfg.VaultAddr)
	if t.cfg.VaultToken != "" {
		env = setEnvVar(env, "VAULT_TOKEN", t.cfg.VaultToken)
	}

	return env
}

// setEnvVar sets or replaces an environment variable in a string slice.
func setEnvVar(env []string, key, value string) []string {
	prefix := key + "="
	for i, e := range env {
		if len(e) >= len(prefix) && e[:len(prefix)] == prefix {
			env[i] = prefix + value
			return env
		}
	}
	return append(env, prefix+value)
}
