package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	mcpclient "github.com/mark3labs/mcp-go/client"
	mcptypes "github.com/mark3labs/mcp-go/mcp"
	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/llm"
)

// StdioMCPClient implements the MCPClient interface using mcp-go's
// stdio transport to communicate with vault-mcp-server.
type StdioMCPClient struct {
	mu        sync.RWMutex
	transport *StdioTransport
	client    *mcpclient.Client
	tools     []MCPTool
	connected bool
	logger    zerolog.Logger
	cfg       TransportConfig
}

// NewStdioMCPClient creates a new MCP client configured for stdio transport.
func NewStdioMCPClient(cfg TransportConfig, logger zerolog.Logger) *StdioMCPClient {
	return &StdioMCPClient{
		transport: NewStdioTransport(cfg, logger),
		logger:    logger.With().Str("component", "mcp.client").Logger(),
		cfg:       cfg,
	}
}

// Connect establishes the MCP connection, performs initialization, and
// discovers available tools from the server.
func (c *StdioMCPClient) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected && c.client != nil {
		return nil
	}

	c.logger.Info().Msg("mcp.client.connecting")

	client, err := c.transport.Start(ctx)
	if err != nil {
		return err
	}
	c.client = client

	// Discover tools.
	tools, err := c.discoverTools(ctx)
	if err != nil {
		c.logger.Warn().Err(err).Msg("mcp.client.tool_discovery_failed")
		// Connection is up even if tool discovery fails — tools may appear later.
	} else {
		c.tools = tools
		c.logger.Info().
			Int("tool_count", len(tools)).
			Msg("mcp.client.tools_discovered")
	}

	c.connected = true
	c.logger.Info().Msg("mcp.client.connected")
	return nil
}

// Disconnect gracefully closes the MCP connection and terminates the subprocess.
func (c *StdioMCPClient) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.connected {
		return nil
	}

	c.logger.Info().Msg("mcp.client.disconnecting")

	err := c.transport.Stop()
	c.client = nil
	c.tools = nil
	c.connected = false

	c.logger.Info().Msg("mcp.client.disconnected")
	return err
}

// Tools returns the discovered MCP tools.
func (c *StdioMCPClient) Tools() []MCPTool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.tools == nil {
		return []MCPTool{}
	}

	result := make([]MCPTool, len(c.tools))
	copy(result, c.tools)
	return result
}

// ToolsAsOpenAIFormat converts the discovered MCP tools to OpenAI
// function-calling format for use with the LLM provider.
func (c *StdioMCPClient) ToolsAsOpenAIFormat() []llm.Tool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.tools) == 0 {
		return []llm.Tool{}
	}

	result := make([]llm.Tool, 0, len(c.tools))
	for _, tool := range c.tools {
		result = append(result, llm.MCPToolToOpenAI(
			tool.Name,
			tool.Description,
			tool.InputSchema,
		))
	}
	return result
}

// CallTool invokes an MCP tool by name with the given arguments.
// It respects context cancellation and applies the configured tool timeout.
func (c *StdioMCPClient) CallTool(ctx context.Context, name string, arguments map[string]any) (*ToolResult, error) {
	c.mu.RLock()
	client := c.client
	connected := c.connected
	c.mu.RUnlock()

	if !connected || client == nil {
		return nil, &MCPConnectionError{
			MCPError: MCPError{
				Code:    "connection",
				Message: "not connected to MCP server",
			},
		}
	}

	c.logger.Info().
		Str("tool_name", name).
		Msg("mcp.client.call_tool")

	c.logger.Debug().
		Str("tool_name", name).
		Interface("arguments", arguments).
		Msg("mcp.client.call_tool.args")

	// Apply per-tool timeout.
	toolTimeout := time.Duration(c.cfg.ToolTimeout) * time.Second
	if toolTimeout <= 0 {
		toolTimeout = 30 * time.Second
	}
	callCtx, cancel := context.WithTimeout(ctx, toolTimeout)
	defer cancel()

	// Build the MCP CallToolRequest.
	req := mcptypes.CallToolRequest{}
	req.Params.Name = name
	req.Params.Arguments = arguments

	result, err := client.CallTool(callCtx, req)
	if err != nil {
		// Check for timeout.
		if callCtx.Err() == context.DeadlineExceeded {
			return nil, &MCPTimeoutError{
				MCPError: MCPError{
					Code:    "timeout",
					Message: fmt.Sprintf("tool %q timed out after %s", name, toolTimeout),
					Err:     err,
				},
			}
		}
		return nil, &MCPToolError{
			MCPError: MCPError{
				Code:    "tool",
				Message: fmt.Sprintf("error calling tool %q", name),
				Err:     err,
			},
			ToolName: name,
		}
	}

	// Extract text content from the result.
	content := extractTextContent(result)

	c.logger.Debug().
		Str("tool_name", name).
		Bool("is_error", result.IsError).
		Int("content_length", len(content)).
		Msg("mcp.client.call_tool.result")

	return &ToolResult{
		Content: content,
		IsError: result.IsError,
	}, nil
}

// HealthCheck pings the MCP server to verify it's responsive.
func (c *StdioMCPClient) HealthCheck(ctx context.Context) error {
	c.mu.RLock()
	client := c.client
	connected := c.connected
	c.mu.RUnlock()

	if !connected || client == nil {
		return &MCPConnectionError{
			MCPError: MCPError{
				Code:    "connection",
				Message: "not connected to MCP server",
			},
		}
	}

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := client.Ping(pingCtx); err != nil {
		return &MCPError{
			Code:    "health",
			Message: "MCP server health check failed",
			Err:     err,
		}
	}

	return nil
}

// IsConnected returns true if the client has an active connection.
func (c *StdioMCPClient) IsConnected() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.connected && c.client != nil
}

// ConnectWithRetry attempts to connect with exponential backoff.
// This is used during startup and for reconnection after failures.
func (c *StdioMCPClient) ConnectWithRetry(ctx context.Context) error {
	initialDelay := c.cfg.ReconnectInitialDelay
	if initialDelay <= 0 {
		initialDelay = 1.0
	}
	maxDelay := c.cfg.ReconnectMaxDelay
	if maxDelay <= 0 {
		maxDelay = 60.0
	}

	delay := initialDelay
	for attempt := 1; ; attempt++ {
		err := c.Connect(ctx)
		if err == nil {
			return nil
		}

		c.logger.Warn().
			Err(err).
			Int("attempt", attempt).
			Float64("next_delay_secs", delay).
			Msg("mcp.client.connect_retry")

		select {
		case <-ctx.Done():
			return fmt.Errorf("connection cancelled after %d attempts: %w", attempt, ctx.Err())
		case <-time.After(time.Duration(delay * float64(time.Second))):
		}

		// Exponential backoff.
		delay = math.Min(delay*2, maxDelay)
	}
}

// discoverTools queries the MCP server for available tools and converts
// them to our internal MCPTool type.
func (c *StdioMCPClient) discoverTools(ctx context.Context) ([]MCPTool, error) {
	listCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	result, err := c.client.ListTools(listCtx, mcptypes.ListToolsRequest{})
	if err != nil {
		return nil, fmt.Errorf("listing tools: %w", err)
	}

	tools := make([]MCPTool, 0, len(result.Tools))
	for _, t := range result.Tools {
		schema := convertInputSchema(t.InputSchema)
		tools = append(tools, MCPTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: schema,
		})
	}

	return tools, nil
}

// convertInputSchema converts mcp-go's ToolInputSchema to a map[string]any
// for our internal representation and OpenAI compatibility.
func convertInputSchema(schema mcptypes.ToolInputSchema) map[string]any {
	// Marshal the schema to JSON and unmarshal back to get a clean map.
	data, err := json.Marshal(schema)
	if err != nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}

	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}

	// Ensure required fields for OpenAI compatibility.
	if _, ok := result["type"]; !ok {
		result["type"] = "object"
	}
	if _, ok := result["properties"]; !ok {
		result["properties"] = map[string]any{}
	}

	return result
}

// extractTextContent extracts text content from an MCP CallToolResult.
// It concatenates all TextContent items with newlines.
func extractTextContent(result *mcptypes.CallToolResult) string {
	if result == nil || len(result.Content) == 0 {
		return ""
	}

	var parts []string
	for _, c := range result.Content {
		if tc, ok := mcptypes.AsTextContent(c); ok {
			parts = append(parts, tc.Text)
		}
	}

	if len(parts) == 0 {
		// Fallback: try to marshal all content as JSON.
		data, err := json.Marshal(result.Content)
		if err == nil {
			return string(data)
		}
		return ""
	}

	if len(parts) == 1 {
		return parts[0]
	}

	result2 := parts[0]
	for i := 1; i < len(parts); i++ {
		result2 += "\n" + parts[i]
	}
	return result2
}
