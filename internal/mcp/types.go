// Package mcp defines the MCP client interface and related types.
package mcp

import (
	"context"

	"github.com/jarlex/vault-agent-operator/internal/llm"
)

// MCPClient abstracts the connection to a Model Context Protocol server.
type MCPClient interface {
	// Connect establishes a connection and discovers available tools.
	Connect(ctx context.Context) error

	// Disconnect cleanly closes the connection and terminates any subprocesses.
	Disconnect() error

	// Tools returns the discovered MCP tools.
	Tools() []MCPTool

	// ToolsAsOpenAIFormat converts MCP tools to OpenAI function-calling format.
	ToolsAsOpenAIFormat() []llm.Tool

	// CallTool invokes an MCP tool by name with the given arguments.
	CallTool(ctx context.Context, name string, arguments map[string]any) (*ToolResult, error)

	// HealthCheck pings the MCP server. Returns nil if healthy.
	HealthCheck(ctx context.Context) error

	// IsConnected returns true if the client has an active connection.
	IsConnected() bool
}

// MCPTool describes a tool discovered from the MCP server.
type MCPTool struct {
	// Name is the tool name as registered in the MCP server.
	Name string `json:"name"`
	// Description is a human-readable description of the tool.
	Description string `json:"description"`
	// InputSchema is the JSON Schema for the tool's input parameters.
	InputSchema map[string]any `json:"input_schema"`
}

// ToolResult holds the output of an MCP tool invocation.
type ToolResult struct {
	// Content is the textual result from the tool.
	Content string
	// IsError indicates whether the tool invocation failed.
	IsError bool
}
