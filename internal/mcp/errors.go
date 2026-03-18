package mcp

import "fmt"

// MCPError is the base error type for MCP-related failures.
type MCPError struct {
	// Code categorizes the error (e.g., "connection", "timeout", "tool").
	Code string
	// Message is the human-readable error description.
	Message string
	// Err is the underlying error, if any.
	Err error
}

func (e *MCPError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("mcp %s: %s: %v", e.Code, e.Message, e.Err)
	}
	return fmt.Sprintf("mcp %s: %s", e.Code, e.Message)
}

// Unwrap returns the underlying error for errors.Is/errors.As support.
func (e *MCPError) Unwrap() error {
	return e.Err
}

// MCPConnectionError indicates a failure to connect to the MCP server.
type MCPConnectionError struct {
	MCPError
}

// As allows errors.As to match *MCPConnectionError as *MCPError.
func (e *MCPConnectionError) As(target interface{}) bool {
	if t, ok := target.(**MCPError); ok {
		*t = &e.MCPError
		return true
	}
	return false
}

// MCPTimeoutError indicates an MCP operation timed out.
type MCPTimeoutError struct {
	MCPError
}

// As allows errors.As to match *MCPTimeoutError as *MCPError.
func (e *MCPTimeoutError) As(target interface{}) bool {
	if t, ok := target.(**MCPError); ok {
		*t = &e.MCPError
		return true
	}
	return false
}

// MCPToolError indicates a tool invocation failed.
type MCPToolError struct {
	MCPError
	// ToolName is the name of the tool that failed.
	ToolName string
}

// As allows errors.As to match *MCPToolError as *MCPError.
func (e *MCPToolError) As(target interface{}) bool {
	if t, ok := target.(**MCPError); ok {
		*t = &e.MCPError
		return true
	}
	return false
}
