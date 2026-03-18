// Package agent implements the hybrid reasoning loop for vault-agent-operator.
package agent

// AgentResult is the output of a single agent execution.
type AgentResult struct {
	// Status is "completed" or "error".
	Status string `json:"status"`
	// Result is the agent's final textual response.
	Result string `json:"result"`
	// Data contains raw MCP tool results returned to the consumer.
	Data []map[string]any `json:"data,omitempty"`
	// ModelUsed is the LLM model that processed the request.
	ModelUsed string `json:"model_used"`
	// DurationMS is the total processing time in milliseconds.
	DurationMS int64 `json:"duration_ms"`
	// Error is a human-readable error message, if any.
	Error *string `json:"error,omitempty"`
	// ErrorCode categorizes the error for HTTP status mapping.
	ErrorCode string `json:"error_code,omitempty"`
	// Iterations is the number of reasoning loop iterations performed.
	Iterations int `json:"iterations"`
	// ToolCalls records each tool invocation for audit/logging.
	ToolCalls []ToolCallRecord `json:"tool_calls,omitempty"`
}

// ToolCallRecord captures the details of a single tool invocation.
type ToolCallRecord struct {
	// ToolName is the name of the MCP tool that was called.
	ToolName string `json:"tool_name"`
	// Arguments is the (redacted) arguments passed to the tool.
	Arguments map[string]any `json:"arguments"`
	// Result is the (redacted) tool result for audit logging.
	Result string `json:"result"`
	// IsError indicates whether the tool invocation failed.
	IsError bool `json:"is_error"`
	// DurationMS is the tool invocation time in milliseconds.
	DurationMS int64 `json:"duration_ms"`
}

// ExecuteRequest is the input to AgentCore.Execute.
type ExecuteRequest struct {
	// Prompt is the natural-language instruction from the consumer.
	Prompt string `json:"prompt"`
	// Model is an optional model alias override.
	Model string `json:"model,omitempty"`
	// SecretData contains structured secret key-value pairs for placeholder substitution.
	SecretData map[string]any `json:"secret_data,omitempty"`
}
