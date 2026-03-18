// Package api defines the HTTP API layer for vault-agent-operator.
package api

// TaskRequest is the JSON body for POST /api/v1/tasks.
type TaskRequest struct {
	// Prompt is the natural-language instruction (required, 1-4096 chars).
	Prompt string `json:"prompt"`
	// Model is an optional model alias override.
	Model *string `json:"model,omitempty"`
	// MaxIterations overrides the default max reasoning-loop iterations.
	MaxIterations *int `json:"max_iterations,omitempty"`
	// SecretData contains structured secret key-value pairs for write operations.
	SecretData map[string]any `json:"secret_data,omitempty"`
}

// TaskResponse is the JSON response for POST /api/v1/tasks.
type TaskResponse struct {
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
}

// HealthResponse is the JSON response for GET /api/v1/health.
type HealthResponse struct {
	// Status is "healthy", "degraded", or "unhealthy".
	Status string `json:"status"`
	// Agent describes the agent's state.
	Agent string `json:"agent"`
	// VaultMCP describes the vault-mcp-server connection state.
	VaultMCP string `json:"vault_mcp"`
	// VaultServer describes the Vault server connectivity.
	VaultServer string `json:"vault_server"`
	// UptimeSeconds is seconds since the server started.
	UptimeSeconds float64 `json:"uptime_seconds"`
	// Version is the build version of the binary.
	Version string `json:"version"`
}

// ModelsResponse is the JSON response for GET /api/v1/models.
type ModelsResponse struct {
	// DefaultModel is the name of the default model.
	DefaultModel string `json:"default_model"`
	// AvailableModels lists all configured models.
	AvailableModels []ModelDetail `json:"available_models"`
}

// ModelDetail describes a single model in the models endpoint response.
type ModelDetail struct {
	// Name is the human-readable model alias.
	Name string `json:"name"`
	// Provider identifies the model provider.
	Provider string `json:"provider"`
	// ModelID is the actual model identifier sent to the API.
	ModelID string `json:"model_id"`
	// SupportsToolCalling indicates whether this model supports function calling.
	SupportsToolCalling bool `json:"supports_tool_calling"`
	// IsDefault is true if this is the default model.
	IsDefault bool `json:"is_default"`
}

// ErrorResponse is a structured error returned by the API.
type ErrorResponse struct {
	// Error is the error message.
	Error string `json:"error"`
	// Detail provides additional error context, if any.
	Detail *string `json:"detail,omitempty"`
	// RequestID is the request's correlation ID.
	RequestID *string `json:"request_id,omitempty"`
}
