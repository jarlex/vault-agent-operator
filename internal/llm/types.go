// Package llm defines the LLM provider interface and related types.
package llm

import "context"

// LLMProvider abstracts LLM completions for the agent core.
type LLMProvider interface {
	// Complete sends a chat completion request and returns the response.
	// Uses ctx for timeout and cancellation.
	Complete(ctx context.Context, req CompletionRequest) (*LLMResponse, error)

	// AvailableModels returns the list of configured models.
	AvailableModels() []ModelInfo
}

// CompletionRequest holds the parameters for an LLM completion call.
type CompletionRequest struct {
	// Messages is the conversation history.
	Messages []Message
	// Tools are the available tool definitions. nil disables tool calling.
	Tools []Tool
	// Model is the model alias to use. Empty string means use the default.
	Model string
}

// Message represents a single message in the LLM conversation.
type Message struct {
	// Role is one of: "system", "user", "assistant", "tool".
	Role string `json:"role"`
	// Content is the text content. nil for tool-call-only assistant messages.
	Content *string `json:"content,omitempty"`
	// ToolCalls is present on assistant messages requesting tool invocations.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID is present on tool-result messages, referencing the original call.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ToolCall represents an LLM's request to invoke a tool.
type ToolCall struct {
	// ID uniquely identifies this tool call within the conversation.
	ID string `json:"id"`
	// Name is the tool function name.
	Name string `json:"name"`
	// Arguments are the parsed tool arguments.
	Arguments map[string]any `json:"arguments"`
}

// LLMResponse holds the parsed result from an LLM completion.
type LLMResponse struct {
	// Content is the text response, if any.
	Content *string
	// ToolCalls are tool invocation requests, if any.
	ToolCalls []ToolCall
	// Model is the model that actually processed the request.
	Model string
	// Usage contains token usage statistics, if available.
	Usage *TokenUsage
}

// TokenUsage records token consumption for a completion.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Tool defines a tool available for LLM function calling (OpenAI format).
type Tool struct {
	// Type is always "function".
	Type string `json:"type"`
	// Function contains the function definition.
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a callable tool function.
type ToolFunction struct {
	// Name is the function name.
	Name string `json:"name"`
	// Description is a human-readable description.
	Description string `json:"description"`
	// Parameters is the JSON Schema for the function parameters.
	Parameters map[string]any `json:"parameters"`
}

// ModelInfo describes an available LLM model.
type ModelInfo struct {
	// Name is the human-readable model alias.
	Name string `json:"name"`
	// Provider identifies the model provider (e.g., "github").
	Provider string `json:"provider"`
	// ModelID is the actual model identifier sent to the API.
	ModelID string `json:"model_id"`
	// SupportsToolCalling indicates whether this model supports function calling.
	SupportsToolCalling bool `json:"supports_tool_calling"`
}
