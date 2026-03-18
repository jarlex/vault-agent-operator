package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// OpenAIProvider implements LLMProvider using the go-openai library.
// It supports any OpenAI-compatible API endpoint (OpenAI, Azure OpenAI,
// GitHub Models) via configurable BaseURL.
type OpenAIProvider struct {
	client       *openai.Client
	models       []ModelInfo
	defaultModel string
	maxRetries   int
	timeout      time.Duration
}

// OpenAIProviderConfig holds configuration for creating an OpenAIProvider.
type OpenAIProviderConfig struct {
	// APIKey is the authentication token (e.g., GitHub PAT).
	APIKey string
	// BaseURL is the API endpoint. Defaults to OpenAI's API.
	BaseURL string
	// DefaultModel is the model alias to use when none is specified.
	DefaultModel string
	// Models is the list of available models.
	Models []ModelInfo
	// MaxRetries is the maximum number of retry attempts for transient errors.
	MaxRetries int
	// RequestTimeout is the per-request timeout in seconds.
	RequestTimeout int
}

// NewOpenAIProvider creates a new OpenAI-compatible LLM provider.
func NewOpenAIProvider(cfg OpenAIProviderConfig) (*OpenAIProvider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	config := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		config.BaseURL = cfg.BaseURL
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 3
	}

	timeout := time.Duration(cfg.RequestTimeout) * time.Second
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	return &OpenAIProvider{
		client:       openai.NewClientWithConfig(config),
		models:       cfg.Models,
		defaultModel: cfg.DefaultModel,
		maxRetries:   maxRetries,
		timeout:      timeout,
	}, nil
}

// Complete sends a chat completion request to the OpenAI-compatible API.
// It handles model resolution, request conversion, retry logic, and
// response parsing.
func (p *OpenAIProvider) Complete(ctx context.Context, req CompletionRequest) (*LLMResponse, error) {
	modelID, err := p.resolveModel(req.Model)
	if err != nil {
		return nil, err
	}

	// Check tool-calling support if tools are requested.
	if len(req.Tools) > 0 {
		if !p.modelSupportsTools(req.Model) {
			return nil, &LLMToolCallUnsupportedError{
				LLMError: LLMError{
					Code:    "tool_call_unsupported",
					Message: fmt.Sprintf("model %q does not support tool calling", req.Model),
				},
				Model: req.Model,
			}
		}
	}

	// Convert internal types to go-openai types.
	messages := convertMessages(req.Messages)
	tools := convertTools(req.Tools)

	openaiReq := openai.ChatCompletionRequest{
		Model:    modelID,
		Messages: messages,
	}
	if len(tools) > 0 {
		openaiReq.Tools = tools
	}

	// Execute with retry logic.
	resp, err := p.executeWithRetry(ctx, openaiReq)
	if err != nil {
		return nil, err
	}

	return p.parseResponse(resp, modelID)
}

// AvailableModels returns the list of configured models.
func (p *OpenAIProvider) AvailableModels() []ModelInfo {
	return p.models
}

// resolveModel maps a model alias to the actual model ID.
// If model is empty, it uses the default model.
func (p *OpenAIProvider) resolveModel(model string) (string, error) {
	if model == "" {
		model = p.defaultModel
	}

	for _, m := range p.models {
		if m.Name == model {
			return m.ModelID, nil
		}
	}

	// If not found by alias, try using the model string directly as a model ID.
	for _, m := range p.models {
		if m.ModelID == model {
			return m.ModelID, nil
		}
	}

	// Fall back to using the string directly — the API will validate it.
	if model != "" {
		return model, nil
	}

	return "", &LLMError{
		Code:    "model_not_found",
		Message: fmt.Sprintf("model %q not found in configured models", model),
	}
}

// modelSupportsTools checks if the given model alias supports tool calling.
func (p *OpenAIProvider) modelSupportsTools(model string) bool {
	if model == "" {
		model = p.defaultModel
	}

	for _, m := range p.models {
		if m.Name == model || m.ModelID == model {
			return m.SupportsToolCalling
		}
	}

	// If model not found in config, assume it supports tools.
	return true
}

// executeWithRetry runs the completion request with exponential backoff
// retry logic for transient errors (rate limits, server errors).
func (p *OpenAIProvider) executeWithRetry(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	var lastErr error

	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		if attempt > 0 {
			delay := p.backoffDelay(attempt)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		// Create a per-request timeout context.
		reqCtx, cancel := context.WithTimeout(ctx, p.timeout)
		resp, err := p.client.CreateChatCompletion(reqCtx, req)
		cancel()

		if err == nil {
			return &resp, nil
		}

		lastErr = err

		// Classify the error to determine if we should retry.
		classified := classifyOpenAIError(err)
		if !isRetryable(classified) {
			return nil, classified
		}
	}

	// All retries exhausted.
	return nil, classifyOpenAIError(lastErr)
}

// backoffDelay calculates the exponential backoff delay for a retry attempt.
// Delay = min(2^(attempt-1), 30) seconds.
func (p *OpenAIProvider) backoffDelay(attempt int) time.Duration {
	seconds := math.Pow(2, float64(attempt-1))
	if seconds > 30 {
		seconds = 30
	}
	return time.Duration(seconds * float64(time.Second))
}

// parseResponse converts a go-openai response to our internal LLMResponse type.
func (p *OpenAIProvider) parseResponse(resp *openai.ChatCompletionResponse, modelID string) (*LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return &LLMResponse{
			Model: modelID,
			Usage: convertUsage(resp.Usage),
		}, nil
	}

	choice := resp.Choices[0]
	result := &LLMResponse{
		Model: modelID,
		Usage: convertUsage(resp.Usage),
	}

	// Parse text content.
	if choice.Message.Content != "" {
		content := choice.Message.Content
		result.Content = &content
	}

	// Parse tool calls.
	if len(choice.Message.ToolCalls) > 0 {
		toolCalls := make([]ToolCall, 0, len(choice.Message.ToolCalls))
		for _, tc := range choice.Message.ToolCalls {
			args := make(map[string]any)
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					// If we can't parse arguments, store as raw string.
					args = map[string]any{"_raw": tc.Function.Arguments}
				}
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: args,
			})
		}
		result.ToolCalls = toolCalls
	}

	return result, nil
}

// convertMessages transforms internal Message types to go-openai format.
func convertMessages(messages []Message) []openai.ChatCompletionMessage {
	result := make([]openai.ChatCompletionMessage, 0, len(messages))
	for _, msg := range messages {
		oaiMsg := openai.ChatCompletionMessage{
			Role: msg.Role,
		}

		if msg.Content != nil {
			oaiMsg.Content = *msg.Content
		}

		if msg.ToolCallID != "" {
			oaiMsg.ToolCallID = msg.ToolCallID
		}

		if len(msg.ToolCalls) > 0 {
			oaiToolCalls := make([]openai.ToolCall, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				argsJSON, _ := json.Marshal(tc.Arguments)
				oaiToolCalls = append(oaiToolCalls, openai.ToolCall{
					ID:   tc.ID,
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      tc.Name,
						Arguments: string(argsJSON),
					},
				})
			}
			oaiMsg.ToolCalls = oaiToolCalls
		}

		result = append(result, oaiMsg)
	}
	return result
}

// convertTools transforms internal Tool definitions to go-openai format.
func convertTools(tools []Tool) []openai.Tool {
	if len(tools) == 0 {
		return nil
	}

	result := make([]openai.Tool, 0, len(tools))
	for _, tool := range tools {
		params := tool.Function.Parameters
		if params == nil {
			params = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}

		// Ensure "properties" key exists (same normalization as Python version).
		if _, ok := params["properties"]; !ok {
			params["properties"] = map[string]any{}
		}

		paramsJSON, _ := json.Marshal(params)

		result = append(result, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  json.RawMessage(paramsJSON),
			},
		})
	}
	return result
}

// convertUsage transforms go-openai usage to our internal TokenUsage type.
func convertUsage(usage openai.Usage) *TokenUsage {
	return &TokenUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
	}
}

// classifyOpenAIError inspects a go-openai error and returns a typed LLM error.
func classifyOpenAIError(err error) error {
	if err == nil {
		return nil
	}

	// Check for context errors.
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return &LLMError{
			Code:    "timeout",
			Message: "request timed out or was cancelled",
			Err:     err,
		}
	}

	// Check for go-openai APIError type.
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return ClassifyError(err, apiErr.HTTPStatusCode)
	}

	// Check for go-openai RequestError type.
	var reqErr *openai.RequestError
	if errors.As(err, &reqErr) {
		return ClassifyError(err, reqErr.HTTPStatusCode)
	}

	// Check error message for common patterns.
	msg := err.Error()
	if strings.Contains(msg, "401") || strings.Contains(msg, "Unauthorized") {
		return ClassifyError(err, http.StatusUnauthorized)
	}
	if strings.Contains(msg, "429") || strings.Contains(msg, "rate limit") {
		return ClassifyError(err, http.StatusTooManyRequests)
	}
	if strings.Contains(msg, "500") || strings.Contains(msg, "502") ||
		strings.Contains(msg, "503") || strings.Contains(msg, "504") {
		return ClassifyError(err, http.StatusInternalServerError)
	}

	// Unknown error.
	return &LLMError{
		Code:    "unknown",
		Message: fmt.Sprintf("unexpected LLM error: %v", err),
		Err:     err,
	}
}

// isRetryable determines if a classified error should trigger a retry.
// Auth errors are NOT retried; rate limit and server errors ARE.
func isRetryable(err error) bool {
	if err == nil {
		return false
	}

	var authErr *LLMAuthError
	if errors.As(err, &authErr) {
		return false
	}

	var toolErr *LLMToolCallUnsupportedError
	if errors.As(err, &toolErr) {
		return false
	}

	var rateLimitErr *LLMRateLimitError
	if errors.As(err, &rateLimitErr) {
		return true
	}

	var serviceErr *LLMServiceError
	if errors.As(err, &serviceErr) {
		return true
	}

	// Context errors are not retryable.
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return false
	}

	// Default: retry unknown errors.
	var llmErr *LLMError
	if errors.As(err, &llmErr) {
		return llmErr.Code == "unknown"
	}

	return false
}
