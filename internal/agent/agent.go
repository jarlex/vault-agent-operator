package agent

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/config"
	"github.com/jarlex/vault-agent-operator/internal/llm"
	"github.com/jarlex/vault-agent-operator/internal/logging"
	"github.com/jarlex/vault-agent-operator/internal/mcp"
	"github.com/jarlex/vault-agent-operator/internal/redaction"
)

// AgentCore is the hybrid reasoning engine. It orchestrates the LLM,
// MCP tools, and secret redaction to execute user requests.
type AgentCore struct {
	llm       llm.LLMProvider
	mcp       mcp.MCPClient
	config    config.AgentConfig
	vaultAddr string
	sanitizer redaction.PromptSanitizer
	redactor  redaction.SecretRedactor
	logger    zerolog.Logger
}

// NewAgentCore constructs a new AgentCore with all its dependencies.
func NewAgentCore(
	llmProvider llm.LLMProvider,
	mcpClient mcp.MCPClient,
	cfg config.AgentConfig,
	vaultAddr string,
	logger zerolog.Logger,
) *AgentCore {
	return &AgentCore{
		llm:       llmProvider,
		mcp:       mcpClient,
		config:    cfg,
		vaultAddr: vaultAddr,
		sanitizer: redaction.NewPromptSanitizer(),
		redactor:  redaction.NewSecretRedactor(),
		logger:    logger.With().Str("component", "agent").Logger(),
	}
}

// Execute runs the agent reasoning loop for a single request.
// This is the main entry point — it creates a fresh SecretContext,
// sanitizes the prompt, loads tools, runs the reasoning loop,
// and returns the result.
func (a *AgentCore) Execute(ctx context.Context, req ExecuteRequest) (*AgentResult, error) {
	startTime := time.Now()
	logger := a.logger
	if l := logging.FromContext(ctx); l.GetLevel() != zerolog.Disabled {
		logger = l.With().Str("component", "agent").Logger()
	}

	logger.Info().
		Int("prompt_length", len(req.Prompt)).
		Str("model", req.Model).
		Bool("has_secret_data", len(req.SecretData) > 0).
		Msg("agent.execute.start")

	// 1. Create a fresh SecretContext for this request (stateless per-request).
	secretCtx := redaction.NewSecretContext()
	defer secretCtx.Destroy()

	// 2. Sanitize the prompt — replace detected secrets with placeholders.
	sanitizedPrompt := a.sanitizer.SanitizePrompt(req.Prompt, req.SecretData, secretCtx)

	logger.Debug().
		Int("original_length", len(req.Prompt)).
		Int("sanitized_length", len(sanitizedPrompt)).
		Int("placeholder_count", secretCtx.PlaceholderCount()).
		Msg("agent.execute.prompt_sanitized")

	// 3. Get tools from MCP and build the tool list for the LLM.
	tools := a.mcp.ToolsAsOpenAIFormat()
	toolNames := make([]string, 0, len(tools))
	for _, t := range tools {
		toolNames = append(toolNames, t.Function.Name)
	}

	// 4. Load and build the system prompt with tool names.
	systemPrompt := LoadSystemPrompt(
		a.config.SystemPromptPath,
		a.vaultAddr,
		toolNames,
		logger,
	)

	// 5. Build initial messages.
	messages := BuildInitialMessages(systemPrompt, sanitizedPrompt)

	// 6. Resolve max iterations.
	maxIterations := a.config.MaxIterations
	if maxIterations <= 0 {
		maxIterations = 10
	}

	// 7. Run the reasoning loop.
	result := a.runLoop(ctx, messages, tools, req.Model, maxIterations, secretCtx, logger)

	// 8. If the result is text-only (no tool calls) and we have placeholders,
	//    restore any placeholder tokens that the LLM may have echoed.
	if result.Result != "" && secretCtx.HasPlaceholders() {
		result.Result = secretCtx.ResolveAllPlaceholders(result.Result)
	}

	// 9. Set timing.
	result.DurationMS = time.Since(startTime).Milliseconds()
	result.ModelUsed = req.Model
	if result.ModelUsed == "" {
		result.ModelUsed = "default"
	}

	logger.Info().
		Str("status", result.Status).
		Int("iterations", result.Iterations).
		Int("tool_calls", len(result.ToolCalls)).
		Int64("duration_ms", result.DurationMS).
		Msg("agent.execute.complete")

	return result, nil
}

// runLoop is the hybrid reasoning loop — the heart of the agent.
// See loop.go for the full implementation.
func (a *AgentCore) runLoop(
	ctx context.Context,
	messages []llm.Message,
	tools []llm.Tool,
	model string,
	maxIterations int,
	secretCtx *redaction.SecretContext,
	logger zerolog.Logger,
) *AgentResult {
	var allToolCalls []ToolCallRecord

	for iteration := 1; iteration <= maxIterations; iteration++ {
		logger.Debug().
			Int("iteration", iteration).
			Int("max_iterations", maxIterations).
			Int("message_count", len(messages)).
			Msg("agent.loop.iteration_start")

		// 1. Call LLM with messages + tools.
		llmResp, err := a.llm.Complete(ctx, llm.CompletionRequest{
			Messages: messages,
			Tools:    tools,
			Model:    model,
		})
		if err != nil {
			// Check for specific LLM error types.
			errCode := classifyLLMError(err)
			errMsg := err.Error()
			return &AgentResult{
				Status:     "error",
				Error:      &errMsg,
				ErrorCode:  errCode,
				Iterations: iteration,
				ToolCalls:  allToolCalls,
			}
		}

		// 2. Check for tool calls.
		if len(llmResp.ToolCalls) > 0 {
			logger.Info().
				Int("tool_call_count", len(llmResp.ToolCalls)).
				Int("iteration", iteration).
				Msg("agent.loop.tool_calls")

			// Append assistant message with tool calls to conversation.
			assistantMsg := llm.Message{
				Role:      "assistant",
				Content:   llmResp.Content,
				ToolCalls: llmResp.ToolCalls,
			}
			messages = append(messages, assistantMsg)

			// Dispatch all tool calls.
			records, rawResults, hasErrors := a.dispatchToolCalls(
				ctx, llmResp.ToolCalls, &messages, secretCtx, logger,
			)
			allToolCalls = append(allToolCalls, records...)

			if !hasErrors {
				// FAST PATH: all tools succeeded — return raw results directly.
				// No second LLM call needed.
				logger.Info().
					Int("iteration", iteration).
					Int("results", len(rawResults)).
					Msg("agent.loop.fast_path")

				return &AgentResult{
					Status:     "completed",
					Data:       rawResults,
					Iterations: iteration,
					ToolCalls:  allToolCalls,
				}
			}

			// ERROR RETRY: continue loop so LLM sees the errors and can retry.
			logger.Info().
				Int("iteration", iteration).
				Msg("agent.loop.error_retry")
			continue
		}

		// 3. LLM responded with text (no tool calls).
		if llmResp.Content != nil && *llmResp.Content != "" {
			logger.Info().
				Int("iteration", iteration).
				Int("content_length", len(*llmResp.Content)).
				Msg("agent.loop.text_response")

			return &AgentResult{
				Status:     "completed",
				Result:     *llmResp.Content,
				Iterations: iteration,
				ToolCalls:  allToolCalls,
			}
		}

		// 4. Empty response — unexpected.
		logger.Warn().
			Int("iteration", iteration).
			Msg("agent.loop.empty_response")

		errMsg := "LLM returned empty response"
		return &AgentResult{
			Status:     "error",
			Error:      &errMsg,
			ErrorCode:  "empty_response",
			Iterations: iteration,
			ToolCalls:  allToolCalls,
		}
	}

	// Max iterations exhausted.
	logger.Warn().
		Int("max_iterations", maxIterations).
		Msg("agent.loop.max_iterations")

	errMsg := fmt.Sprintf("max iterations (%d) reached", maxIterations)
	return &AgentResult{
		Status:     "error",
		Error:      &errMsg,
		ErrorCode:  "max_iterations",
		Iterations: maxIterations,
		ToolCalls:  allToolCalls,
	}
}

// dispatchToolCalls executes all tool calls from an LLM response.
// For each tool call:
//  1. Restores placeholders to real values in arguments
//  2. Calls MCP tool with real arguments
//  3. On SUCCESS: appends minimal ack to messages (LLM never sees raw results)
//  4. On ERROR: appends redacted error to messages (for LLM retry)
//  5. Records in audit trail (redacted)
//
// Returns the tool call records, raw results for the API consumer, and
// whether any errors occurred.
func (a *AgentCore) dispatchToolCalls(
	ctx context.Context,
	toolCalls []llm.ToolCall,
	messages *[]llm.Message,
	secretCtx *redaction.SecretContext,
	logger zerolog.Logger,
) ([]ToolCallRecord, []map[string]any, bool) {
	records := make([]ToolCallRecord, 0, len(toolCalls))
	rawResults := make([]map[string]any, 0, len(toolCalls))
	hasErrors := false

	for _, tc := range toolCalls {
		record, rawResult, isError := a.executeSingleTool(
			ctx, tc, messages, secretCtx, logger,
		)
		records = append(records, record)
		rawResults = append(rawResults, rawResult)
		if isError {
			hasErrors = true
		}
	}

	return records, rawResults, hasErrors
}

// executeSingleTool handles one tool call through the full security pipeline:
//
//	LLM args (placeholders) → restore → real args → MCP call → ack/error → messages
func (a *AgentCore) executeSingleTool(
	ctx context.Context,
	tc llm.ToolCall,
	messages *[]llm.Message,
	secretCtx *redaction.SecretContext,
	logger zerolog.Logger,
) (ToolCallRecord, map[string]any, bool) {
	startTime := time.Now()

	logger.Info().
		Str("tool_name", tc.Name).
		Str("tool_call_id", tc.ID).
		Msg("agent.tool.executing")

	// 1. Restore placeholders in arguments to real values.
	realArgs := a.redactor.RestoreMapPlaceholders(tc.Arguments, secretCtx)

	// 2. Call MCP tool with real arguments.
	result, err := a.mcp.CallTool(ctx, tc.Name, realArgs)

	duration := time.Since(startTime).Milliseconds()

	if err != nil {
		// Tool invocation failed at the transport/protocol level.
		errMsg := a.redactor.RedactErrorMessage(err.Error(), secretCtx)

		logger.Error().
			Str("tool_name", tc.Name).
			Int64("duration_ms", duration).
			Str("error", errMsg).
			Msg("agent.tool.error")

		// Append redacted error as tool result message for LLM retry.
		toolResultMsg := llm.Message{
			Role:       "tool",
			ToolCallID: tc.ID,
			Content:    stringPtr(fmt.Sprintf(`{"error": %q, "tool_name": %q}`, errMsg, tc.Name)),
		}
		*messages = append(*messages, toolResultMsg)

		// Record for audit (redacted).
		record := ToolCallRecord{
			ToolName:   tc.Name,
			Arguments:  a.redactArguments(tc.Arguments),
			Result:     errMsg,
			IsError:    true,
			DurationMS: duration,
		}

		rawResult := map[string]any{
			"tool_name": tc.Name,
			"error":     errMsg,
			"is_error":  true,
		}

		return record, rawResult, true
	}

	// Tool returned a response (may still be an error response from the tool).
	if result.IsError {
		// Tool-level error (the MCP tool executed but reported an error).
		redactedContent := a.redactor.RedactErrorMessage(result.Content, secretCtx)

		logger.Warn().
			Str("tool_name", tc.Name).
			Int64("duration_ms", duration).
			Msg("agent.tool.tool_error")

		// Append redacted error for LLM retry.
		toolResultMsg := llm.Message{
			Role:       "tool",
			ToolCallID: tc.ID,
			Content:    stringPtr(fmt.Sprintf(`{"error": %q, "tool_name": %q}`, redactedContent, tc.Name)),
		}
		*messages = append(*messages, toolResultMsg)

		record := ToolCallRecord{
			ToolName:   tc.Name,
			Arguments:  a.redactArguments(tc.Arguments),
			Result:     redactedContent,
			IsError:    true,
			DurationMS: duration,
		}

		rawResult := map[string]any{
			"tool_name": tc.Name,
			"error":     result.Content,
			"is_error":  true,
		}

		return record, rawResult, true
	}

	// 3. SUCCESS: LLM gets a minimal ack (NO secrets).
	logger.Info().
		Str("tool_name", tc.Name).
		Int64("duration_ms", duration).
		Int("result_length", len(result.Content)).
		Msg("agent.tool.success")

	ackContent := fmt.Sprintf(`{"status":"ok","tool_name":%q}`, tc.Name)
	toolResultMsg := llm.Message{
		Role:       "tool",
		ToolCallID: tc.ID,
		Content:    stringPtr(ackContent),
	}
	*messages = append(*messages, toolResultMsg)

	// 4. Redacted result for audit trail.
	redactedResult := a.redactor.RedactToolResult(tc.Name, result.Content, secretCtx)

	record := ToolCallRecord{
		ToolName:   tc.Name,
		Arguments:  a.redactArguments(tc.Arguments),
		Result:     redactedResult,
		IsError:    false,
		DurationMS: duration,
	}

	// 5. Raw result for API consumer (contains actual secret data).
	rawResult := map[string]any{
		"tool_name": tc.Name,
		"result":    result.Content,
		"is_error":  false,
	}

	return record, rawResult, false
}

// redactArguments creates a copy of arguments with any secret values
// replaced by "[REDACTED]" for audit logging. The original placeholder
// format from the LLM is kept as-is since it's already safe.
func (a *AgentCore) redactArguments(args map[string]any) map[string]any {
	if len(args) == 0 {
		return args
	}
	// Arguments from the LLM already contain placeholders, not real values.
	// They're safe for audit purposes.
	result := make(map[string]any, len(args))
	for k, v := range args {
		result[k] = v
	}
	return result
}

// classifyLLMError maps an LLM error to an error code string for the AgentResult.
func classifyLLMError(err error) string {
	var authErr *llm.LLMAuthError
	if errors.As(err, &authErr) {
		return "llm_auth"
	}

	var rateLimitErr *llm.LLMRateLimitError
	if errors.As(err, &rateLimitErr) {
		return "llm_rate_limit"
	}

	var serviceErr *llm.LLMServiceError
	if errors.As(err, &serviceErr) {
		return "llm_service"
	}

	var toolErr *llm.LLMToolCallUnsupportedError
	if errors.As(err, &toolErr) {
		return "llm_tool_unsupported"
	}

	var llmErr *llm.LLMError
	if errors.As(err, &llmErr) {
		return "llm_" + llmErr.Code
	}

	return "llm_error"
}
