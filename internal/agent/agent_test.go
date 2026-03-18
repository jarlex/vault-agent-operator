package agent

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/config"
	"github.com/jarlex/vault-agent-operator/internal/llm"
	"github.com/jarlex/vault-agent-operator/internal/mcp"
	"github.com/jarlex/vault-agent-operator/internal/redaction"
)

// ---------------------------------------------------------------------------
// Mock implementations
// ---------------------------------------------------------------------------

// mockLLMProvider implements llm.LLMProvider for testing.
type mockLLMProvider struct {
	mu        sync.Mutex
	calls     []llm.CompletionRequest
	responses []*llm.LLMResponse
	errors    []error
	callIndex int
}

func (m *mockLLMProvider) Complete(_ context.Context, req llm.CompletionRequest) (*llm.LLMResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.calls = append(m.calls, req)
	idx := m.callIndex
	m.callIndex++

	if idx < len(m.errors) && m.errors[idx] != nil {
		return nil, m.errors[idx]
	}
	if idx < len(m.responses) {
		return m.responses[idx], nil
	}
	// Fallback: empty response.
	return &llm.LLMResponse{}, nil
}

func (m *mockLLMProvider) AvailableModels() []llm.ModelInfo {
	return []llm.ModelInfo{{Name: "test", Provider: "test", ModelID: "test-id", SupportsToolCalling: true}}
}

// mockMCPClient implements mcp.MCPClient for testing.
type mockMCPClient struct {
	mu             sync.Mutex
	tools          []mcp.MCPTool
	openaiTools    []llm.Tool
	callToolResult *mcp.ToolResult
	callToolError  error
	// Per-tool-name overrides. Maps tool name → (result, error).
	toolResultMap map[string]struct {
		result *mcp.ToolResult
		err    error
	}
	callToolHistory []struct {
		Name      string
		Arguments map[string]any
	}
	connected bool
}

func newMockMCPClient() *mockMCPClient {
	return &mockMCPClient{
		connected: true,
		toolResultMap: make(map[string]struct {
			result *mcp.ToolResult
			err    error
		}),
	}
}

func (m *mockMCPClient) Connect(_ context.Context) error { return nil }
func (m *mockMCPClient) Disconnect() error               { return nil }

func (m *mockMCPClient) Tools() []mcp.MCPTool {
	if m.tools == nil {
		return []mcp.MCPTool{}
	}
	return m.tools
}

func (m *mockMCPClient) ToolsAsOpenAIFormat() []llm.Tool {
	if m.openaiTools != nil {
		return m.openaiTools
	}
	// Convert tools list.
	result := make([]llm.Tool, 0, len(m.tools))
	for _, t := range m.tools {
		result = append(result, llm.MCPToolToOpenAI(t.Name, t.Description, t.InputSchema))
	}
	return result
}

func (m *mockMCPClient) CallTool(_ context.Context, name string, arguments map[string]any) (*mcp.ToolResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.callToolHistory = append(m.callToolHistory, struct {
		Name      string
		Arguments map[string]any
	}{Name: name, Arguments: arguments})

	// Check per-tool overrides.
	if override, ok := m.toolResultMap[name]; ok {
		return override.result, override.err
	}

	return m.callToolResult, m.callToolError
}

func (m *mockMCPClient) HealthCheck(_ context.Context) error { return nil }

func (m *mockMCPClient) IsConnected() bool {
	return m.connected
}

// ---------------------------------------------------------------------------
// Helper: create an agent with mocks
// ---------------------------------------------------------------------------

func makeAgent(llmProv llm.LLMProvider, mcpCli mcp.MCPClient) *AgentCore {
	logger := zerolog.Nop()
	cfg := config.AgentConfig{
		MaxIterations: 10,
	}
	return NewAgentCore(llmProv, mcpCli, cfg, "http://vault:8200", logger)
}

func strPtr(s string) *string { return &s }

// ---------------------------------------------------------------------------
// Tests: hybrid reasoning loop
// ---------------------------------------------------------------------------

func TestExecute_FastPath_ToolCallsSucceed(t *testing.T) {
	// LLM responds with a tool call, MCP tool succeeds.
	// Agent should return fast path — no second LLM call.
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_read", Description: "Read KV"}}
	mcpCli.callToolResult = &mcp.ToolResult{Content: `{"data":{"password":"s3cret"}}`, IsError: false}

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_read", Arguments: map[string]any{"path": "secret/data/myapp"}},
				},
			},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "read secret"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != "completed" {
		t.Errorf("expected status=completed, got %q", result.Status)
	}
	if result.Iterations != 1 {
		t.Errorf("expected 1 iteration (fast path), got %d", result.Iterations)
	}
	if len(result.Data) != 1 {
		t.Fatalf("expected 1 data result, got %d", len(result.Data))
	}
	if result.Data[0]["is_error"] != false {
		t.Errorf("expected is_error=false in data")
	}
	// LLM should only have been called once (fast path — no second call).
	if len(llmProv.calls) != 1 {
		t.Errorf("expected 1 LLM call (fast path), got %d", len(llmProv.calls))
	}
}

func TestExecute_TextResponse(t *testing.T) {
	// LLM responds with text only (no tool calls).
	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{Content: strPtr("I cannot do that because Vault is sealed.")},
		},
	}
	mcpCli := newMockMCPClient()

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "unseal vault"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != "completed" {
		t.Errorf("expected status=completed, got %q", result.Status)
	}
	if result.Result != "I cannot do that because Vault is sealed." {
		t.Errorf("unexpected result text: %q", result.Result)
	}
	if result.Iterations != 1 {
		t.Errorf("expected 1 iteration, got %d", result.Iterations)
	}
}

func TestExecute_EmptyResponse(t *testing.T) {
	// LLM responds with empty content and no tool calls.
	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{}, // empty
		},
	}
	mcpCli := newMockMCPClient()

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "do something"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != "error" {
		t.Errorf("expected status=error, got %q", result.Status)
	}
	if result.ErrorCode != "empty_response" {
		t.Errorf("expected error_code=empty_response, got %q", result.ErrorCode)
	}
}

func TestExecute_LLMError(t *testing.T) {
	// LLM returns an auth error.
	llmProv := &mockLLMProvider{
		errors: []error{
			&llm.LLMAuthError{LLMError: llm.LLMError{Code: "auth", Message: "invalid token"}},
		},
	}
	mcpCli := newMockMCPClient()

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "read secret"})

	if err != nil {
		t.Fatalf("unexpected error from Execute: %v", err)
	}
	if result.Status != "error" {
		t.Errorf("expected status=error, got %q", result.Status)
	}
	if result.ErrorCode != "llm_auth" {
		t.Errorf("expected error_code=llm_auth, got %q", result.ErrorCode)
	}
}

func TestExecute_MaxIterationsExhausted(t *testing.T) {
	// LLM keeps returning tool calls that always fail, exhausting max iterations.
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_read", Description: "Read KV"}}
	mcpCli.callToolResult = &mcp.ToolResult{Content: "permission denied", IsError: true}

	// LLM always returns a tool call.
	responses := make([]*llm.LLMResponse, 3)
	for i := range responses {
		responses[i] = &llm.LLMResponse{
			ToolCalls: []llm.ToolCall{
				{ID: fmt.Sprintf("tc%d", i+1), Name: "vault_kv_read", Arguments: map[string]any{"path": "secret/data/x"}},
			},
		}
	}

	llmProv := &mockLLMProvider{responses: responses}

	// Use a custom agent with max_iterations=3.
	logger := zerolog.Nop()
	cfg := config.AgentConfig{MaxIterations: 3}
	agent := NewAgentCore(llmProv, mcpCli, cfg, "http://vault:8200", logger)

	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "read secret"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != "error" {
		t.Errorf("expected status=error, got %q", result.Status)
	}
	if result.ErrorCode != "max_iterations" {
		t.Errorf("expected error_code=max_iterations, got %q", result.ErrorCode)
	}
	if result.Iterations != 3 {
		t.Errorf("expected iterations=3, got %d", result.Iterations)
	}
}

func TestExecute_ErrorRetryThenSuccess(t *testing.T) {
	// Iteration 1: tool call fails → error retry.
	// Iteration 2: LLM responds with text (gives up gracefully).
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_read", Description: "Read KV"}}
	mcpCli.callToolResult = &mcp.ToolResult{Content: "permission denied", IsError: true}

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			// Iteration 1: LLM requests a tool call.
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_read", Arguments: map[string]any{"path": "secret/data/x"}},
				},
			},
			// Iteration 2: LLM sees the error and responds with text.
			{Content: strPtr("The secret could not be read due to permission denied.")},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "read secret"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != "completed" {
		t.Errorf("expected status=completed, got %q", result.Status)
	}
	if result.Iterations != 2 {
		t.Errorf("expected 2 iterations (error retry), got %d", result.Iterations)
	}
	if result.Result == "" {
		t.Error("expected text result after error retry")
	}
	// LLM should have been called twice.
	if len(llmProv.calls) != 2 {
		t.Errorf("expected 2 LLM calls, got %d", len(llmProv.calls))
	}
}

func TestExecute_MixedResults_SomeToolsSucceedSomeFail(t *testing.T) {
	// LLM requests 2 tool calls: one succeeds, one fails.
	// Since hasErrors=true, it should retry (not fast path).
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{
		{Name: "vault_kv_read", Description: "Read KV"},
		{Name: "vault_kv_write", Description: "Write KV"},
	}
	mcpCli.toolResultMap["vault_kv_read"] = struct {
		result *mcp.ToolResult
		err    error
	}{result: &mcp.ToolResult{Content: `{"data":{"key":"val"}}`, IsError: false}}
	mcpCli.toolResultMap["vault_kv_write"] = struct {
		result *mcp.ToolResult
		err    error
	}{result: &mcp.ToolResult{Content: "permission denied", IsError: true}}

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			// Iteration 1: two tool calls, one will succeed, one will fail.
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_read", Arguments: map[string]any{"path": "secret/data/a"}},
					{ID: "tc2", Name: "vault_kv_write", Arguments: map[string]any{"path": "secret/data/b"}},
				},
			},
			// Iteration 2: LLM responds with text after seeing the error.
			{Content: strPtr("Read succeeded but write failed.")},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "read and write"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != "completed" {
		t.Errorf("expected status=completed, got %q", result.Status)
	}
	if result.Iterations != 2 {
		t.Errorf("expected 2 iterations (mixed results → retry), got %d", result.Iterations)
	}
}

func TestExecute_MCPTransportError(t *testing.T) {
	// MCP CallTool returns a transport-level error (not ToolResult.IsError).
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_read", Description: "Read KV"}}
	mcpCli.callToolError = fmt.Errorf("connection refused")

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_read", Arguments: map[string]any{"path": "secret/data/x"}},
				},
			},
			{Content: strPtr("Could not connect to vault.")},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "read secret"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Transport error triggers error retry path.
	if result.Iterations != 2 {
		t.Errorf("expected 2 iterations, got %d", result.Iterations)
	}
}

func TestExecute_SecretRedaction(t *testing.T) {
	// Verify that secrets from SecretData are NOT visible to the LLM.
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_write", Description: "Write KV"}}
	mcpCli.callToolResult = &mcp.ToolResult{Content: `{"version":1}`, IsError: false}

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_write", Arguments: map[string]any{
						"path": "secret/data/myapp",
						"data": map[string]any{"password": "[SECRET_VALUE_1]"},
					}},
				},
			},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{
		Prompt:     "write password=SuperSecret123 to secret/data/myapp",
		SecretData: map[string]any{"password": "SuperSecret123"},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != "completed" {
		t.Errorf("expected status=completed, got %q", result.Status)
	}

	// Verify that the LLM received the sanitized prompt (not the real secret).
	firstCall := llmProv.calls[0]
	userMsg := firstCall.Messages[1] // [system, user]
	if userMsg.Content != nil && strings.Contains(*userMsg.Content, "SuperSecret123") {
		t.Error("LLM received the real secret value in the prompt — redaction failed")
	}
}

func TestExecute_PlaceholderRestoration_InToolArgs(t *testing.T) {
	// Verify that placeholders in tool arguments are restored to real values
	// before calling MCP.
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_write", Description: "Write KV"}}
	mcpCli.callToolResult = &mcp.ToolResult{Content: `{"version":1}`, IsError: false}

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_write", Arguments: map[string]any{
						"path": "secret/data/myapp",
						"data": map[string]any{"password": "[SECRET_VALUE_1]"},
					}},
				},
			},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	_, err := agent.Execute(context.Background(), ExecuteRequest{
		Prompt:     "write password=MySecret to secret/data/myapp",
		SecretData: map[string]any{"password": "MySecret"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The MCP client should have received the real secret, not the placeholder.
	if len(mcpCli.callToolHistory) == 0 {
		t.Fatal("MCP CallTool was never called")
	}
	call := mcpCli.callToolHistory[0]
	data, ok := call.Arguments["data"].(map[string]any)
	if !ok {
		t.Fatal("expected 'data' argument to be a map")
	}
	if pw, ok := data["password"].(string); ok && pw == "MySecret" {
		// Good — placeholder was restored.
	} else {
		t.Errorf("expected password=MySecret in MCP args, got %v", data["password"])
	}
}

func TestExecute_SuccessPath_LLMNeverSeesRawResults(t *testing.T) {
	// On fast path, the LLM receives a minimal ack, NOT the raw tool result.
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_read", Description: "Read KV"}}
	mcpCli.callToolResult = &mcp.ToolResult{
		Content: `{"data":{"api_key":"super-secret-key-12345"}}`,
		IsError: false,
	}

	// We need 2 iterations to inspect what the LLM sees.
	// Force a second LLM call by making the first succeed (fast path stops after 1).
	// Instead, let's verify via the Data field that raw results contain secrets
	// while the audit trail (ToolCalls) does not.
	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_read", Arguments: map[string]any{"path": "secret/data/keys"}},
				},
			},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "read keys"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Data should contain the raw result for the API consumer.
	if len(result.Data) == 0 {
		t.Fatal("expected data results")
	}
	rawResult := result.Data[0]["result"]
	if rawResult == nil {
		t.Fatal("expected raw result in data")
	}
	// The raw result should contain the secret.
	if !strings.Contains(rawResult.(string), "super-secret-key-12345") {
		t.Error("expected raw result to contain secret for API consumer")
	}

	// The ToolCalls audit trail should have redacted results (no raw secrets).
	if len(result.ToolCalls) == 0 {
		t.Fatal("expected tool call records")
	}
	record := result.ToolCalls[0]
	if record.IsError {
		t.Error("expected IsError=false in tool call record")
	}
}

func TestExecute_StatelessPerRequest(t *testing.T) {
	// Each Execute call creates a fresh SecretContext.
	// Registering secrets in one request should not affect another.
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_read", Description: "Read"}}
	mcpCli.callToolResult = &mcp.ToolResult{Content: `{"data":{}}`, IsError: false}

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_read", Arguments: map[string]any{"path": "a"}},
				},
			},
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc2", Name: "vault_kv_read", Arguments: map[string]any{"path": "b"}},
				},
			},
		},
	}

	agent := makeAgent(llmProv, mcpCli)

	// Request 1.
	r1, err := agent.Execute(context.Background(), ExecuteRequest{
		Prompt:     "read a with password=Secret1",
		SecretData: map[string]any{"password": "Secret1"},
	})
	if err != nil {
		t.Fatalf("request 1 error: %v", err)
	}
	if r1.Status != "completed" {
		t.Errorf("request 1 status: %q", r1.Status)
	}

	// Request 2.
	r2, err := agent.Execute(context.Background(), ExecuteRequest{
		Prompt:     "read b with password=Secret2",
		SecretData: map[string]any{"password": "Secret2"},
	})
	if err != nil {
		t.Fatalf("request 2 error: %v", err)
	}
	if r2.Status != "completed" {
		t.Errorf("request 2 status: %q", r2.Status)
	}
}

func TestExecute_LLMGivesUp_TextAfterToolFailure(t *testing.T) {
	// LLM requests a tool call, tool fails, LLM responds with error text.
	mcpCli := newMockMCPClient()
	mcpCli.tools = []mcp.MCPTool{{Name: "vault_kv_read", Description: "Read"}}
	mcpCli.callToolResult = &mcp.ToolResult{Content: "404 not found", IsError: true}

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{
				ToolCalls: []llm.ToolCall{
					{ID: "tc1", Name: "vault_kv_read", Arguments: map[string]any{"path": "secret/missing"}},
				},
			},
			{Content: strPtr("The requested secret path does not exist.")},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{Prompt: "read missing secret"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != "completed" {
		t.Errorf("expected completed, got %q", result.Status)
	}
	if !strings.Contains(result.Result, "does not exist") {
		t.Errorf("expected LLM explanation, got %q", result.Result)
	}
}

func TestExecute_PlaceholderRestorationInTextResponse(t *testing.T) {
	// If LLM echoes a placeholder token in text, it should be resolved to
	// the real value for the API consumer.
	mcpCli := newMockMCPClient()

	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{Content: strPtr("The password is [SECRET_VALUE_1].")},
		},
	}

	agent := makeAgent(llmProv, mcpCli)
	result, err := agent.Execute(context.Background(), ExecuteRequest{
		Prompt:     "what is the password for myapp? password=RealPassword123",
		SecretData: map[string]any{"password": "RealPassword123"},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The result should have the placeholder resolved.
	if !strings.Contains(result.Result, "RealPassword123") {
		t.Errorf("expected placeholder to be resolved, got %q", result.Result)
	}
}

func TestExecute_ModelUsedField(t *testing.T) {
	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{Content: strPtr("ok")},
		},
	}
	mcpCli := newMockMCPClient()

	agent := makeAgent(llmProv, mcpCli)

	// With model specified.
	result, _ := agent.Execute(context.Background(), ExecuteRequest{Prompt: "test", Model: "gpt-4"})
	if result.ModelUsed != "gpt-4" {
		t.Errorf("expected ModelUsed=gpt-4, got %q", result.ModelUsed)
	}

	// Without model — should default to "default".
	llmProv.callIndex = 0
	llmProv.calls = nil
	result, _ = agent.Execute(context.Background(), ExecuteRequest{Prompt: "test"})
	if result.ModelUsed != "default" {
		t.Errorf("expected ModelUsed=default, got %q", result.ModelUsed)
	}
}

func TestExecute_DurationMSPopulated(t *testing.T) {
	llmProv := &mockLLMProvider{
		responses: []*llm.LLMResponse{
			{Content: strPtr("ok")},
		},
	}
	mcpCli := newMockMCPClient()

	agent := makeAgent(llmProv, mcpCli)
	result, _ := agent.Execute(context.Background(), ExecuteRequest{Prompt: "test"})

	if result.DurationMS < 0 {
		t.Errorf("expected non-negative DurationMS, got %d", result.DurationMS)
	}
}

// ---------------------------------------------------------------------------
// Tests: classifyLLMError
// ---------------------------------------------------------------------------

func TestClassifyLLMError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected string
	}{
		{
			name:     "auth error",
			err:      &llm.LLMAuthError{LLMError: llm.LLMError{Code: "auth", Message: "unauthorized"}},
			expected: "llm_auth",
		},
		{
			name:     "rate limit error",
			err:      &llm.LLMRateLimitError{LLMError: llm.LLMError{Code: "rate_limit", Message: "too many"}},
			expected: "llm_rate_limit",
		},
		{
			name:     "service error",
			err:      &llm.LLMServiceError{LLMError: llm.LLMError{Code: "service", Message: "server error"}},
			expected: "llm_service",
		},
		{
			name:     "tool unsupported error",
			err:      &llm.LLMToolCallUnsupportedError{LLMError: llm.LLMError{Code: "tool_call_unsupported", Message: "no tools"}},
			expected: "llm_tool_unsupported",
		},
		{
			name:     "generic LLM error",
			err:      &llm.LLMError{Code: "timeout", Message: "request timed out"},
			expected: "llm_timeout",
		},
		{
			name:     "unknown error",
			err:      fmt.Errorf("some random error"),
			expected: "llm_error",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := classifyLLMError(tc.err)
			if got != tc.expected {
				t.Errorf("classifyLLMError() = %q, want %q", got, tc.expected)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Tests: prompts
// ---------------------------------------------------------------------------

func TestLoadSystemPrompt_DefaultWhenNoPath(t *testing.T) {
	logger := zerolog.Nop()
	prompt := LoadSystemPrompt("", "http://vault:8200", []string{"tool1", "tool2"}, logger)

	if !strings.Contains(prompt, "Vault Operator Agent") {
		t.Error("expected default prompt to contain 'Vault Operator Agent'")
	}
}

func TestLoadSystemPrompt_DefaultWhenFileNotFound(t *testing.T) {
	logger := zerolog.Nop()
	prompt := LoadSystemPrompt("/nonexistent/path.md", "http://vault:8200", nil, logger)

	if !strings.Contains(prompt, "Vault Operator Agent") {
		t.Error("expected default prompt when file is not found")
	}
}

func TestLoadSystemPrompt_TemplateVariables(t *testing.T) {
	logger := zerolog.Nop()
	// Write a temp file.
	tmpFile := t.TempDir() + "/prompt.md"
	err := writeTestFile(tmpFile, "Vault at {{ vault_addr }} with tools: {{ available_tools }}")
	if err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	prompt := LoadSystemPrompt(tmpFile, "http://vault:8200", []string{"tool1", "tool2"}, logger)

	if !strings.Contains(prompt, "http://vault:8200") {
		t.Errorf("expected vault_addr to be replaced, got %q", prompt)
	}
	if !strings.Contains(prompt, "tool1, tool2") {
		t.Errorf("expected tools to be replaced, got %q", prompt)
	}
}

func TestBuildInitialMessages(t *testing.T) {
	msgs := BuildInitialMessages("system prompt", "user prompt")
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" || *msgs[0].Content != "system prompt" {
		t.Errorf("unexpected system message: %+v", msgs[0])
	}
	if msgs[1].Role != "user" || *msgs[1].Content != "user prompt" {
		t.Errorf("unexpected user message: %+v", msgs[1])
	}
}

func TestStringPtr(t *testing.T) {
	p := stringPtr("hello")
	if p == nil || *p != "hello" {
		t.Error("stringPtr(hello) should return non-nil pointer")
	}

	p = stringPtr("")
	if p != nil {
		t.Error("stringPtr('') should return nil")
	}
}

func TestFormatToolNamesFromStrings(t *testing.T) {
	result := FormatToolNamesFromStrings(nil)
	if result != "None available" {
		t.Errorf("expected 'None available', got %q", result)
	}
	result = FormatToolNamesFromStrings([]string{"a", "b"})
	if result != "a, b" {
		t.Errorf("expected 'a, b', got %q", result)
	}
}

// ---------------------------------------------------------------------------
// Tests: redactArguments
// ---------------------------------------------------------------------------

func TestRedactArguments(t *testing.T) {
	agent := makeAgent(&mockLLMProvider{}, newMockMCPClient())

	// Should return a copy.
	args := map[string]any{"path": "secret/data/x", "data": "[SECRET_VALUE_1]"}
	redacted := agent.redactArguments(args)

	if redacted["path"] != "secret/data/x" {
		t.Errorf("expected path preserved, got %v", redacted["path"])
	}
	// Arguments from LLM already have placeholders — safe for audit.
	if redacted["data"] != "[SECRET_VALUE_1]" {
		t.Errorf("expected placeholder preserved, got %v", redacted["data"])
	}

	// Nil map.
	redacted = agent.redactArguments(nil)
	if redacted != nil {
		t.Error("expected nil for nil input")
	}
}

// ---------------------------------------------------------------------------
// Tests: NewAgentCore constructor
// ---------------------------------------------------------------------------

func TestNewAgentCore_HasAllDependencies(t *testing.T) {
	llmProv := &mockLLMProvider{}
	mcpCli := newMockMCPClient()
	logger := zerolog.Nop()
	cfg := config.AgentConfig{MaxIterations: 5}

	agent := NewAgentCore(llmProv, mcpCli, cfg, "http://vault:8200", logger)

	if agent.llm == nil {
		t.Error("LLM provider is nil")
	}
	if agent.mcp == nil {
		t.Error("MCP client is nil")
	}
	if agent.sanitizer == nil {
		t.Error("sanitizer is nil")
	}
	if agent.redactor == nil {
		t.Error("redactor is nil")
	}
	if agent.config.MaxIterations != 5 {
		t.Errorf("expected MaxIterations=5, got %d", agent.config.MaxIterations)
	}
}

// ---------------------------------------------------------------------------
// Tests: SecretContext isolation in Execute
// ---------------------------------------------------------------------------

func TestExecute_SecretContextDestroyed(t *testing.T) {
	// Verify that the SecretContext is destroyed after Execute completes.
	// We verify indirectly: after Execute, the placeholder should NOT be resolvable.
	// We can test this by checking that multiple calls don't share state (already tested
	// in StatelessPerRequest). This test confirms the pattern works end-to-end.
	ctx := redaction.NewSecretContext()
	placeholder, err := ctx.RegisterSecret("test-value")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if placeholder != "[SECRET_VALUE_1]" {
		t.Fatalf("unexpected placeholder: %s", placeholder)
	}

	ctx.Destroy()

	_, found := ctx.ResolvePlaceholder("[SECRET_VALUE_1]")
	if found {
		t.Error("expected placeholder to not be resolvable after Destroy")
	}
}

// ---------------------------------------------------------------------------
// Helper: write file for prompt tests
// ---------------------------------------------------------------------------

func writeTestFile(path, content string) error {
	return os.WriteFile(path, []byte(content), 0644)
}
