package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/agent"
	"github.com/jarlex/vault-agent-operator/internal/config"
	"github.com/jarlex/vault-agent-operator/internal/llm"
	"github.com/jarlex/vault-agent-operator/internal/logging"
)

// --- Mock implementations ---

type mockAgentExecutor struct {
	result *agent.AgentResult
	err    error
}

func (m *mockAgentExecutor) Execute(_ context.Context, _ agent.ExecuteRequest) (*agent.AgentResult, error) {
	return m.result, m.err
}

type mockMCPHealthChecker struct {
	connected bool
	healthErr error
}

func (m *mockMCPHealthChecker) HealthCheck(_ context.Context) error {
	return m.healthErr
}

func (m *mockMCPHealthChecker) IsConnected() bool {
	return m.connected
}

type mockModelLister struct {
	models []llm.ModelInfo
}

func (m *mockModelLister) AvailableModels() []llm.ModelInfo {
	return m.models
}

// --- Helpers ---

func newTestHandler(agentExec AgentExecutor, mcpClient MCPHealthChecker, llmProvider ModelLister) *Handler {
	logger := zerolog.Nop()
	return NewHandler(agentExec, mcpClient, llmProvider, HandlerConfig{
		LLMConfig: config.LLMConfig{DefaultModel: "default"},
		Version:   "test-1.0.0",
	}, logger)
}

// injectLogger injects a nop logger into the request context so middleware and
// handlers that call logging.FromContext don't get a nil logger.
func injectLogger(r *http.Request) *http.Request {
	ctx := logging.NewContext(r.Context(), zerolog.Nop())
	return r.WithContext(ctx)
}

// --- CreateTask tests ---

func TestCreateTask_ValidRequest(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{
			result: &agent.AgentResult{
				Status:     "completed",
				Result:     "Secret value is abc123",
				ModelUsed:  "gpt-4o",
				DurationMS: 1234,
			},
		},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	body := `{"prompt":"Read secret/data/test"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp TaskResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.Status != "completed" {
		t.Errorf("expected status=completed, got %q", resp.Status)
	}
	if resp.Result != "Secret value is abc123" {
		t.Errorf("unexpected result: %q", resp.Result)
	}
	if resp.ModelUsed != "gpt-4o" {
		t.Errorf("expected model=gpt-4o, got %q", resp.ModelUsed)
	}
	if resp.DurationMS != 1234 {
		t.Errorf("expected duration=1234, got %d", resp.DurationMS)
	}
}

func TestCreateTask_InvalidJSON(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader("not json{"))
	req.Header.Set("Content-Type", "application/json")
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}

	var resp ErrorResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode error: %v", err)
	}
	if !strings.Contains(resp.Error, "Invalid JSON") {
		t.Errorf("expected 'Invalid JSON' in error, got %q", resp.Error)
	}
}

func TestCreateTask_EmptyPrompt(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	body := `{"prompt":""}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

func TestCreateTask_PromptTooLong(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	longPrompt := strings.Repeat("a", 4097)
	body, _ := json.Marshal(TaskRequest{Prompt: longPrompt})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", bytes.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

func TestCreateTask_InvalidMaxIterations(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	tests := []struct {
		name string
		val  int
	}{
		{"zero", 0},
		{"negative", -1},
		{"too high", 101},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			val := tc.val
			body, _ := json.Marshal(TaskRequest{Prompt: "test", MaxIterations: &val})
			req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", bytes.NewReader(body))
			req = injectLogger(req)
			w := httptest.NewRecorder()

			h.CreateTask(w, req)

			if w.Code != http.StatusBadRequest {
				t.Errorf("expected status 400, got %d", w.Code)
			}
		})
	}
}

func TestCreateTask_WithModel(t *testing.T) {
	var capturedReq agent.ExecuteRequest
	h := newTestHandler(
		&mockAgentExecutor{
			result: &agent.AgentResult{Status: "completed", Result: "ok"},
		},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	// Override with a capturing executor.
	h.agent = &capturingExecutor{
		result:   &agent.AgentResult{Status: "completed", Result: "ok"},
		captured: &capturedReq,
	}

	model := "gpt-4o"
	body, _ := json.Marshal(TaskRequest{Prompt: "test", Model: &model})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", bytes.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
	if capturedReq.Model != "gpt-4o" {
		t.Errorf("expected model=gpt-4o in execute request, got %q", capturedReq.Model)
	}
}

type capturingExecutor struct {
	result   *agent.AgentResult
	captured *agent.ExecuteRequest
}

func (c *capturingExecutor) Execute(_ context.Context, req agent.ExecuteRequest) (*agent.AgentResult, error) {
	*c.captured = req
	return c.result, nil
}

func TestCreateTask_AgentError(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{err: errors.New("agent crashed")},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	body := `{"prompt":"test"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", w.Code)
	}
}

func TestCreateTask_ErrorCodeMapping(t *testing.T) {
	tests := []struct {
		errorCode  string
		wantStatus int
	}{
		{"", http.StatusOK},
		{"llm_auth", http.StatusBadGateway},
		{"llm_rate_limit", http.StatusTooManyRequests},
		{"llm_service", http.StatusBadGateway},
		{"max_iterations", http.StatusOK},
		{"empty_response", http.StatusInternalServerError},
		{"something_unknown", http.StatusInternalServerError},
	}

	for _, tc := range tests {
		t.Run(tc.errorCode, func(t *testing.T) {
			errMsg := "test error"
			h := newTestHandler(
				&mockAgentExecutor{
					result: &agent.AgentResult{
						Status:    "error",
						ErrorCode: tc.errorCode,
						Error:     &errMsg,
					},
				},
				&mockMCPHealthChecker{connected: true},
				&mockModelLister{},
			)

			body := `{"prompt":"test"}`
			req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(body))
			req = injectLogger(req)
			w := httptest.NewRecorder()

			h.CreateTask(w, req)

			if w.Code != tc.wantStatus {
				t.Errorf("errorCode=%q: expected status %d, got %d", tc.errorCode, tc.wantStatus, w.Code)
			}
		})
	}
}

// --- Health tests ---

func TestHealth_Healthy(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true, healthErr: nil},
		&mockModelLister{},
	)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.Health(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp HealthResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Status != "healthy" {
		t.Errorf("expected status=healthy, got %q", resp.Status)
	}
	if resp.Agent != "running" {
		t.Errorf("expected agent=running, got %q", resp.Agent)
	}
	if resp.VaultMCP != "connected" {
		t.Errorf("expected vault_mcp=connected, got %q", resp.VaultMCP)
	}
	if resp.VaultServer != "ok" {
		t.Errorf("expected vault_server=ok, got %q", resp.VaultServer)
	}
	if resp.Version != "test-1.0.0" {
		t.Errorf("expected version=test-1.0.0, got %q", resp.Version)
	}
}

func TestHealth_Disconnected(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: false},
		&mockModelLister{},
	)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.Health(w, req)

	var resp HealthResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Status != "unhealthy" {
		t.Errorf("expected status=unhealthy, got %q", resp.Status)
	}
	if resp.VaultMCP != "disconnected" {
		t.Errorf("expected vault_mcp=disconnected, got %q", resp.VaultMCP)
	}
}

func TestHealth_Degraded(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true, healthErr: errors.New("vault unreachable")},
		&mockModelLister{},
	)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.Health(w, req)

	var resp HealthResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Status != "degraded" {
		t.Errorf("expected status=degraded, got %q", resp.Status)
	}
	if resp.VaultServer != "error" {
		t.Errorf("expected vault_server=error, got %q", resp.VaultServer)
	}
}

func TestHealth_UptimeSeconds(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	// Force startTime to be in the past.
	h.startTime = time.Now().Add(-10 * time.Second)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.Health(w, req)

	var resp HealthResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.UptimeSeconds < 9 {
		t.Errorf("expected uptime >= 9s, got %f", resp.UptimeSeconds)
	}
}

// --- Models tests ---

func TestModels(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{
			models: []llm.ModelInfo{
				{Name: "default", Provider: "github", ModelID: "gpt-4o", SupportsToolCalling: true},
				{Name: "small", Provider: "github", ModelID: "gpt-4o-mini", SupportsToolCalling: true},
			},
		},
	)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/models", nil)
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.Models(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp ModelsResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.DefaultModel != "default" {
		t.Errorf("expected default_model=default, got %q", resp.DefaultModel)
	}
	if len(resp.AvailableModels) != 2 {
		t.Fatalf("expected 2 models, got %d", len(resp.AvailableModels))
	}
	if !resp.AvailableModels[0].IsDefault {
		t.Error("expected first model to be default")
	}
	if resp.AvailableModels[1].IsDefault {
		t.Error("expected second model not to be default")
	}
}

func TestModels_Empty(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{models: []llm.ModelInfo{}},
	)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/models", nil)
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.Models(w, req)

	var resp ModelsResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.AvailableModels) != 0 {
		t.Errorf("expected empty models, got %d", len(resp.AvailableModels))
	}
}

// --- mapErrorCodeToStatus tests ---

func TestMapErrorCodeToStatus(t *testing.T) {
	tests := []struct {
		code string
		want int
	}{
		{"", http.StatusOK},
		{"llm_auth", http.StatusBadGateway},
		{"llm_rate_limit", http.StatusTooManyRequests},
		{"llm_service", http.StatusBadGateway},
		{"max_iterations", http.StatusOK},
		{"empty_response", http.StatusInternalServerError},
		{"unknown_code", http.StatusInternalServerError},
	}

	for _, tc := range tests {
		t.Run(tc.code, func(t *testing.T) {
			got := mapErrorCodeToStatus(tc.code)
			if got != tc.want {
				t.Errorf("mapErrorCodeToStatus(%q) = %d, want %d", tc.code, got, tc.want)
			}
		})
	}
}

// --- writeJSON / writeError tests ---

func TestWriteJSON(t *testing.T) {
	w := httptest.NewRecorder()
	writeJSON(w, http.StatusCreated, map[string]string{"key": "value"})

	if w.Code != http.StatusCreated {
		t.Errorf("expected status 201, got %d", w.Code)
	}
	ct := w.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("expected Content-Type=application/json, got %q", ct)
	}

	var body map[string]string
	json.NewDecoder(w.Body).Decode(&body)
	if body["key"] != "value" {
		t.Errorf("expected key=value, got %v", body)
	}
}

func TestWriteError(t *testing.T) {
	w := httptest.NewRecorder()
	detail := "more info"
	writeError(w, http.StatusBadRequest, "bad request", &detail)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}

	var resp ErrorResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Error != "bad request" {
		t.Errorf("expected error=bad request, got %q", resp.Error)
	}
	if resp.Detail == nil || *resp.Detail != "more info" {
		t.Errorf("expected detail=more info, got %v", resp.Detail)
	}
}

// --- Middleware tests ---

func TestRequestIDMiddleware_GeneratesID(t *testing.T) {
	handler := RequestIDMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := RequestIDFromContext(r.Context())
		if id == "" {
			t.Error("expected request ID in context")
		}
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	ctx := logging.NewContext(req.Context(), zerolog.Nop())
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	rid := w.Header().Get("X-Request-ID")
	if rid == "" {
		t.Error("expected X-Request-ID header")
	}
}

func TestRequestIDMiddleware_PreservesExisting(t *testing.T) {
	handler := RequestIDMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := RequestIDFromContext(r.Context())
		if id != "my-custom-id" {
			t.Errorf("expected my-custom-id, got %q", id)
		}
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("X-Request-ID", "my-custom-id")
	ctx := logging.NewContext(req.Context(), zerolog.Nop())
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Header().Get("X-Request-ID") != "my-custom-id" {
		t.Error("expected existing X-Request-ID to be preserved")
	}
}

func TestRecoveryMiddleware_CatchesPanic(t *testing.T) {
	handler := RecoveryMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	ctx := logging.NewContext(req.Context(), zerolog.Nop())
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", w.Code)
	}

	var resp ErrorResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Error != "Internal server error" {
		t.Errorf("expected 'Internal server error', got %q", resp.Error)
	}
}

func TestRecoveryMiddleware_NoPanic(t *testing.T) {
	handler := RecoveryMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	ctx := logging.NewContext(req.Context(), zerolog.Nop())
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func TestTimeoutMiddleware(t *testing.T) {
	handler := TimeoutMiddleware(50 * time.Millisecond)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deadline, ok := r.Context().Deadline()
		if !ok {
			t.Error("expected deadline in context")
		}
		if time.Until(deadline) > 100*time.Millisecond {
			t.Error("deadline seems too far in the future")
		}
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func TestLoggingMiddleware(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("hello"))
	})
	handler := LoggingMiddleware(inner)

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	ctx := logging.NewContext(req.Context(), zerolog.Nop())
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

// --- statusWriter tests ---

func TestStatusWriter(t *testing.T) {
	w := httptest.NewRecorder()
	sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}

	sw.WriteHeader(http.StatusNotFound)
	sw.Write([]byte("not found"))

	if sw.status != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", sw.status)
	}
	if sw.written != 9 {
		t.Errorf("expected 9 bytes written, got %d", sw.written)
	}
}

// --- RequestIDFromContext ---

func TestRequestIDFromContext_NotSet(t *testing.T) {
	id := RequestIDFromContext(context.Background())
	if id != "" {
		t.Errorf("expected empty string, got %q", id)
	}
}

// --- Additional API tests for coverage ---

// Test full server routing via chi (integration-level).
func TestServer_Routes(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{
			result: &agent.AgentResult{Status: "completed", Result: "ok"},
		},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{models: []llm.ModelInfo{
			{Name: "default", Provider: "github", ModelID: "gpt-4o", SupportsToolCalling: true},
		}},
	)
	logger := zerolog.Nop()
	srv := NewServer(ServerConfig{
		Host:           "127.0.0.1",
		Port:           0,
		RequestTimeout: 30 * time.Second,
		Version:        "test",
	}, handler, logger)

	// Test POST /api/v1/tasks via the full router.
	body := `{"prompt":"test prompt"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("POST /api/v1/tasks: expected 200, got %d", w.Code)
	}

	// Verify X-Request-ID is present.
	if rid := w.Header().Get("X-Request-ID"); rid == "" {
		t.Error("expected X-Request-ID in response")
	}
}

func TestServer_HealthRoute(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	logger := zerolog.Nop()
	srv := NewServer(ServerConfig{
		Host:    "127.0.0.1",
		Port:    0,
		Version: "test",
	}, handler, logger)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	w := httptest.NewRecorder()
	srv.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("GET /api/v1/health: expected 200, got %d", w.Code)
	}

	// Verify Content-Type.
	ct := w.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("expected Content-Type=application/json, got %q", ct)
	}
}

func TestServer_ModelsRoute(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{models: []llm.ModelInfo{
			{Name: "default", Provider: "github", ModelID: "gpt-4o"},
		}},
	)
	logger := zerolog.Nop()
	srv := NewServer(ServerConfig{
		Host:    "127.0.0.1",
		Port:    0,
		Version: "test",
	}, handler, logger)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/models", nil)
	w := httptest.NewRecorder()
	srv.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("GET /api/v1/models: expected 200, got %d", w.Code)
	}
}

func TestServer_NotFoundRoute(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	logger := zerolog.Nop()
	srv := NewServer(ServerConfig{
		Host:    "127.0.0.1",
		Port:    0,
		Version: "test",
	}, handler, logger)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/nonexistent", nil)
	w := httptest.NewRecorder()
	srv.router.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound && w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 404 or 405, got %d", w.Code)
	}
}

func TestServer_MethodNotAllowed(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	logger := zerolog.Nop()
	srv := NewServer(ServerConfig{
		Host:    "127.0.0.1",
		Port:    0,
		Version: "test",
	}, handler, logger)

	// GET on a POST-only route.
	req := httptest.NewRequest(http.MethodGet, "/api/v1/tasks", nil)
	w := httptest.NewRecorder()
	srv.router.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed && w.Code != http.StatusNotFound {
		t.Errorf("expected 405 or 404, got %d", w.Code)
	}
}

// Test that the request ID middleware propagates to response header via full router.
func TestServer_RequestIDInResponse(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	logger := zerolog.Nop()
	srv := NewServer(ServerConfig{
		Host:           "127.0.0.1",
		Port:           0,
		RequestTimeout: 10 * time.Second,
		Version:        "test",
	}, handler, logger)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	w := httptest.NewRecorder()
	srv.router.ServeHTTP(w, req)

	rid := w.Header().Get("X-Request-ID")
	if rid == "" {
		t.Error("expected X-Request-ID in response header")
	}
}

// Test that custom X-Request-ID from client is preserved through full router.
func TestServer_CustomRequestIDPreserved(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	logger := zerolog.Nop()
	srv := NewServer(ServerConfig{
		Host:           "127.0.0.1",
		Port:           0,
		RequestTimeout: 10 * time.Second,
		Version:        "test",
	}, handler, logger)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req.Header.Set("X-Request-ID", "custom-req-id-123")
	w := httptest.NewRecorder()
	srv.router.ServeHTTP(w, req)

	rid := w.Header().Get("X-Request-ID")
	if rid != "custom-req-id-123" {
		t.Errorf("expected custom request ID to be preserved, got %q", rid)
	}
}

// Test CreateTask with secret_data.
func TestCreateTask_WithSecretData(t *testing.T) {
	var capturedReq agent.ExecuteRequest
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	h.agent = &capturingExecutor{
		result:   &agent.AgentResult{Status: "completed", Result: "ok"},
		captured: &capturedReq,
	}

	body := `{"prompt":"write secret","secret_data":{"password":"s3cret","api_key":"key123"}}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if capturedReq.SecretData == nil {
		t.Fatal("expected secret_data to be captured")
	}
	if capturedReq.SecretData["password"] != "s3cret" {
		t.Errorf("expected password=s3cret, got %v", capturedReq.SecretData["password"])
	}
}

// Test CreateTask with empty body (not even valid JSON).
func TestCreateTask_EmptyBody(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(""))
	req.Header.Set("Content-Type", "application/json")
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty body, got %d", w.Code)
	}
}

// Test response with data field.
func TestCreateTask_ResponseWithData(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{
			result: &agent.AgentResult{
				Status:     "completed",
				Result:     "Found secret",
				ModelUsed:  "gpt-4o",
				DurationMS: 500,
				Data:       []map[string]any{{"key": "value", "nested": map[string]any{"a": "b"}}},
			},
		},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	body := `{"prompt":"read secret"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp TaskResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 data item, got %d", len(resp.Data))
	}
	if resp.Data[0]["key"] != "value" {
		t.Errorf("expected data[0].key=value, got %v", resp.Data[0]["key"])
	}
}

// Test response with error field.
func TestCreateTask_ResponseWithError(t *testing.T) {
	errMsg := "rate limited"
	h := newTestHandler(
		&mockAgentExecutor{
			result: &agent.AgentResult{
				Status:    "error",
				ErrorCode: "llm_rate_limit",
				Error:     &errMsg,
			},
		},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	body := `{"prompt":"test"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusTooManyRequests {
		t.Errorf("expected 429, got %d", w.Code)
	}

	var resp TaskResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Status != "error" {
		t.Errorf("expected status=error, got %q", resp.Status)
	}
	if resp.Error == nil || *resp.Error != "rate limited" {
		t.Errorf("expected error='rate limited', got %v", resp.Error)
	}
}

// Test writeError with nil detail.
func TestWriteError_NilDetail(t *testing.T) {
	w := httptest.NewRecorder()
	writeError(w, http.StatusNotFound, "not found", nil)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}

	var resp ErrorResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Error != "not found" {
		t.Errorf("expected error='not found', got %q", resp.Error)
	}
	if resp.Detail != nil {
		t.Errorf("expected nil detail, got %v", resp.Detail)
	}
}

// Test recovery middleware with a handler that panics with a non-string value.
func TestRecoveryMiddleware_PanicWithNonString(t *testing.T) {
	handler := RecoveryMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic(42)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	ctx := logging.NewContext(req.Context(), zerolog.Nop())
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d", w.Code)
	}
}

// Test that LoggingMiddleware captures status codes from errors.
func TestLoggingMiddleware_ErrorStatus(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		w.Write([]byte("bad gateway"))
	})
	handler := LoggingMiddleware(inner)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", nil)
	ctx := logging.NewContext(req.Context(), zerolog.Nop())
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadGateway {
		t.Errorf("expected 502, got %d", w.Code)
	}
}

// Test statusWriter without explicit WriteHeader call.
func TestStatusWriter_DefaultStatus(t *testing.T) {
	w := httptest.NewRecorder()
	sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}

	// Write without calling WriteHeader — should keep default 200.
	sw.Write([]byte("hello"))

	if sw.status != http.StatusOK {
		t.Errorf("expected default status 200, got %d", sw.status)
	}
	if sw.written != 5 {
		t.Errorf("expected 5 bytes, got %d", sw.written)
	}
}

// Test Addr returns nil (as documented).
func TestServer_Addr(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	logger := zerolog.Nop()
	srv := NewServer(ServerConfig{
		Host: "127.0.0.1",
		Port: 0,
	}, handler, logger)

	if srv.Addr() != nil {
		t.Error("expected nil Addr before server starts")
	}
}

// Test CreateTask with max_iterations=1 (boundary valid).
func TestCreateTask_ValidMaxIterations(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{
			result: &agent.AgentResult{Status: "completed", Result: "ok"},
		},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	val := 1
	body, _ := json.Marshal(TaskRequest{Prompt: "test", MaxIterations: &val})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", bytes.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200 for max_iterations=1, got %d", w.Code)
	}
}

// Test CreateTask with max_iterations=100 (boundary valid).
func TestCreateTask_ValidMaxIterations100(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{
			result: &agent.AgentResult{Status: "completed", Result: "ok"},
		},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	val := 100
	body, _ := json.Marshal(TaskRequest{Prompt: "test", MaxIterations: &val})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", bytes.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200 for max_iterations=100, got %d", w.Code)
	}
}

// Test prompt at exactly 4096 chars (boundary valid).
func TestCreateTask_PromptExactly4096(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{
			result: &agent.AgentResult{Status: "completed", Result: "ok"},
		},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	prompt := strings.Repeat("a", 4096)
	body, _ := json.Marshal(TaskRequest{Prompt: prompt})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", bytes.NewReader(body))
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.CreateTask(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200 for 4096-char prompt, got %d", w.Code)
	}
}

// Test NewServer with zero timeout (no timeout middleware).
func TestNewServer_NoTimeout(t *testing.T) {
	handler := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)
	logger := zerolog.Nop()

	// RequestTimeout=0 means no timeout middleware.
	srv := NewServer(ServerConfig{
		Host:           "127.0.0.1",
		Port:           0,
		RequestTimeout: 0,
		Version:        "test",
	}, handler, logger)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	w := httptest.NewRecorder()
	srv.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

// Test NewHandler creates handler with correct version.
func TestNewHandler_Fields(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{},
	)

	if h.version != "test-1.0.0" {
		t.Errorf("expected version=test-1.0.0, got %q", h.version)
	}
}

// Test injectLoggerMiddleware.
func TestInjectLoggerMiddleware(t *testing.T) {
	logger := zerolog.Nop()
	mw := injectLoggerMiddleware(logger)

	var gotLogger bool
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		l := logging.FromContext(r.Context())
		// If we got a logger (not nop from empty context), it was injected.
		l.Info().Msg("test")
		gotLogger = true
	})

	handler := mw(inner)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if !gotLogger {
		t.Error("expected logger to be injected into context")
	}
}

// Test Models with nil models (lister returns nil).
func TestModels_NilModels(t *testing.T) {
	h := newTestHandler(
		&mockAgentExecutor{},
		&mockMCPHealthChecker{connected: true},
		&mockModelLister{models: nil},
	)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/models", nil)
	req = injectLogger(req)
	w := httptest.NewRecorder()

	h.Models(w, req)

	var resp ModelsResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.AvailableModels == nil {
		// Should be empty array, not nil.
		if len(resp.AvailableModels) != 0 {
			t.Errorf("expected empty models, got %d", len(resp.AvailableModels))
		}
	}
}
