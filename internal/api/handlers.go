package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/agent"
	"github.com/jarlex/vault-agent-operator/internal/config"
	"github.com/jarlex/vault-agent-operator/internal/llm"
	"github.com/jarlex/vault-agent-operator/internal/logging"
)

// Handler holds the HTTP route handlers and their dependencies.
type Handler struct {
	agent     AgentExecutor
	mcp       MCPHealthChecker
	llm       ModelLister
	llmConfig config.LLMConfig
	logger    zerolog.Logger
	startTime time.Time
	version   string
}

// AgentExecutor runs agent tasks. Satisfied by *agent.AgentCore.
type AgentExecutor interface {
	Execute(ctx context.Context, req agent.ExecuteRequest) (*agent.AgentResult, error)
}

// MCPHealthChecker checks MCP connectivity. Satisfied by mcp.MCPClient.
type MCPHealthChecker interface {
	HealthCheck(ctx context.Context) error
	IsConnected() bool
}

// ModelLister lists available models. Satisfied by llm.LLMProvider.
type ModelLister interface {
	AvailableModels() []llm.ModelInfo
}

// HandlerConfig contains configuration for the handler.
type HandlerConfig struct {
	LLMConfig config.LLMConfig
	Version   string
}

// NewHandler creates a new Handler with all dependencies.
func NewHandler(
	agentExec AgentExecutor,
	mcpClient MCPHealthChecker,
	llmProvider ModelLister,
	cfg HandlerConfig,
	logger zerolog.Logger,
) *Handler {
	return &Handler{
		agent:     agentExec,
		mcp:       mcpClient,
		llm:       llmProvider,
		llmConfig: cfg.LLMConfig,
		logger:    logger.With().Str("component", "api").Logger(),
		startTime: time.Now(),
		version:   cfg.Version,
	}
}

// CreateTask handles POST /api/v1/tasks.
func (h *Handler) CreateTask(w http.ResponseWriter, r *http.Request) {
	logger := logging.FromContext(r.Context())

	// 1. Decode request body.
	var req TaskRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		logger.Warn().Err(err).Msg("api.create_task.invalid_json")
		writeError(w, http.StatusBadRequest, "Invalid JSON request body", nil)
		return
	}

	// 2. Validate.
	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt is required", nil)
		return
	}
	if len(req.Prompt) > 4096 {
		writeError(w, http.StatusBadRequest,
			"prompt exceeds maximum length of 4096 characters", nil)
		return
	}
	if req.MaxIterations != nil && (*req.MaxIterations < 1 || *req.MaxIterations > 100) {
		writeError(w, http.StatusBadRequest,
			"max_iterations must be between 1 and 100", nil)
		return
	}

	// 3. Build execute request.
	execReq := agent.ExecuteRequest{
		Prompt:     req.Prompt,
		SecretData: req.SecretData,
	}
	if req.Model != nil {
		execReq.Model = *req.Model
	}

	// 4. Execute agent task.
	result, err := h.agent.Execute(r.Context(), execReq)
	if err != nil {
		logger.Error().Err(err).Msg("api.create_task.execute_error")
		writeError(w, http.StatusInternalServerError,
			"Internal server error", nil)
		return
	}

	// 5. Map AgentResult to TaskResponse.
	resp := TaskResponse{
		Status:     result.Status,
		Result:     result.Result,
		Data:       result.Data,
		ModelUsed:  result.ModelUsed,
		DurationMS: result.DurationMS,
		Error:      result.Error,
	}

	// 6. Determine HTTP status code from error code.
	status := mapErrorCodeToStatus(result.ErrorCode)

	logger.Info().
		Str("status", result.Status).
		Int("http_status", status).
		Int64("duration_ms", result.DurationMS).
		Msg("api.create_task.complete")

	writeJSON(w, status, resp)
}

// Health handles GET /api/v1/health.
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	overallStatus := "healthy"
	mcpStatus := "connected"
	vaultStatus := "unknown"

	if !h.mcp.IsConnected() {
		overallStatus = "unhealthy"
		mcpStatus = "disconnected"
		vaultStatus = "unreachable"
	} else {
		err := h.mcp.HealthCheck(r.Context())
		if err != nil {
			overallStatus = "degraded"
			mcpStatus = "connected"
			vaultStatus = "error"
		} else {
			vaultStatus = "ok"
		}
	}

	resp := HealthResponse{
		Status:        overallStatus,
		Agent:         "running",
		VaultMCP:      mcpStatus,
		VaultServer:   vaultStatus,
		UptimeSeconds: time.Since(h.startTime).Seconds(),
		Version:       h.version,
	}

	writeJSON(w, http.StatusOK, resp)
}

// Models handles GET /api/v1/models.
func (h *Handler) Models(w http.ResponseWriter, r *http.Request) {
	models := h.llm.AvailableModels()

	details := make([]ModelDetail, 0, len(models))
	for _, m := range models {
		details = append(details, ModelDetail{
			Name:                m.Name,
			Provider:            m.Provider,
			ModelID:             m.ModelID,
			SupportsToolCalling: m.SupportsToolCalling,
			IsDefault:           m.Name == h.llmConfig.DefaultModel,
		})
	}

	resp := ModelsResponse{
		DefaultModel:    h.llmConfig.DefaultModel,
		AvailableModels: details,
	}

	writeJSON(w, http.StatusOK, resp)
}

// writeJSON marshals v as JSON and writes it to w with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError writes a structured ErrorResponse JSON to the response.
func writeError(w http.ResponseWriter, status int, msg string, detail *string) {
	resp := ErrorResponse{
		Error:  msg,
		Detail: detail,
	}
	writeJSON(w, status, resp)
}

// mapErrorCodeToStatus maps agent error codes to HTTP status codes.
func mapErrorCodeToStatus(errorCode string) int {
	switch errorCode {
	case "":
		return http.StatusOK
	case "llm_auth":
		return http.StatusBadGateway
	case "llm_rate_limit":
		return http.StatusTooManyRequests
	case "llm_service":
		return http.StatusBadGateway
	case "max_iterations":
		return http.StatusOK
	case "empty_response":
		return http.StatusInternalServerError
	default:
		return http.StatusInternalServerError
	}
}
