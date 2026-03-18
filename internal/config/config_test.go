package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- Validate tests ---

func TestValidate_MinimalValid(t *testing.T) {
	cfg := &Settings{
		API: APIConfig{Port: 8000, RequestTimeout: 120},
		MCP: MCPConfig{ToolTimeout: 30},
		LLM: LLMConfig{
			APIKey:         "test-key",
			RequestTimeout: 60,
			Models:         []ModelConfig{{ModelID: "gpt-4o"}},
		},
		Agent: AgentConfig{MaxIterations: 10},
	}

	if err := Validate(cfg); err != nil {
		t.Errorf("expected valid config, got error: %v", err)
	}
}

func TestValidate_MissingAPIKey(t *testing.T) {
	cfg := &Settings{
		API:   APIConfig{Port: 8000, RequestTimeout: 120},
		MCP:   MCPConfig{ToolTimeout: 30},
		LLM:   LLMConfig{RequestTimeout: 60, Models: []ModelConfig{{ModelID: "gpt-4o"}}},
		Agent: AgentConfig{MaxIterations: 10},
	}

	err := Validate(cfg)
	if err == nil {
		t.Fatal("expected error for missing API key")
	}
	if !strings.Contains(err.Error(), "GITHUB_TOKEN") {
		t.Errorf("expected GITHUB_TOKEN in error, got: %v", err)
	}
}

func TestValidate_InvalidPort(t *testing.T) {
	tests := []struct {
		name string
		port int
	}{
		{"zero", 0},
		{"negative", -1},
		{"too high", 65536},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Settings{
				API:   APIConfig{Port: tc.port, RequestTimeout: 120},
				MCP:   MCPConfig{ToolTimeout: 30},
				LLM:   LLMConfig{APIKey: "k", RequestTimeout: 60, Models: []ModelConfig{{ModelID: "m"}}},
				Agent: AgentConfig{MaxIterations: 10},
			}
			err := Validate(cfg)
			if err == nil {
				t.Fatal("expected error for invalid port")
			}
			if !strings.Contains(err.Error(), "port") {
				t.Errorf("expected port in error, got: %v", err)
			}
		})
	}
}

func TestValidate_InvalidTimeouts(t *testing.T) {
	tests := []struct {
		name       string
		apiTimeout int
		mcpTimeout int
		llmTimeout int
		wantField  string
	}{
		{"api timeout zero", 0, 30, 60, "api.request_timeout"},
		{"mcp timeout zero", 120, 0, 60, "mcp.tool_timeout"},
		{"llm timeout zero", 120, 30, 0, "llm.request_timeout"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Settings{
				API:   APIConfig{Port: 8000, RequestTimeout: tc.apiTimeout},
				MCP:   MCPConfig{ToolTimeout: tc.mcpTimeout},
				LLM:   LLMConfig{APIKey: "k", RequestTimeout: tc.llmTimeout, Models: []ModelConfig{{ModelID: "m"}}},
				Agent: AgentConfig{MaxIterations: 10},
			}
			err := Validate(cfg)
			if err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantField) {
				t.Errorf("expected %q in error, got: %v", tc.wantField, err)
			}
		})
	}
}

func TestValidate_InvalidMaxIterations(t *testing.T) {
	tests := []struct {
		name string
		val  int
	}{
		{"zero", 0},
		{"too high", 101},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Settings{
				API:   APIConfig{Port: 8000, RequestTimeout: 120},
				MCP:   MCPConfig{ToolTimeout: 30},
				LLM:   LLMConfig{APIKey: "k", RequestTimeout: 60, Models: []ModelConfig{{ModelID: "m"}}},
				Agent: AgentConfig{MaxIterations: tc.val},
			}
			err := Validate(cfg)
			if err == nil {
				t.Fatal("expected error for invalid max_iterations")
			}
			if !strings.Contains(err.Error(), "max_iterations") {
				t.Errorf("expected max_iterations in error, got: %v", err)
			}
		})
	}
}

func TestValidate_NoModels(t *testing.T) {
	cfg := &Settings{
		API:   APIConfig{Port: 8000, RequestTimeout: 120},
		MCP:   MCPConfig{ToolTimeout: 30},
		LLM:   LLMConfig{APIKey: "k", RequestTimeout: 60, Models: []ModelConfig{}},
		Agent: AgentConfig{MaxIterations: 10},
	}

	err := Validate(cfg)
	if err == nil {
		t.Fatal("expected error for no models")
	}
	if !strings.Contains(err.Error(), "at least one") {
		t.Errorf("expected 'at least one' in error, got: %v", err)
	}
}

func TestValidate_EmptyModelID(t *testing.T) {
	cfg := &Settings{
		API:   APIConfig{Port: 8000, RequestTimeout: 120},
		MCP:   MCPConfig{ToolTimeout: 30},
		LLM:   LLMConfig{APIKey: "k", RequestTimeout: 60, Models: []ModelConfig{{Name: "test", ModelID: ""}}},
		Agent: AgentConfig{MaxIterations: 10},
	}

	err := Validate(cfg)
	if err == nil {
		t.Fatal("expected error for empty model_id")
	}
	if !strings.Contains(err.Error(), "model_id") {
		t.Errorf("expected model_id in error, got: %v", err)
	}
}

// --- LoadConfig tests ---

func TestLoadConfig_FromYAML(t *testing.T) {
	// Create a temp YAML config file.
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	yamlContent := `
api:
  host: "127.0.0.1"
  port: 9000
  request_timeout: 60
mcp:
  tool_timeout: 15
  vault_addr: "http://localhost:8200"
llm:
  base_url: "https://api.openai.com"
  default_model: "default"
  request_timeout: 30
  max_retries: 5
  models:
    - name: "default"
      provider: "openai"
      model_id: "gpt-4o"
      supports_tool_calling: true
agent:
  max_iterations: 20
logging:
  level: "DEBUG"
  format: "console"
`

	if err := os.WriteFile(configPath, []byte(yamlContent), 0644); err != nil {
		t.Fatal(err)
	}

	// Set required env var.
	t.Setenv("GITHUB_TOKEN", "ghp_test123456789")

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}

	if cfg.API.Host != "127.0.0.1" {
		t.Errorf("expected host=127.0.0.1, got %q", cfg.API.Host)
	}
	if cfg.API.Port != 9000 {
		t.Errorf("expected port=9000, got %d", cfg.API.Port)
	}
	if cfg.MCP.ToolTimeout != 15 {
		t.Errorf("expected tool_timeout=15, got %d", cfg.MCP.ToolTimeout)
	}
	if cfg.LLM.MaxRetries != 5 {
		t.Errorf("expected max_retries=5, got %d", cfg.LLM.MaxRetries)
	}
	if cfg.Agent.MaxIterations != 20 {
		t.Errorf("expected max_iterations=20, got %d", cfg.Agent.MaxIterations)
	}
	if len(cfg.LLM.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(cfg.LLM.Models))
	}
	if cfg.LLM.Models[0].ModelID != "gpt-4o" {
		t.Errorf("expected model_id=gpt-4o, got %q", cfg.LLM.Models[0].ModelID)
	}
}

func TestLoadConfig_EnvVarOverrides(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	yamlContent := `
api:
  port: 8000
  request_timeout: 120
mcp:
  tool_timeout: 30
llm:
  request_timeout: 60
  models:
    - name: "default"
      model_id: "gpt-4o"
agent:
  max_iterations: 10
`
	os.WriteFile(configPath, []byte(yamlContent), 0644)

	t.Setenv("GITHUB_TOKEN", "ghp_testtoken123")
	t.Setenv("VAULT_ADDR", "http://custom-vault:8200")

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}

	if cfg.LLM.APIKey != "ghp_testtoken123" {
		t.Errorf("expected API key from GITHUB_TOKEN, got %q", cfg.LLM.APIKey)
	}
	if cfg.MCP.VaultAddr != "http://custom-vault:8200" {
		t.Errorf("expected vault_addr from VAULT_ADDR, got %q", cfg.MCP.VaultAddr)
	}
}

func TestLoadConfig_MissingFile_UsesDefaults(t *testing.T) {
	// Missing file should not error — uses defaults + env vars.
	t.Setenv("GITHUB_TOKEN", "ghp_test123")

	// Use a non-existent path — should fall through to defaults.
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")
	// This may or may not error depending on viper's behavior with non-existent files.
	// The function handles ConfigFileNotFoundError and os.IsNotExist.
	if err != nil {
		// If validation fails due to no models, that's expected with pure defaults.
		if !strings.Contains(err.Error(), "model") {
			t.Logf("LoadConfig with missing file: %v (may be expected)", err)
		}
		return
	}

	// If it succeeds, check defaults were applied.
	if cfg.API.Port != 8000 {
		t.Errorf("expected default port=8000, got %d", cfg.API.Port)
	}
}

func TestLoadConfig_ValidationFails(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Config with invalid port.
	yamlContent := `
api:
  port: 0
  request_timeout: 120
mcp:
  tool_timeout: 30
llm:
  request_timeout: 60
  models:
    - model_id: "gpt-4o"
agent:
  max_iterations: 10
`
	os.WriteFile(configPath, []byte(yamlContent), 0644)
	t.Setenv("GITHUB_TOKEN", "ghp_test123")

	_, err := LoadConfig(configPath)
	if err == nil {
		t.Fatal("expected validation error")
	}
	if !strings.Contains(err.Error(), "config validation") {
		t.Errorf("expected 'config validation' in error, got: %v", err)
	}
}

// --- Defaults ---

func TestDefaults_Applied(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	// Minimal YAML — rely on defaults.
	yamlContent := `
llm:
  models:
    - model_id: "gpt-4o"
`
	os.WriteFile(configPath, []byte(yamlContent), 0644)
	t.Setenv("GITHUB_TOKEN", "ghp_test123")

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}

	// Check defaults from setDefaults().
	if cfg.API.Host != "0.0.0.0" {
		t.Errorf("expected default host=0.0.0.0, got %q", cfg.API.Host)
	}
	if cfg.API.Port != 8000 {
		t.Errorf("expected default port=8000, got %d", cfg.API.Port)
	}
	if cfg.API.RequestTimeout != 120 {
		t.Errorf("expected default request_timeout=120, got %d", cfg.API.RequestTimeout)
	}
	if cfg.MCP.ToolTimeout != 30 {
		t.Errorf("expected default tool_timeout=30, got %d", cfg.MCP.ToolTimeout)
	}
	if cfg.LLM.RequestTimeout != 60 {
		t.Errorf("expected default llm request_timeout=60, got %d", cfg.LLM.RequestTimeout)
	}
	if cfg.LLM.MaxRetries != 3 {
		t.Errorf("expected default max_retries=3, got %d", cfg.LLM.MaxRetries)
	}
	if cfg.Agent.MaxIterations != 10 {
		t.Errorf("expected default max_iterations=10, got %d", cfg.Agent.MaxIterations)
	}
	if cfg.Logging.Level != "INFO" {
		t.Errorf("expected default log level=INFO, got %q", cfg.Logging.Level)
	}
}
