package config

import (
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/viper"
)

// DefaultConfigPath is the default path to the YAML configuration file.
const DefaultConfigPath = "config/default.yaml"

// LoadConfig loads configuration from a YAML file and environment variables.
// Environment variables take precedence over YAML values.
// configPath may be empty to use the default path.
//
// Due to a known viper limitation, AutomaticEnv() can shadow entire config
// sections when a short env var name collides with a top-level YAML key
// (e.g. AGENT=1 shadows the "agent:" section). To work around this, we
// unmarshal from the YAML+defaults first (without AutomaticEnv), then
// overlay explicit env var bindings on top.
func LoadConfig(configPath string) (*Settings, error) {
	v := viper.New()

	setDefaults(v)

	// Load YAML config file.
	if configPath == "" {
		configPath = DefaultConfigPath
	}
	v.SetConfigFile(configPath)
	if err := v.ReadInConfig(); err != nil {
		// Config file is optional — proceed with defaults + env vars.
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			if !os.IsNotExist(err) {
				return nil, fmt.Errorf("reading config file %s: %w", configPath, err)
			}
		}
	}

	// Step 1: Unmarshal from YAML + defaults BEFORE enabling AutomaticEnv.
	// This ensures nested sections are read correctly without env var collision.
	var cfg Settings
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshalling config: %w", err)
	}

	// Step 2: Now bind explicit env vars and apply overrides.
	// We do NOT use AutomaticEnv() because env vars like AGENT=1 can shadow
	// entire YAML sections. Instead, all env vars are bound explicitly via
	// bindFlatAliases, which is safe and predictable.
	bindFlatAliases(v)
	applyEnvOverrides(v, &cfg)

	if err := Validate(&cfg); err != nil {
		return nil, fmt.Errorf("config validation: %w", err)
	}

	return &cfg, nil
}

// setDefaults registers default values matching the Python version.
func setDefaults(v *viper.Viper) {
	// API defaults.
	v.SetDefault("api.host", "0.0.0.0")
	v.SetDefault("api.port", 8000)
	v.SetDefault("api.request_timeout", 120)

	// mTLS defaults.
	v.SetDefault("mtls.enabled", true)
	v.SetDefault("mtls.ca_cert_path", "/certs/ca.pem")
	v.SetDefault("mtls.server_cert_path", "/certs/server.pem")
	v.SetDefault("mtls.server_key_path", "/certs/server-key.pem")

	// MCP defaults.
	v.SetDefault("mcp.transport", "stdio")
	v.SetDefault("mcp.server_binary", "/usr/local/bin/vault-mcp-server")
	v.SetDefault("mcp.server_url", "http://vault-mcp-server:3000")
	v.SetDefault("mcp.vault_addr", "http://vault:8200")
	v.SetDefault("mcp.vault_token", "")
	v.SetDefault("mcp.tool_timeout", 30)
	v.SetDefault("mcp.reconnect_initial_delay", 1.0)
	v.SetDefault("mcp.reconnect_max_delay", 60.0)

	// LLM defaults.
	v.SetDefault("llm.base_url", "https://models.inference.ai.azure.com")
	v.SetDefault("llm.default_model", "default")
	v.SetDefault("llm.request_timeout", 60)
	v.SetDefault("llm.max_retries", 3)

	// Agent defaults.
	v.SetDefault("agent.max_iterations", 10)
	v.SetDefault("agent.system_prompt_path", "config/prompts/system.md")

	// Scheduler defaults.
	v.SetDefault("scheduler.enabled", true)

	// Logging defaults.
	v.SetDefault("logging.level", "INFO")
	v.SetDefault("logging.format", "json")
}

// bindFlatAliases binds flat env var names to nested config paths.
// These mirror the Python version's convenience aliases.
func bindFlatAliases(v *viper.Viper) {
	_ = v.BindEnv("llm.api_key", "GITHUB_TOKEN")
	_ = v.BindEnv("mcp.vault_addr", "VAULT_ADDR")
	_ = v.BindEnv("mcp.vault_token", "VAULT_TOKEN")
	_ = v.BindEnv("mtls.enabled", "MTLS_ENABLED")
	_ = v.BindEnv("logging.level", "LOG_LEVEL")
	_ = v.BindEnv("logging.format", "LOG_FORMAT")
	_ = v.BindEnv("mcp.transport", "MCP_TRANSPORT")
	_ = v.BindEnv("llm.default_model", "DEFAULT_MODEL")

	// Also support the nested double-underscore variants explicitly.
	_ = v.BindEnv("api.host", "SERVER__HOST", "API__HOST")
	_ = v.BindEnv("api.port", "SERVER__PORT", "API__PORT")
	_ = v.BindEnv("api.request_timeout", "SERVER__REQUEST_TIMEOUT", "API__REQUEST_TIMEOUT")
	_ = v.BindEnv("llm.default_model", "LLM__DEFAULT_MODEL", "DEFAULT_MODEL")
	_ = v.BindEnv("llm.max_retries", "LLM__MAX_RETRIES")
	_ = v.BindEnv("llm.request_timeout", "LLM__REQUEST_TIMEOUT")
	_ = v.BindEnv("mcp.vault_addr", "MCP__VAULT_ADDR", "VAULT_ADDR")
	_ = v.BindEnv("mcp.vault_token", "MCP__VAULT_TOKEN", "VAULT_TOKEN")
	_ = v.BindEnv("mcp.tool_timeout", "MCP__TOOL_TIMEOUT")
	_ = v.BindEnv("agent.max_iterations", "AGENT__MAX_ITERATIONS")
	_ = v.BindEnv("scheduler.enabled", "SCHEDULER__ENABLED")
	_ = v.BindEnv("logging.level", "LOGGING__LEVEL", "LOG_LEVEL")
	_ = v.BindEnv("logging.format", "LOGGING__FORMAT", "LOG_FORMAT")
}

// applyEnvOverrides applies environment variable overrides on top of the
// already-unmarshalled config. This replaces AutomaticEnv() with explicit
// env var reads, avoiding the viper bug where short env var names (e.g.
// AGENT=1) shadow entire YAML sections.
//
// Each env var binding from bindFlatAliases is resolved here. If an env var
// is set, its value overrides the corresponding config field.
func applyEnvOverrides(v *viper.Viper, cfg *Settings) {
	// GITHUB_TOKEN → llm.api_key
	if val := v.GetString("llm.api_key"); val != "" {
		cfg.LLM.APIKey = val
	}

	// Ensure MCP vault fields from flat aliases are applied.
	if val := v.GetString("mcp.vault_addr"); val != "" {
		cfg.MCP.VaultAddr = val
	}
	if val := v.GetString("mcp.vault_token"); val != "" {
		cfg.MCP.VaultToken = val
	}

	// Direct env var overrides for fields that have explicit bindings.
	applyEnvString("LOG_LEVEL", &cfg.Logging.Level)
	applyEnvString("LOGGING__LEVEL", &cfg.Logging.Level)
	applyEnvString("LOG_FORMAT", &cfg.Logging.Format)
	applyEnvString("LOGGING__FORMAT", &cfg.Logging.Format)
	applyEnvString("MTLS_ENABLED", nil) // handled via viper binding
	applyEnvString("MCP_TRANSPORT", &cfg.MCP.Transport)
	applyEnvString("DEFAULT_MODEL", &cfg.LLM.DefaultModel)
	applyEnvString("LLM__DEFAULT_MODEL", &cfg.LLM.DefaultModel)

	applyEnvString("SERVER__HOST", &cfg.API.Host)
	applyEnvString("API__HOST", &cfg.API.Host)
	applyEnvInt("SERVER__PORT", &cfg.API.Port)
	applyEnvInt("API__PORT", &cfg.API.Port)
	applyEnvInt("SERVER__REQUEST_TIMEOUT", &cfg.API.RequestTimeout)
	applyEnvInt("API__REQUEST_TIMEOUT", &cfg.API.RequestTimeout)
	applyEnvInt("LLM__MAX_RETRIES", &cfg.LLM.MaxRetries)
	applyEnvInt("LLM__REQUEST_TIMEOUT", &cfg.LLM.RequestTimeout)
	applyEnvInt("MCP__TOOL_TIMEOUT", &cfg.MCP.ToolTimeout)
	applyEnvInt("AGENT__MAX_ITERATIONS", &cfg.Agent.MaxIterations)

	// MTLS_ENABLED needs special handling (bool).
	if val, ok := os.LookupEnv("MTLS_ENABLED"); ok {
		cfg.MTLS.Enabled = val == "true" || val == "1" || val == "yes"
	}
	if val, ok := os.LookupEnv("SCHEDULER__ENABLED"); ok {
		cfg.Scheduler.Enabled = val == "true" || val == "1" || val == "yes"
	}
}

// applyEnvString sets *dst to the env var value if it is set and non-empty.
func applyEnvString(envVar string, dst *string) {
	if dst == nil {
		return
	}
	if val, ok := os.LookupEnv(envVar); ok && val != "" {
		*dst = val
	}
}

// applyEnvInt sets *dst to the env var value parsed as int, if set and valid.
func applyEnvInt(envVar string, dst *int) {
	if val, ok := os.LookupEnv(envVar); ok && val != "" {
		if n, err := strconv.Atoi(val); err == nil {
			*dst = n
		}
	}
}

// Validate checks the configuration for required fields and valid ranges.
func Validate(cfg *Settings) error {
	// Required fields.
	if cfg.LLM.APIKey == "" {
		return fmt.Errorf("required configuration 'GITHUB_TOKEN' is missing")
	}

	// Port range.
	if cfg.API.Port < 1 || cfg.API.Port > 65535 {
		return fmt.Errorf("api.port must be between 1 and 65535, got %d", cfg.API.Port)
	}

	// Timeouts.
	if cfg.API.RequestTimeout < 1 {
		return fmt.Errorf("api.request_timeout must be >= 1, got %d", cfg.API.RequestTimeout)
	}
	if cfg.MCP.ToolTimeout < 1 {
		return fmt.Errorf("mcp.tool_timeout must be >= 1, got %d", cfg.MCP.ToolTimeout)
	}
	if cfg.LLM.RequestTimeout < 1 {
		return fmt.Errorf("llm.request_timeout must be >= 1, got %d", cfg.LLM.RequestTimeout)
	}

	// Agent.
	if cfg.Agent.MaxIterations < 1 || cfg.Agent.MaxIterations > 100 {
		return fmt.Errorf("agent.max_iterations must be between 1 and 100, got %d", cfg.Agent.MaxIterations)
	}

	// At least one model must be configured.
	if len(cfg.LLM.Models) == 0 {
		return fmt.Errorf("at least one LLM model must be configured")
	}
	for i, m := range cfg.LLM.Models {
		if m.ModelID == "" {
			return fmt.Errorf("llm.models[%d].model_id must not be empty", i)
		}
	}

	return nil
}
