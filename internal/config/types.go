// Package config defines configuration types for vault-agent-operator.
package config

// Settings is the top-level configuration struct.
type Settings struct {
	API       APIConfig       `mapstructure:"api"       json:"api"`
	MTLS      MTLSConfig      `mapstructure:"mtls"      json:"mtls"`
	MCP       MCPConfig       `mapstructure:"mcp"       json:"mcp"`
	LLM       LLMConfig       `mapstructure:"llm"       json:"llm"`
	Agent     AgentConfig     `mapstructure:"agent"     json:"agent"`
	Scheduler SchedulerConfig `mapstructure:"scheduler" json:"scheduler"`
	Logging   LoggingConfig   `mapstructure:"logging"   json:"logging"`
}

// APIConfig holds HTTP server configuration.
type APIConfig struct {
	Host           string `mapstructure:"host"            json:"host"`
	Port           int    `mapstructure:"port"            json:"port"`
	RequestTimeout int    `mapstructure:"request_timeout" json:"request_timeout"`
}

// MTLSConfig holds mutual TLS configuration.
type MTLSConfig struct {
	Enabled        bool   `mapstructure:"enabled"          json:"enabled"`
	CACertPath     string `mapstructure:"ca_cert_path"     json:"ca_cert_path"`
	ServerCertPath string `mapstructure:"server_cert_path" json:"server_cert_path"`
	ServerKeyPath  string `mapstructure:"server_key_path"  json:"server_key_path"`
}

// MCPConfig holds MCP client configuration.
type MCPConfig struct {
	Transport            string  `mapstructure:"transport"              json:"transport"`
	ServerBinary         string  `mapstructure:"server_binary"          json:"server_binary"`
	ServerURL            string  `mapstructure:"server_url"             json:"server_url"`
	VaultAddr            string  `mapstructure:"vault_addr"             json:"vault_addr"`
	VaultToken           string  `mapstructure:"vault_token"            json:"vault_token"`
	ToolTimeout          int     `mapstructure:"tool_timeout"           json:"tool_timeout"`
	ReconnectInitialDely float64 `mapstructure:"reconnect_initial_delay" json:"reconnect_initial_delay"`
	ReconnectMaxDelay    float64 `mapstructure:"reconnect_max_delay"    json:"reconnect_max_delay"`
}

// LLMConfig holds LLM provider configuration.
type LLMConfig struct {
	APIKey         string        `mapstructure:"api_key"         json:"-"`
	BaseURL        string        `mapstructure:"base_url"        json:"base_url"`
	DefaultModel   string        `mapstructure:"default_model"   json:"default_model"`
	RequestTimeout int           `mapstructure:"request_timeout" json:"request_timeout"`
	MaxRetries     int           `mapstructure:"max_retries"     json:"max_retries"`
	Models         []ModelConfig `mapstructure:"models"          json:"models"`
}

// ModelConfig describes a single LLM model.
type ModelConfig struct {
	Name                string `mapstructure:"name"                  json:"name"`
	Provider            string `mapstructure:"provider"              json:"provider"`
	ModelID             string `mapstructure:"model_id"              json:"model_id"`
	SupportsToolCalling bool   `mapstructure:"supports_tool_calling" json:"supports_tool_calling"`
}

// AgentConfig holds agent reasoning-loop configuration.
type AgentConfig struct {
	MaxIterations    int    `mapstructure:"max_iterations"     json:"max_iterations"`
	SystemPromptPath string `mapstructure:"system_prompt_path" json:"system_prompt_path"`
}

// SchedulerConfig holds scheduler configuration.
type SchedulerConfig struct {
	Enabled bool               `mapstructure:"enabled" json:"enabled"`
	Tasks   []ScheduledTaskDef `mapstructure:"tasks"   json:"tasks"`
}

// ScheduledTaskDef defines a single scheduled task in config.
type ScheduledTaskDef struct {
	ID      string `mapstructure:"id"      json:"id"`
	Cron    string `mapstructure:"cron"    json:"cron"`
	Prompt  string `mapstructure:"prompt"  json:"prompt"`
	Enabled bool   `mapstructure:"enabled" json:"enabled"`
}

// LoggingConfig holds logging configuration.
type LoggingConfig struct {
	Level          string   `mapstructure:"level"           json:"level"`
	Format         string   `mapstructure:"format"          json:"format"`
	RedactPatterns []string `mapstructure:"redact_patterns" json:"redact_patterns"`
}
