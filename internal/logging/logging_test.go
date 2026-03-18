package logging

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/config"
)

// --- SetupLogger ---

func TestSetupLogger_JSONFormat(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "INFO",
		Format: "json",
	}

	logger := SetupLogger(cfg)

	// Verify the logger works by writing to a buffer.
	var buf bytes.Buffer
	testLogger := logger.Output(&buf)
	testLogger.Info().Msg("test message")

	// Should produce valid JSON.
	var logEntry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &logEntry); err != nil {
		t.Fatalf("expected valid JSON output, got error: %v, output: %q", err, buf.String())
	}

	// Should contain the service field.
	if logEntry["service"] != "vault-agent-operator" {
		t.Errorf("expected service=vault-agent-operator, got %v", logEntry["service"])
	}

	// Should contain the message.
	if logEntry["message"] != "test message" {
		t.Errorf("expected message=test message, got %v", logEntry["message"])
	}

	// Should have a timestamp.
	if _, ok := logEntry["time"]; !ok {
		t.Error("expected time field in log entry")
	}
}

func TestSetupLogger_ConsoleFormat(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "DEBUG",
		Format: "console",
	}

	logger := SetupLogger(cfg)

	// Console format produces human-readable output, not JSON.
	var buf bytes.Buffer
	testLogger := logger.Output(&buf)
	testLogger.Debug().Msg("debug msg")

	output := buf.String()
	if output == "" {
		t.Error("expected non-empty output from console logger")
	}
}

func TestSetupLogger_DebugLevel(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "DEBUG",
		Format: "json",
	}

	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)

	testLogger.Debug().Msg("debug message")
	if buf.Len() == 0 {
		t.Error("expected debug message to be logged at DEBUG level")
	}
}

func TestSetupLogger_InfoLevel_FiltersDebug(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "INFO",
		Format: "json",
	}

	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)

	// Debug message should be filtered out at INFO level.
	testLogger.Debug().Msg("should not appear")
	if buf.Len() > 0 {
		t.Error("expected debug message to be filtered at INFO level")
	}

	// Info message should appear.
	testLogger.Info().Msg("should appear")
	if buf.Len() == 0 {
		t.Error("expected info message to be logged at INFO level")
	}
}

func TestSetupLogger_WarnLevel(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "WARN",
		Format: "json",
	}

	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)

	testLogger.Info().Msg("should not appear")
	if buf.Len() > 0 {
		t.Error("expected info message to be filtered at WARN level")
	}

	testLogger.Warn().Msg("warning")
	if buf.Len() == 0 {
		t.Error("expected warn message to be logged at WARN level")
	}
}

func TestSetupLogger_ErrorLevel(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "ERROR",
		Format: "json",
	}

	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)

	testLogger.Warn().Msg("should not appear")
	if buf.Len() > 0 {
		t.Error("expected warn message to be filtered at ERROR level")
	}

	testLogger.Error().Msg("error msg")
	if buf.Len() == 0 {
		t.Error("expected error message to be logged at ERROR level")
	}
}

func TestSetupLogger_DefaultLevelForUnknown(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "UNKNOWN",
		Format: "json",
	}

	// Unknown level should default to INFO.
	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)

	testLogger.Debug().Msg("should not appear")
	if buf.Len() > 0 {
		t.Error("expected debug message to be filtered at default INFO level")
	}

	testLogger.Info().Msg("should appear")
	if buf.Len() == 0 {
		t.Error("expected info message to be logged at default INFO level")
	}
}

func TestSetupLogger_CustomRedactPatterns(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:          "INFO",
		Format:         "json",
		RedactPatterns: []string{"api_key", "secret_token"},
	}

	// Should not panic.
	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)
	testLogger.Info().Msg("test")

	if buf.Len() == 0 {
		t.Error("expected logger to produce output")
	}
}

func TestSetupLogger_EmptyRedactPatterns_UsesDefaults(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:          "INFO",
		Format:         "json",
		RedactPatterns: nil,
	}

	// Should use default patterns without panicking.
	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)
	testLogger.Info().Msg("ok")
	if buf.Len() == 0 {
		t.Error("expected logger to produce output with default patterns")
	}
}

func TestSetupLogger_WarningLevelAlias(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "WARNING",
		Format: "json",
	}

	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)

	testLogger.Info().Msg("should not appear")
	if buf.Len() > 0 {
		t.Error("expected info to be filtered at WARNING level")
	}

	testLogger.Warn().Msg("warning")
	if buf.Len() == 0 {
		t.Error("expected warn to appear at WARNING level")
	}
}

// --- parseLevel ---

func TestParseLevel(t *testing.T) {
	tests := []struct {
		input string
		want  zerolog.Level
	}{
		{"DEBUG", zerolog.DebugLevel},
		{"debug", zerolog.DebugLevel},
		{"INFO", zerolog.InfoLevel},
		{"info", zerolog.InfoLevel},
		{"WARN", zerolog.WarnLevel},
		{"warn", zerolog.WarnLevel},
		{"WARNING", zerolog.WarnLevel},
		{"warning", zerolog.WarnLevel},
		{"ERROR", zerolog.ErrorLevel},
		{"error", zerolog.ErrorLevel},
		{"unknown", zerolog.InfoLevel},
		{"", zerolog.InfoLevel},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got := parseLevel(tc.input)
			if got != tc.want {
				t.Errorf("parseLevel(%q) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

// --- WithRequestID ---

func TestWithRequestID(t *testing.T) {
	var buf bytes.Buffer
	logger := zerolog.New(&buf)

	child := WithRequestID(logger, "req-123")
	child.Info().Msg("test")

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatal(err)
	}

	if entry["request_id"] != "req-123" {
		t.Errorf("expected request_id=req-123, got %v", entry["request_id"])
	}
}

// --- NewContext / FromContext ---

func TestNewContext_FromContext(t *testing.T) {
	logger := zerolog.Nop()
	ctx := NewContext(context.Background(), logger)

	got := FromContext(ctx)
	// Verify we get back a logger (not nil/nop).
	// zerolog.Logger has no equality check, so just verify it works.
	got.Info().Msg("test")
}

func TestFromContext_NoLogger_ReturnsNop(t *testing.T) {
	// When no logger is in context, FromContext returns a Nop logger.
	got := FromContext(context.Background())

	// Nop logger should not panic when used.
	got.Info().Msg("should not crash")
}

func TestFromContext_WrongType_ReturnsNop(t *testing.T) {
	// Put something other than a logger in the key.
	ctx := context.WithValue(context.Background(), loggerKey, "not a logger")
	got := FromContext(ctx)

	// Should return Nop without panicking.
	got.Info().Msg("should not crash")
}

// --- IsSensitiveKey ---

func TestIsSensitiveKey(t *testing.T) {
	patterns := []string{"token", "password", "secret", "key", "authorization"}

	tests := []struct {
		key  string
		want bool
	}{
		{"github_token", true},
		{"GITHUB_TOKEN", true},
		{"password", true},
		{"db_password", true},
		{"secret_value", true},
		{"api_key", true},
		{"Authorization", true},
		{"username", false},
		{"hostname", false},
		{"port", false},
		{"", false},
		{"vault_addr", false},
	}

	for _, tc := range tests {
		t.Run(tc.key, func(t *testing.T) {
			got := IsSensitiveKey(tc.key, patterns)
			if got != tc.want {
				t.Errorf("IsSensitiveKey(%q) = %v, want %v", tc.key, got, tc.want)
			}
		})
	}
}

func TestIsSensitiveKey_EmptyPatterns(t *testing.T) {
	got := IsSensitiveKey("token", nil)
	if got {
		t.Error("expected false with empty patterns")
	}
}

func TestIsSensitiveKey_CaseInsensitive(t *testing.T) {
	patterns := []string{"TOKEN"}

	if !IsSensitiveKey("github_token", patterns) {
		t.Error("expected IsSensitiveKey to be case-insensitive")
	}
	if !IsSensitiveKey("GITHUB_TOKEN", patterns) {
		t.Error("expected IsSensitiveKey to be case-insensitive (upper)")
	}
}

// --- RedactValue ---

func TestRedactValue(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"github PAT ghp_", "ghp_abcdef1234567890abcdef", "[REDACTED]"},
		{"github PAT gho_", "gho_abcdef1234567890abcdef", "[REDACTED]"},
		{"github PAT ghs_", "ghs_abcdef1234567890abcdef", "[REDACTED]"},
		{"github PAT ghr_", "ghr_abcdef1234567890abcdef", "[REDACTED]"},
		{"vault token", "hvs.CAESIAHfTbW1234567890", "[REDACTED]"},
		{"openai key long", "sk-proj-abcdefghijklmnopqrst", "[REDACTED]"},
		{"openai key short (not matched)", "sk-short", "sk-short"},
		{"PEM content", "-----BEGIN PRIVATE KEY-----\nMIIEvg...", "[REDACTED]"},
		{"PEM cert", "data with -----BEGIN CERTIFICATE----- in it", "[REDACTED]"},
		{"normal value", "hello world", "hello world"},
		{"empty string", "", ""},
		{"number string", "12345", "12345"},
		{"url", "http://vault:8200", "http://vault:8200"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := RedactValue(tc.input)
			if got != tc.want {
				t.Errorf("RedactValue(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// --- isSecretPattern ---

func TestIsSecretPattern(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"ghp_abc123def456", true},
		{"gho_abc123def456", true},
		{"ghs_abc123def456", true},
		{"ghr_abc123def456", true},
		{"hvs.test_token_123", true},
		{"sk-proj-longkeymorethan20chars", true},
		{"sk-short", false}, // < 20 chars
		{"-----BEGIN RSA PRIVATE KEY-----", true},
		{"contains -----BEGIN CERTIFICATE----- somewhere", true},
		{"normal_string", false},
		{"", false},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got := isSecretPattern(tc.input)
			if got != tc.want {
				t.Errorf("isSecretPattern(%q) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

// --- redactionHook ---

func TestNewRedactionHook_DefaultPatterns(t *testing.T) {
	hook := newRedactionHook(nil)

	if len(hook.sensitiveKeys) == 0 {
		t.Error("expected default patterns when nil is passed")
	}

	// Default patterns should include token, password, etc.
	expected := []string{"token", "password", "secret", "key", "authorization", "credential", "private"}
	if len(hook.sensitiveKeys) != len(expected) {
		t.Errorf("expected %d default patterns, got %d", len(expected), len(hook.sensitiveKeys))
	}
}

func TestNewRedactionHook_CustomPatterns(t *testing.T) {
	hook := newRedactionHook([]string{"API_KEY", "Secret_Token"})

	if len(hook.sensitiveKeys) != 2 {
		t.Fatalf("expected 2 patterns, got %d", len(hook.sensitiveKeys))
	}

	// Patterns should be lowercased.
	for _, p := range hook.sensitiveKeys {
		if p != strings.ToLower(p) {
			t.Errorf("expected pattern to be lowercase, got %q", p)
		}
	}
}

func TestRedactionHook_Run_DoesNotPanic(t *testing.T) {
	hook := newRedactionHook(nil)

	// Run is a no-op but must not panic.
	// Create a test event by writing to a buffer.
	var buf bytes.Buffer
	logger := zerolog.New(&buf).Hook(hook)
	logger.Info().Str("token", "secret_value").Msg("test")

	// The hook ran without panicking — that's the test.
	if buf.Len() == 0 {
		t.Error("expected log output")
	}
}

// --- Integration: logger with all features ---

func TestSetupLogger_ServiceField(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "INFO",
		Format: "json",
	}

	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)
	testLogger.Info().Msg("service test")

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("failed to parse log: %v", err)
	}

	if entry["service"] != "vault-agent-operator" {
		t.Errorf("expected service=vault-agent-operator, got %v", entry["service"])
	}
}

func TestSetupLogger_WithRequestIDIntegration(t *testing.T) {
	cfg := config.LoggingConfig{
		Level:  "INFO",
		Format: "json",
	}

	logger := SetupLogger(cfg)

	var buf bytes.Buffer
	testLogger := logger.Output(&buf)
	child := WithRequestID(testLogger, "abc-123")
	child.Info().Msg("request log")

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("failed to parse log: %v", err)
	}

	if entry["request_id"] != "abc-123" {
		t.Errorf("expected request_id=abc-123, got %v", entry["request_id"])
	}
	if entry["service"] != "vault-agent-operator" {
		t.Errorf("expected service=vault-agent-operator, got %v", entry["service"])
	}
}

// --- Context round-trip ---

func TestContextRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	original := zerolog.New(&buf).With().Str("test_field", "test_value").Logger()

	ctx := NewContext(context.Background(), original)
	retrieved := FromContext(ctx)

	retrieved.Info().Msg("round trip")

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("failed to parse log: %v", err)
	}

	if entry["test_field"] != "test_value" {
		t.Errorf("expected test_field=test_value after round trip, got %v", entry["test_field"])
	}
}
