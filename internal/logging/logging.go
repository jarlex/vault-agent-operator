// Package logging provides structured logging setup for vault-agent-operator.
package logging

import (
	"context"
	"io"
	"os"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/config"
)

// contextKey is an unexported type for context keys in this package.
type contextKey string

const (
	// loggerKey is the context key for a request-scoped logger.
	loggerKey contextKey = "logger"
	// requestIDKey is the context key for the request ID.
	requestIDKey contextKey = "request_id"
)

// SetupLogger creates and configures a zerolog.Logger from the given config.
func SetupLogger(cfg config.LoggingConfig) zerolog.Logger {
	zerolog.TimeFieldFormat = time.RFC3339

	var output io.Writer = os.Stdout
	if strings.EqualFold(cfg.Format, "console") {
		output = zerolog.ConsoleWriter{Out: os.Stdout, TimeFormat: time.RFC3339}
	}

	level := parseLevel(cfg.Level)

	logger := zerolog.New(output).
		With().
		Timestamp().
		Str("service", "vault-agent-operator").
		Logger().
		Level(level).
		Hook(newRedactionHook(cfg.RedactPatterns))

	return logger
}

// WithRequestID returns a child logger with the given request ID bound.
func WithRequestID(logger zerolog.Logger, requestID string) zerolog.Logger {
	return logger.With().Str("request_id", requestID).Logger()
}

// NewContext returns a new context with the logger attached.
func NewContext(ctx context.Context, logger zerolog.Logger) context.Context {
	return context.WithValue(ctx, loggerKey, logger)
}

// FromContext extracts the logger from context, or returns a disabled logger.
func FromContext(ctx context.Context) zerolog.Logger {
	if l, ok := ctx.Value(loggerKey).(zerolog.Logger); ok {
		return l
	}
	return zerolog.Nop()
}

// parseLevel maps a level string to a zerolog.Level.
func parseLevel(level string) zerolog.Level {
	switch strings.ToUpper(level) {
	case "DEBUG":
		return zerolog.DebugLevel
	case "INFO":
		return zerolog.InfoLevel
	case "WARN", "WARNING":
		return zerolog.WarnLevel
	case "ERROR":
		return zerolog.ErrorLevel
	default:
		return zerolog.InfoLevel
	}
}

// redactionHook is a zerolog.Hook that redacts sensitive field values from log events.
type redactionHook struct {
	// sensitiveKeys are field name patterns that indicate a value should be redacted.
	sensitiveKeys []string
}

// newRedactionHook creates a redaction hook with the given sensitive field patterns.
// If patterns is empty, sensible defaults are used.
func newRedactionHook(patterns []string) redactionHook {
	if len(patterns) == 0 {
		patterns = []string{
			"token", "password", "secret", "key",
			"authorization", "credential", "private",
		}
	}
	lower := make([]string, len(patterns))
	for i, p := range patterns {
		lower[i] = strings.ToLower(p)
	}
	return redactionHook{sensitiveKeys: lower}
}

// Run implements zerolog.Hook. It is called for every log event.
//
// IMPORTANT — zerolog limitation:
// zerolog hooks receive the *zerolog.Event but cannot inspect or modify fields
// that have already been added to it. The Event type exposes only methods to
// *add* new fields — there is no getter or iterator for existing fields.
// This means automatic field-level redaction (scanning each key/value pair and
// replacing matches) is structurally impossible via the Hook interface.
//
// How redaction works in vault-agent-operator instead:
//  1. **Application-layer redaction**: Callers MUST use [RedactValue] to sanitise
//     values before passing them to the logger (e.g. `logger.Str("token", RedactValue(tok))`).
//  2. **Key checking**: Use [IsSensitiveKey] to test whether a field name suggests
//     the value is secret, and redact accordingly before logging.
//  3. **Secret redactor subsystem**: The internal/redaction package provides
//     [redaction.SecretRedactor] which replaces known secrets in arbitrary text.
//     Use it for large blobs (LLM prompts/responses) before logging.
//
// This hook is intentionally a no-op. It is kept in the logger chain so that a
// future io.Writer-based redaction layer (wrapping the output) can be added
// without changing callers.
func (h redactionHook) Run(e *zerolog.Event, level zerolog.Level, msg string) {
	// Intentionally empty — see doc comment above.
}

// RedactValue checks a string against known secret value patterns
// and returns "[REDACTED]" if it matches.
func RedactValue(value string) string {
	if isSecretPattern(value) {
		return "[REDACTED]"
	}
	return value
}

// isSecretPattern checks if a value looks like a known secret format.
func isSecretPattern(value string) bool {
	// GitHub Personal Access Token.
	if strings.HasPrefix(value, "ghp_") || strings.HasPrefix(value, "gho_") ||
		strings.HasPrefix(value, "ghs_") || strings.HasPrefix(value, "ghr_") {
		return true
	}
	// Vault token.
	if strings.HasPrefix(value, "hvs.") {
		return true
	}
	// OpenAI API key.
	if strings.HasPrefix(value, "sk-") && len(value) > 20 {
		return true
	}
	// PEM content.
	if strings.Contains(value, "-----BEGIN") {
		return true
	}
	return false
}

// IsSensitiveKey returns true if the key name suggests it holds a secret value.
func IsSensitiveKey(key string, patterns []string) bool {
	lower := strings.ToLower(key)
	for _, p := range patterns {
		if strings.Contains(lower, strings.ToLower(p)) {
			return true
		}
	}
	return false
}
