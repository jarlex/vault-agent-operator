package redaction

import (
	"strings"
)

// SecretRedactor provides methods for redacting secret values from tool results
// and error messages, and for restoring placeholders in tool arguments.
type SecretRedactor interface {
	// RedactToolResult redacts secret values from an MCP tool result.
	// Returns a string safe for inclusion in LLM messages and audit logs.
	RedactToolResult(toolName string, result string, ctx *SecretContext) string

	// RedactErrorMessage redacts known secret values from an error message
	// so it can be safely sent to the LLM for retry reasoning.
	RedactErrorMessage(errorMsg string, ctx *SecretContext) string

	// RestorePlaceholders replaces placeholder tokens in tool-call arguments
	// with their real secret values so the MCP tool receives real data.
	RestorePlaceholders(text string, ctx *SecretContext) string

	// RestoreMapPlaceholders replaces placeholder tokens in a map of
	// tool-call arguments with their real secret values.
	RestoreMapPlaceholders(arguments map[string]any, ctx *SecretContext) map[string]any
}

// DefaultSecretRedactor implements SecretRedactor using the policy-based
// redaction system.
type DefaultSecretRedactor struct{}

// NewSecretRedactor creates a new DefaultSecretRedactor.
func NewSecretRedactor() *DefaultSecretRedactor {
	return &DefaultSecretRedactor{}
}

// RedactToolResult selects the appropriate redaction policy for the given
// tool name and applies it to redact secret values from the result.
func (r *DefaultSecretRedactor) RedactToolResult(toolName string, result string, ctx *SecretContext) string {
	if result == "" || ctx == nil {
		return result
	}

	policy := GetPolicyForTool(toolName)
	return policy.Redact(result, ctx)
}

// RedactErrorMessage replaces any known secret values that may have leaked
// into an error message with their placeholder tokens. This ensures the
// LLM never sees real secret values even in error paths.
func (r *DefaultSecretRedactor) RedactErrorMessage(errorMsg string, ctx *SecretContext) string {
	if errorMsg == "" || ctx == nil {
		return errorMsg
	}

	return ctx.redactKnownValues(errorMsg)
}

// RestorePlaceholders replaces all placeholder tokens in the given text
// with their real secret values. Used to prepare tool arguments for
// MCP invocation.
func (r *DefaultSecretRedactor) RestorePlaceholders(text string, ctx *SecretContext) string {
	if text == "" || ctx == nil {
		return text
	}

	return ctx.ResolveAllPlaceholders(text)
}

// RestoreMapPlaceholders recursively walks a map of tool-call arguments
// and replaces any placeholder tokens with real secret values.
func (r *DefaultSecretRedactor) RestoreMapPlaceholders(arguments map[string]any, ctx *SecretContext) map[string]any {
	if len(arguments) == 0 || ctx == nil {
		return arguments
	}

	result := make(map[string]any, len(arguments))
	for key, value := range arguments {
		result[key] = restoreValue(value, ctx)
	}
	return result
}

// restoreValue recursively restores placeholders in a single value.
func restoreValue(value any, ctx *SecretContext) any {
	switch v := value.(type) {
	case string:
		return ctx.ResolveAllPlaceholders(v)
	case map[string]any:
		result := make(map[string]any, len(v))
		for key, val := range v {
			result[key] = restoreValue(val, ctx)
		}
		return result
	case []any:
		result := make([]any, len(v))
		for i, val := range v {
			result[i] = restoreValue(val, ctx)
		}
		return result
	default:
		return value
	}
}

// redactKnownValues replaces all known secret values in text with their
// corresponding placeholders. This is the reverse of ResolveAllPlaceholders.
func (c *SecretContext) redactKnownValues(text string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.destroyed || len(c.valueToPlaceholder) == 0 {
		return text
	}

	result := text
	for value, placeholder := range c.valueToPlaceholder {
		if value != "" {
			result = strings.ReplaceAll(result, value, placeholder)
		}
	}
	return result
}
