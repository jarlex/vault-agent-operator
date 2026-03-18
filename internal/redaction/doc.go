// Package redaction implements secret value protection for vault-agent-operator.
// It ensures that secret values never reach the LLM by providing placeholder
// substitution, tool result redaction, and prompt sanitization.
package redaction
