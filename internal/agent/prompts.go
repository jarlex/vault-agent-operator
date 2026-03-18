package agent

import (
	"fmt"
	"os"
	"strings"

	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/llm"
)

// defaultSystemPrompt is the fallback system prompt used when the
// prompt file cannot be loaded.
const defaultSystemPrompt = `You are a Vault Operator Agent — an API executor that manages HashiCorp Vault through available MCP tools.

You are an API executor, NOT a chatbot. You operate as a backend service that receives operations and executes them.

- NEVER ask clarifying questions.
- NEVER ask for confirmation.
- ALWAYS attempt to execute the requested operation using available tools immediately.
- If the request is ambiguous, make reasonable assumptions and proceed.
- If you cannot fulfill the request, return a clear error message explaining why.

You ALWAYS use the tools provided. You NEVER fabricate Vault data or tool results.`

// LoadSystemPrompt loads the system prompt from a file or returns a default.
// It supports simple template variables: {{ vault_addr }} and {{ available_tools }}.
func LoadSystemPrompt(promptPath string, vaultAddr string, toolNames []string, logger zerolog.Logger) string {
	prompt := loadPromptFile(promptPath, logger)
	prompt = applyPromptTemplateVars(prompt, vaultAddr, toolNames)
	return prompt
}

// loadPromptFile reads the system prompt from a file.
// If the file cannot be read, the default embedded prompt is returned.
func loadPromptFile(path string, logger zerolog.Logger) string {
	if path == "" {
		logger.Debug().Msg("agent.prompts.no_path_configured")
		return defaultSystemPrompt
	}

	data, err := os.ReadFile(path)
	if err != nil {
		logger.Warn().
			Str("path", path).
			Err(err).
			Msg("agent.prompts.load_failed_using_default")
		return defaultSystemPrompt
	}

	prompt := strings.TrimSpace(string(data))
	if prompt == "" {
		logger.Warn().
			Str("path", path).
			Msg("agent.prompts.empty_file_using_default")
		return defaultSystemPrompt
	}

	logger.Debug().
		Str("path", path).
		Int("length", len(prompt)).
		Msg("agent.prompts.loaded")

	return prompt
}

// applyPromptTemplateVars replaces Jinja-style template variables
// {{ vault_addr }} and {{ available_tools }} in the prompt.
func applyPromptTemplateVars(prompt string, vaultAddr string, toolNames []string) string {
	toolList := "None available"
	if len(toolNames) > 0 {
		toolList = strings.Join(toolNames, ", ")
	}

	result := prompt
	result = strings.ReplaceAll(result, "{{ vault_addr }}", vaultAddr)
	result = strings.ReplaceAll(result, "{{vault_addr}}", vaultAddr)
	result = strings.ReplaceAll(result, "{{ available_tools }}", toolList)
	result = strings.ReplaceAll(result, "{{available_tools}}", toolList)

	return result
}

// BuildInitialMessages constructs the initial message list for the LLM:
// [system_prompt, user_prompt].
func BuildInitialMessages(systemPrompt string, userPrompt string) []llm.Message {
	return []llm.Message{
		{
			Role:    "system",
			Content: stringPtr(systemPrompt),
		},
		{
			Role:    "user",
			Content: stringPtr(userPrompt),
		},
	}
}

// stringPtr returns a pointer to a string — helper for building messages
// where Content is *string.
func stringPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// FormatToolNames extracts tool names from a slice of tools for
// prompt template substitution.
func FormatToolNames(tools []ToolInfo) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name)
	}
	return names
}

// ToolInfo is a minimal representation of a tool for prompt building.
// This avoids importing the mcp or llm packages in the prompts file.
type ToolInfo struct {
	Name        string
	Description string
}

// FormatToolNamesFromStrings is a convenience alias when you already have string names.
func FormatToolNamesFromStrings(names []string) string {
	if len(names) == 0 {
		return "None available"
	}
	return fmt.Sprintf("%s", strings.Join(names, ", "))
}
