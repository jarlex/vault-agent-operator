package redaction

import (
	"fmt"
	"regexp"
	"strings"
)

// PromptSanitizer detects and replaces secret-like values in user prompts
// with placeholders via a SecretContext. This prevents accidental exposure
// of secrets in LLM conversations.
type PromptSanitizer interface {
	// SanitizePrompt replaces detected secret values in the prompt with
	// placeholders. secretData provides known structured secrets; the
	// sanitizer also detects inline secret patterns.
	SanitizePrompt(prompt string, secretData map[string]any, ctx *SecretContext) string
}

// DefaultPromptSanitizer implements PromptSanitizer with pattern-based
// detection for common secret formats.
type DefaultPromptSanitizer struct{}

// NewPromptSanitizer creates a new DefaultPromptSanitizer.
func NewPromptSanitizer() *DefaultPromptSanitizer {
	return &DefaultPromptSanitizer{}
}

// SanitizePrompt replaces secret values in the prompt with placeholders.
// It processes structured secretData first, then applies inline pattern detection.
func (s *DefaultPromptSanitizer) SanitizePrompt(prompt string, secretData map[string]any, ctx *SecretContext) string {
	if prompt == "" {
		return prompt
	}

	result := prompt

	// Step 1: Register and replace structured secret_data values.
	if len(secretData) > 0 {
		result = s.sanitizeStructuredData(result, secretData, ctx)
	}

	// Step 2: Detect and replace inline secret patterns.
	result = s.sanitizeInlinePatterns(result, ctx)

	return result
}

// sanitizeStructuredData registers all values from the structured secretData
// map and replaces them in the prompt text.
func (s *DefaultPromptSanitizer) sanitizeStructuredData(prompt string, data map[string]any, ctx *SecretContext) string {
	result := prompt
	s.registerAndReplace(data, ctx, &result)
	return result
}

// registerAndReplace recursively registers all string values in a map
// as secrets and replaces them in the text.
func (s *DefaultPromptSanitizer) registerAndReplace(data map[string]any, ctx *SecretContext, text *string) {
	for _, value := range data {
		switch v := value.(type) {
		case string:
			if v != "" {
				placeholder, err := ctx.RegisterSecret(v)
				if err == nil && placeholder != "" {
					*text = strings.ReplaceAll(*text, v, placeholder)
				}
			}
		case map[string]any:
			s.registerAndReplace(v, ctx, text)
		case []any:
			for _, item := range v {
				if sv, ok := item.(string); ok && sv != "" {
					placeholder, err := ctx.RegisterSecret(sv)
					if err == nil && placeholder != "" {
						*text = strings.ReplaceAll(*text, sv, placeholder)
					}
				}
				if mv, ok := item.(map[string]any); ok {
					s.registerAndReplace(mv, ctx, text)
				}
			}
		default:
			// Convert non-string values to string for checking.
			sv := fmt.Sprintf("%v", v)
			if sv != "" && sv != "<nil>" {
				placeholder, err := ctx.RegisterSecret(sv)
				if err == nil && placeholder != "" {
					*text = strings.ReplaceAll(*text, sv, placeholder)
				}
			}
		}
	}
}

// secretKeywords are words that typically precede a secret value in text.
var secretKeywords = []string{
	"password", "passwd", "pass",
	"token", "api_key", "apikey", "api-key",
	"secret", "credential",
	"authorization", "auth",
	"private_key", "private-key",
}

// Compiled patterns for inline secret detection.
var (
	// key=value patterns: password=mysecret, token="mysecret", api_key='mysecret'
	keyValuePattern = compileKeyValuePattern()

	// Known token prefixes (GitHub PAT, Vault tokens, OpenAI keys, etc.)
	ghTokenPattern    = regexp.MustCompile(`\bgh[porst]_[A-Za-z0-9_]{16,}\b`)
	vaultTokenPattern = regexp.MustCompile(`\bhvs\.[A-Za-z0-9_\-]{20,}\b`)
	openAIKeyPattern  = regexp.MustCompile(`\bsk-[A-Za-z0-9]{20,}\b`)

	// Base64-encoded values that look like secrets (long, no spaces).
	base64Pattern = regexp.MustCompile(`\b[A-Za-z0-9+/]{40,}={0,2}\b`)

	// Hex-encoded tokens (40+ chars, common in API keys).
	hexTokenPattern = regexp.MustCompile(`\b[0-9a-fA-F]{40,}\b`)
)

// compileKeyValuePattern builds a regex that matches keyword=value pairs
// where the value may be quoted or unquoted.
func compileKeyValuePattern() *regexp.Regexp {
	keywords := strings.Join(secretKeywords, "|")
	// Match: keyword=value, keyword="value", keyword='value'
	// Also match with spaces around =
	pattern := fmt.Sprintf(
		`(?i)(?:%s)\s*[=:]\s*(?:"([^"]+)"|'([^']+)'|(\S+))`,
		keywords,
	)
	return regexp.MustCompile(pattern)
}

// sanitizeInlinePatterns detects common secret patterns in text and
// replaces them with placeholders.
func (s *DefaultPromptSanitizer) sanitizeInlinePatterns(text string, ctx *SecretContext) string {
	result := text

	// Detect known token formats.
	result = s.replaceMatches(result, ghTokenPattern, ctx)
	result = s.replaceMatches(result, vaultTokenPattern, ctx)
	result = s.replaceMatches(result, openAIKeyPattern, ctx)

	// Detect key=value patterns.
	result = s.replaceKeyValueMatches(result, ctx)

	// Detect long hex tokens (potential API keys).
	result = s.replaceMatches(result, hexTokenPattern, ctx)

	// Detect long base64 values (potential encoded secrets).
	// Only match if they're long enough to likely be secrets.
	result = s.replaceBase64Matches(result, ctx)

	return result
}

// replaceMatches finds all regex matches and registers them as secrets.
func (s *DefaultPromptSanitizer) replaceMatches(text string, re *regexp.Regexp, ctx *SecretContext) string {
	return re.ReplaceAllStringFunc(text, func(match string) string {
		placeholder, err := ctx.RegisterSecret(match)
		if err != nil || placeholder == "" {
			return match
		}
		return placeholder
	})
}

// replaceKeyValueMatches handles key=value patterns, extracting the value part.
func (s *DefaultPromptSanitizer) replaceKeyValueMatches(text string, ctx *SecretContext) string {
	return keyValuePattern.ReplaceAllStringFunc(text, func(match string) string {
		submatches := keyValuePattern.FindStringSubmatch(match)
		if submatches == nil {
			return match
		}

		// Find the captured value (double-quoted, single-quoted, or unquoted).
		var value string
		for i := 1; i < len(submatches); i++ {
			if submatches[i] != "" {
				value = submatches[i]
				break
			}
		}

		if value == "" {
			return match
		}

		placeholder, err := ctx.RegisterSecret(value)
		if err != nil || placeholder == "" {
			return match
		}

		// Replace only the value portion in the match.
		return strings.Replace(match, value, placeholder, 1)
	})
}

// replaceBase64Matches handles base64-encoded values, only targeting
// values that are long enough to plausibly be secrets.
func (s *DefaultPromptSanitizer) replaceBase64Matches(text string, ctx *SecretContext) string {
	return base64Pattern.ReplaceAllStringFunc(text, func(match string) string {
		// Skip short matches that are likely not secrets.
		if len(match) < 40 {
			return match
		}
		// Skip if it looks like a normal word (all lowercase/uppercase alpha).
		if isLikelyWord(match) {
			return match
		}
		placeholder, err := ctx.RegisterSecret(match)
		if err != nil || placeholder == "" {
			return match
		}
		return placeholder
	})
}

// isLikelyWord checks if a string looks like a natural language word
// (all alphabetic, no digits, no special chars).
func isLikelyWord(s string) bool {
	for _, r := range s {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')) {
			return false
		}
	}
	return true
}
