package redaction

import (
	"strings"
	"testing"
)

func TestSanitizePrompt_EmptyPrompt(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := s.SanitizePrompt("", nil, ctx)
	if result != "" {
		t.Errorf("expected empty, got %q", result)
	}
}

func TestSanitizePrompt_NoSecrets(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	prompt := "Read the secret at secret/data/myapp"
	result := s.SanitizePrompt(prompt, nil, ctx)
	if result != prompt {
		t.Errorf("expected unchanged prompt, got %q", result)
	}
}

func TestSanitizePrompt_StructuredSecretData(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	prompt := "Write password=SuperSecret123 to vault"
	secretData := map[string]any{"password": "SuperSecret123"}
	result := s.SanitizePrompt(prompt, secretData, ctx)

	if strings.Contains(result, "SuperSecret123") {
		t.Error("secret value should have been replaced")
	}
	if !strings.Contains(result, "[SECRET_VALUE_") {
		t.Error("expected placeholder in result")
	}
}

func TestSanitizePrompt_NestedSecretData(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	prompt := "password is nested-secret-val"
	secretData := map[string]any{
		"data": map[string]any{
			"password": "nested-secret-val",
		},
	}
	result := s.SanitizePrompt(prompt, secretData, ctx)

	if strings.Contains(result, "nested-secret-val") {
		t.Error("nested secret should be replaced")
	}
}

func TestSanitizePrompt_ArraySecretData(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	prompt := "tokens are tok1 and tok2"
	secretData := map[string]any{
		"tokens": []any{"tok1", "tok2"},
	}
	result := s.SanitizePrompt(prompt, secretData, ctx)

	if strings.Contains(result, "tok1") || strings.Contains(result, "tok2") {
		t.Errorf("array secrets should be replaced, got %q", result)
	}
}

func TestSanitizePrompt_KeyValuePattern_Unquoted(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	prompt := "set password=MySecret123 on the server"
	result := s.SanitizePrompt(prompt, nil, ctx)

	if strings.Contains(result, "MySecret123") {
		t.Errorf("password=value should be sanitized, got %q", result)
	}
}

func TestSanitizePrompt_KeyValuePattern_DoubleQuoted(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	prompt := `set token="my-secret-token" on the server`
	result := s.SanitizePrompt(prompt, nil, ctx)

	if strings.Contains(result, "my-secret-token") {
		t.Errorf("quoted token should be sanitized, got %q", result)
	}
}

func TestSanitizePrompt_KeyValuePattern_SingleQuoted(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	prompt := `set api_key='sk-abcdefgh12345678901234567890' on the server`
	result := s.SanitizePrompt(prompt, nil, ctx)

	if strings.Contains(result, "sk-abcdefgh12345678901234567890") {
		t.Errorf("single-quoted api_key should be sanitized, got %q", result)
	}
}

func TestSanitizePrompt_GitHubToken(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	token := "ghp_ABCDEFGHIJKLMNOPQRSTUVWXyz1234567890"
	prompt := "use token " + token + " for auth"
	result := s.SanitizePrompt(prompt, nil, ctx)

	if strings.Contains(result, token) {
		t.Errorf("GitHub PAT should be sanitized, got %q", result)
	}
}

func TestSanitizePrompt_VaultToken(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	token := "hvs.ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	prompt := "use vault token " + token
	result := s.SanitizePrompt(prompt, nil, ctx)

	if strings.Contains(result, token) {
		t.Errorf("Vault token should be sanitized, got %q", result)
	}
}

func TestSanitizePrompt_OpenAIKey(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	key := "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
	prompt := "use key " + key
	result := s.SanitizePrompt(prompt, nil, ctx)

	if strings.Contains(result, key) {
		t.Errorf("OpenAI key should be sanitized, got %q", result)
	}
}

func TestSanitizePrompt_HexToken(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	hexToken := "abcdef0123456789abcdef0123456789abcdef0123456789"
	prompt := "the API key is " + hexToken
	result := s.SanitizePrompt(prompt, nil, ctx)

	if strings.Contains(result, hexToken) {
		t.Errorf("hex token should be sanitized, got %q", result)
	}
}

func TestSanitizePrompt_SafePromptUnchanged(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	prompt := "list all secrets under secret/data/ and report"
	result := s.SanitizePrompt(prompt, nil, ctx)

	if result != prompt {
		t.Errorf("safe prompt should be unchanged, got %q", result)
	}
}

func TestSanitizePrompt_MultiplePatterns(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ghToken := "ghp_ABCDEFGHIJKLMNOPQRSTUVWXyz1234567890"
	prompt := "set password=Secret123 and use token " + ghToken
	result := s.SanitizePrompt(prompt, nil, ctx)

	if strings.Contains(result, "Secret123") {
		t.Error("password should be sanitized")
	}
	if strings.Contains(result, ghToken) {
		t.Error("GitHub token should be sanitized")
	}
}

func TestIsLikelyWord(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"hello", true},
		{"HELLO", true},
		{"Hello", true},
		{"hello123", false},
		{"he+lo", false},
		{"", true}, // empty string has no non-alpha chars
	}
	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			if got := isLikelyWord(tc.input); got != tc.expected {
				t.Errorf("isLikelyWord(%q) = %v, want %v", tc.input, got, tc.expected)
			}
		})
	}
}

func TestSanitizePrompt_NonStringSecretData(t *testing.T) {
	s := NewPromptSanitizer()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	// Non-string values in secretData should also be handled.
	prompt := "port is 8080 and flag is true"
	secretData := map[string]any{
		"port": 8080,
		"flag": true,
	}
	result := s.SanitizePrompt(prompt, secretData, ctx)

	// The numeric/boolean values get converted to string and replaced.
	if strings.Contains(result, "8080") {
		t.Errorf("numeric value should be sanitized, got %q", result)
	}
}
