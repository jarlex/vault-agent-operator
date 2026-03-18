package redaction

import (
	"encoding/json"
	"strings"
)

// RedactionPolicy defines how tool results are redacted for a specific tool type.
type RedactionPolicy interface {
	// Redact processes a raw tool result string and returns a redacted version
	// with secret values replaced by placeholders via the SecretContext.
	Redact(result string, ctx *SecretContext) string
}

// kvReadAllowedMetadata lists metadata fields that are safe to keep unredacted
// in KV read results. These are operational metadata, not secret values.
var kvReadAllowedMetadata = map[string]bool{
	"request_id":           true,
	"lease_id":             true,
	"renewable":            true,
	"lease_duration":       true,
	"wrap_info":            true,
	"warnings":             true,
	"auth":                 true,
	"mount_type":           true,
	"created_time":         true,
	"custom_metadata":      true,
	"deletion_time":        true,
	"destroyed":            true,
	"version":              true,
	"cas_required":         true,
	"delete_version_after": true,
	"max_versions":         true,
	"oldest_version":       true,
	"current_version":      true,
}

// KVReadPolicy redacts data values from Vault KV read operations while
// preserving metadata fields (version, created_time, etc.).
type KVReadPolicy struct{}

// Redact processes KV read results. It keeps metadata but redacts
// all values in the "data" section.
func (p *KVReadPolicy) Redact(result string, ctx *SecretContext) string {
	var parsed map[string]any
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		// If we can't parse, apply conservative redaction.
		return redactAllValues(result, ctx)
	}

	redactKVData(parsed, ctx)
	out, err := json.Marshal(parsed)
	if err != nil {
		return redactAllValues(result, ctx)
	}
	return string(out)
}

// redactKVData recursively redacts the "data" section of KV responses
// while keeping allowed metadata keys.
func redactKVData(obj map[string]any, ctx *SecretContext) {
	for key, value := range obj {
		if kvReadAllowedMetadata[key] {
			continue
		}
		switch v := value.(type) {
		case map[string]any:
			// If this is the "data" key, redact all leaf values within it.
			if key == "data" {
				redactMapValues(v, ctx)
			} else {
				redactKVData(v, ctx)
			}
		case string:
			placeholder, err := ctx.RegisterSecret(v)
			if err == nil && placeholder != "" {
				obj[key] = placeholder
			}
		}
	}
}

// redactMapValues redacts all string leaf values in a map, recursively.
func redactMapValues(obj map[string]any, ctx *SecretContext) {
	for key, value := range obj {
		switch v := value.(type) {
		case string:
			if v != "" {
				placeholder, err := ctx.RegisterSecret(v)
				if err == nil && placeholder != "" {
					obj[key] = placeholder
				}
			}
		case map[string]any:
			redactMapValues(v, ctx)
		case []any:
			redactSliceValues(v, ctx)
			obj[key] = v
		}
	}
}

// redactSliceValues redacts all string values in a slice, recursively.
func redactSliceValues(arr []any, ctx *SecretContext) {
	for i, value := range arr {
		switch v := value.(type) {
		case string:
			if v != "" {
				placeholder, err := ctx.RegisterSecret(v)
				if err == nil && placeholder != "" {
					arr[i] = placeholder
				}
			}
		case map[string]any:
			redactMapValues(v, ctx)
		case []any:
			redactSliceValues(v, ctx)
		}
	}
}

// KVWritePolicy redacts input values from Vault KV write operations,
// keeping only safe metadata in the response.
type KVWritePolicy struct{}

// Redact processes KV write results. Write responses typically contain
// metadata about the written secret, not the secret data itself.
// We still redact any string values that aren't known metadata.
func (p *KVWritePolicy) Redact(result string, ctx *SecretContext) string {
	var parsed map[string]any
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		return redactAllValues(result, ctx)
	}

	redactNonMetadata(parsed, ctx)
	out, err := json.Marshal(parsed)
	if err != nil {
		return redactAllValues(result, ctx)
	}
	return string(out)
}

// redactNonMetadata redacts all values except known metadata fields.
func redactNonMetadata(obj map[string]any, ctx *SecretContext) {
	for key, value := range obj {
		if kvReadAllowedMetadata[key] {
			continue
		}
		switch v := value.(type) {
		case string:
			if v != "" {
				placeholder, err := ctx.RegisterSecret(v)
				if err == nil && placeholder != "" {
					obj[key] = placeholder
				}
			}
		case map[string]any:
			redactNonMetadata(v, ctx)
		case []any:
			redactSliceValues(v, ctx)
			obj[key] = v
		}
	}
}

// PKIPolicy redacts private keys and sensitive PKI material while
// preserving certificate metadata (serial number, issuer, expiry, etc.).
type PKIPolicy struct{}

// pkiAllowedFields lists PKI metadata fields that are safe to keep.
var pkiAllowedFields = map[string]bool{
	"serial_number":   true,
	"certificate":     true, // public cert is not secret
	"issuing_ca":      true, // CA cert is not secret
	"ca_chain":        true,
	"expiration":      true,
	"revocation_time": true,
}

// Redact processes PKI tool results, redacting private keys and other
// sensitive material while keeping certificate metadata.
func (p *PKIPolicy) Redact(result string, ctx *SecretContext) string {
	var parsed map[string]any
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		return redactPEMContent(result, ctx)
	}

	redactPKIFields(parsed, ctx)
	out, err := json.Marshal(parsed)
	if err != nil {
		return redactPEMContent(result, ctx)
	}
	return string(out)
}

// redactPKIFields redacts private key fields in PKI responses.
func redactPKIFields(obj map[string]any, ctx *SecretContext) {
	for key, value := range obj {
		switch v := value.(type) {
		case string:
			// Always redact private keys, even in allowed fields.
			if strings.Contains(v, "-----BEGIN") && strings.Contains(v, "PRIVATE KEY") {
				placeholder, err := ctx.RegisterSecret(v)
				if err == nil && placeholder != "" {
					obj[key] = placeholder
				}
				continue
			}
			// Redact fields not in the allowlist.
			if !pkiAllowedFields[key] && v != "" {
				placeholder, err := ctx.RegisterSecret(v)
				if err == nil && placeholder != "" {
					obj[key] = placeholder
				}
			}
		case map[string]any:
			redactPKIFields(v, ctx)
		}
	}
}

// redactPEMContent is a fallback that redacts PEM-encoded content from
// raw strings when JSON parsing fails.
func redactPEMContent(text string, ctx *SecretContext) string {
	result := text
	// Redact private key blocks.
	for {
		beginIdx := strings.Index(result, "-----BEGIN")
		if beginIdx == -1 {
			break
		}
		endMarker := "-----END"
		endIdx := strings.Index(result[beginIdx:], endMarker)
		if endIdx == -1 {
			break
		}
		// Find the trailing "-----" that closes the END line.
		// Skip past "-----END" (8 chars) to avoid matching its own leading dashes.
		afterEnd := beginIdx + endIdx + len(endMarker)
		endLineIdx := strings.Index(result[afterEnd:], "-----")
		if endLineIdx == -1 {
			break
		}
		fullEnd := afterEnd + endLineIdx + 5 // 5 = len("-----")
		pemBlock := result[beginIdx:fullEnd]

		if strings.Contains(pemBlock, "PRIVATE KEY") {
			placeholder, err := ctx.RegisterSecret(pemBlock)
			if err == nil && placeholder != "" {
				result = result[:beginIdx] + placeholder + result[fullEnd:]
			} else {
				break
			}
		} else {
			// Skip non-private-key PEM blocks.
			break
		}
	}
	return result
}

// ListPolicy is a no-op policy for list operations, which do not
// return secret data — only key names.
type ListPolicy struct{}

// Redact returns the result unchanged for list operations.
func (p *ListPolicy) Redact(result string, _ *SecretContext) string {
	return result
}

// DefaultPolicy applies conservative redaction to unknown tool types,
// redacting ALL string values.
type DefaultPolicy struct{}

// Redact applies conservative redaction, replacing all string values
// with placeholders.
func (p *DefaultPolicy) Redact(result string, ctx *SecretContext) string {
	return redactAllValues(result, ctx)
}

// redactAllValues attempts to parse the result as JSON and redact all
// string values. If parsing fails, it registers the entire string as a secret.
func redactAllValues(result string, ctx *SecretContext) string {
	var parsed any
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		// Not JSON — redact the entire value.
		if result != "" {
			placeholder, err := ctx.RegisterSecret(result)
			if err == nil && placeholder != "" {
				return placeholder
			}
		}
		return result
	}

	switch v := parsed.(type) {
	case map[string]any:
		redactMapValues(v, ctx)
		out, _ := json.Marshal(v)
		return string(out)
	case []any:
		redactSliceValues(v, ctx)
		out, _ := json.Marshal(v)
		return string(out)
	case string:
		if v != "" {
			placeholder, err := ctx.RegisterSecret(v)
			if err == nil && placeholder != "" {
				return placeholder
			}
		}
	}
	return result
}

// GetPolicyForTool returns the appropriate redaction policy for a given
// tool name based on prefix matching.
func GetPolicyForTool(toolName string) RedactionPolicy {
	lower := strings.ToLower(toolName)

	// List operations — no redaction needed.
	if strings.Contains(lower, "list") {
		return &ListPolicy{}
	}

	// KV read operations.
	if strings.Contains(lower, "read") || strings.Contains(lower, "get") {
		return &KVReadPolicy{}
	}

	// KV write operations.
	if strings.Contains(lower, "write") || strings.Contains(lower, "put") ||
		strings.Contains(lower, "create") || strings.Contains(lower, "update") {
		return &KVWritePolicy{}
	}

	// PKI operations.
	if strings.Contains(lower, "pki") || strings.Contains(lower, "cert") ||
		strings.Contains(lower, "issue") || strings.Contains(lower, "sign") {
		return &PKIPolicy{}
	}

	// Default: conservative — redact everything.
	return &DefaultPolicy{}
}
