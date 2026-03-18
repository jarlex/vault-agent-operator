package redaction

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestKVReadPolicy_BasicRedaction(t *testing.T) {
	p := &KVReadPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"request_id":"req-1","data":{"username":"admin","password":"hunter2"},"renewable":false}`
	redacted := p.Redact(result, ctx)

	// Metadata preserved.
	if !strings.Contains(redacted, `"request_id"`) {
		t.Error("request_id should be preserved")
	}
	if !strings.Contains(redacted, `"renewable"`) {
		t.Error("renewable should be preserved")
	}
	// Data values redacted.
	if strings.Contains(redacted, "hunter2") {
		t.Error("password should be redacted")
	}
	if strings.Contains(redacted, "admin") {
		t.Error("username should be redacted")
	}
}

func TestKVReadPolicy_InvalidJSON(t *testing.T) {
	p := &KVReadPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	// Invalid JSON — should apply conservative redaction.
	result := "not-json-content-with-secrets"
	redacted := p.Redact(result, ctx)

	// Should be redacted as a whole string.
	if redacted == result {
		t.Error("invalid JSON should still be redacted")
	}
}

func TestKVReadPolicy_PreservesMetadataKeys(t *testing.T) {
	p := &KVReadPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"request_id":"r1","lease_id":"l1","renewable":true,"lease_duration":3600,"version":2,"created_time":"2024-01-01T00:00:00Z","data":{"secret":"val"}}`
	redacted := p.Redact(result, ctx)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(redacted), &parsed); err != nil {
		t.Fatalf("redacted result is not valid JSON: %v", err)
	}

	// All metadata keys should be preserved.
	for _, key := range []string{"request_id", "lease_id", "renewable", "lease_duration", "version", "created_time"} {
		if _, ok := parsed[key]; !ok {
			t.Errorf("expected metadata key %q to be preserved", key)
		}
	}
}

func TestKVWritePolicy(t *testing.T) {
	p := &KVWritePolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"request_id":"r1","version":1,"data":"sensitive"}`
	redacted := p.Redact(result, ctx)

	if strings.Contains(redacted, "sensitive") {
		t.Error("non-metadata data should be redacted")
	}
	// request_id and version are metadata — preserved.
	if !strings.Contains(redacted, `"request_id"`) {
		t.Error("request_id should be preserved")
	}
}

func TestPKIPolicy_RedactPrivateKey(t *testing.T) {
	p := &PKIPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"serial_number":"AB:CD","certificate":"cert-pem","private_key":"-----BEGIN RSA PRIVATE KEY-----\nMIIE...","expiration":1234567890}`
	redacted := p.Redact(result, ctx)

	if strings.Contains(redacted, "PRIVATE KEY") {
		t.Error("private key should be redacted")
	}
	// serial_number and certificate are allowed.
	if !strings.Contains(redacted, "AB:CD") {
		t.Error("serial_number should be preserved")
	}
}

func TestPKIPolicy_PreservesAllowedFields(t *testing.T) {
	p := &PKIPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"serial_number":"AA:BB","certificate":"public-cert","issuing_ca":"ca-cert","expiration":9999}`
	redacted := p.Redact(result, ctx)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(redacted), &parsed); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if parsed["serial_number"] != "AA:BB" {
		t.Error("serial_number should be preserved")
	}
	if parsed["certificate"] != "public-cert" {
		t.Error("certificate should be preserved")
	}
}

func TestPKIPolicy_InvalidJSON(t *testing.T) {
	p := &PKIPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	// PEM content in raw string.
	result := "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
	redacted := p.Redact(result, ctx)

	if strings.Contains(redacted, "PRIVATE KEY") {
		t.Error("PEM private key should be redacted even in raw text")
	}
}

func TestListPolicy_NoRedaction(t *testing.T) {
	p := &ListPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"keys":["secret1","secret2","secret3"]}`
	redacted := p.Redact(result, ctx)

	if redacted != result {
		t.Errorf("list policy should not redact, got %q", redacted)
	}
}

func TestDefaultPolicy_RedactsEverything(t *testing.T) {
	p := &DefaultPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"key":"value","nested":{"deep":"secret"}}`
	redacted := p.Redact(result, ctx)

	if strings.Contains(redacted, "value") {
		t.Error("all string values should be redacted")
	}
	if strings.Contains(redacted, "secret") {
		t.Error("nested values should be redacted")
	}
}

func TestDefaultPolicy_NonJSONString(t *testing.T) {
	p := &DefaultPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := "raw sensitive data"
	redacted := p.Redact(result, ctx)

	if redacted == result {
		t.Error("non-JSON string should be redacted")
	}
	if !strings.Contains(redacted, "[SECRET_VALUE_") {
		t.Error("expected placeholder")
	}
}

func TestGetPolicyForTool(t *testing.T) {
	tests := []struct {
		toolName     string
		expectedType string
	}{
		{"vault_kv_read", "*redaction.KVReadPolicy"},
		{"vault_kv_get", "*redaction.KVReadPolicy"},
		{"vault_kv_list", "*redaction.ListPolicy"},
		{"vault_secret_list_all", "*redaction.ListPolicy"},
		{"vault_kv_write", "*redaction.KVWritePolicy"},
		{"vault_kv_put", "*redaction.KVWritePolicy"},
		{"vault_kv_create", "*redaction.KVWritePolicy"},
		{"vault_kv_update", "*redaction.KVWritePolicy"},
		{"vault_pki_issue", "*redaction.PKIPolicy"},
		{"vault_cert_sign", "*redaction.PKIPolicy"},
		{"vault_pki_list", "*redaction.ListPolicy"}, // list takes priority
		{"unknown_tool", "*redaction.DefaultPolicy"},
		{"vault_status", "*redaction.DefaultPolicy"},
	}

	for _, tc := range tests {
		t.Run(tc.toolName, func(t *testing.T) {
			policy := GetPolicyForTool(tc.toolName)
			typeName := typeNameOf(policy)
			if typeName != tc.expectedType {
				t.Errorf("GetPolicyForTool(%q) = %s, want %s", tc.toolName, typeName, tc.expectedType)
			}
		})
	}
}

func typeNameOf(v any) string {
	return strings.Replace(strings.Replace(
		strings.Replace(typeOf(v), "redaction.", "redaction.", 1),
		"*", "*", 1), " ", "", -1)
}

func typeOf(v any) string {
	return strings.TrimPrefix(strings.TrimPrefix(
		strings.Replace(
			strings.Replace(typeString(v), " ", "", -1),
			"redaction.", "redaction.", 1),
		""), "")
}

func typeString(v any) string {
	if v == nil {
		return "<nil>"
	}
	return formatType(v)
}

func formatType(v any) string {
	switch v.(type) {
	case *KVReadPolicy:
		return "*redaction.KVReadPolicy"
	case *KVWritePolicy:
		return "*redaction.KVWritePolicy"
	case *PKIPolicy:
		return "*redaction.PKIPolicy"
	case *ListPolicy:
		return "*redaction.ListPolicy"
	case *DefaultPolicy:
		return "*redaction.DefaultPolicy"
	default:
		return "unknown"
	}
}

func TestRedactAllValues_JSONArray(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `["secret1","secret2"]`
	redacted := redactAllValues(result, ctx)

	if strings.Contains(redacted, "secret1") || strings.Contains(redacted, "secret2") {
		t.Error("all values in array should be redacted")
	}
}

func TestRedactAllValues_JSONString(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `"just-a-string"`
	redacted := redactAllValues(result, ctx)

	if strings.Contains(redacted, "just-a-string") {
		t.Error("string JSON value should be redacted")
	}
}

func TestRedactMapValues_NestedMap(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	obj := map[string]any{
		"level1": map[string]any{
			"level2": "deep-secret",
		},
	}

	redactMapValues(obj, ctx)

	l1 := obj["level1"].(map[string]any)
	if l1["level2"] == "deep-secret" {
		t.Error("nested string should be redacted")
	}
}

func TestRedactSliceValues_NestedSlice(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	arr := []any{"val1", map[string]any{"key": "val2"}, []any{"val3"}}
	redactSliceValues(arr, ctx)

	if arr[0] == "val1" {
		t.Error("string in slice should be redacted")
	}
}

func TestKVReadPolicy_DataSectionRedacted(t *testing.T) {
	p := &KVReadPolicy{}
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"data":{"db_password":"secret123","api_key":"key456"}}`
	redacted := p.Redact(result, ctx)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(redacted), &parsed); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	data := parsed["data"].(map[string]any)
	for key, val := range data {
		s, ok := val.(string)
		if !ok {
			continue
		}
		if !strings.HasPrefix(s, "[SECRET_VALUE_") {
			t.Errorf("data[%q] should be a placeholder, got %q", key, s)
		}
	}
}
