package redaction

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRedactToolResult_KVRead(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	kvResult := `{"request_id":"abc","data":{"password":"s3cret","api_key":"key123"}}`
	redacted := r.RedactToolResult("vault_kv_read", kvResult, ctx)

	// Metadata should be preserved.
	if !strings.Contains(redacted, `"request_id"`) {
		t.Error("expected request_id to be preserved")
	}
	// Data values should be redacted.
	if strings.Contains(redacted, "s3cret") {
		t.Error("password should be redacted")
	}
	if strings.Contains(redacted, "key123") {
		t.Error("api_key should be redacted")
	}
	// Placeholders should be present.
	if !strings.Contains(redacted, "[SECRET_VALUE_") {
		t.Error("expected placeholders in redacted result")
	}
}

func TestRedactToolResult_UnknownTool(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := `{"data":"sensitive-data"}`
	redacted := r.RedactToolResult("unknown_tool", result, ctx)

	if strings.Contains(redacted, "sensitive-data") {
		t.Error("default policy should redact all string values")
	}
}

func TestRedactToolResult_EmptyResult(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	if r.RedactToolResult("any_tool", "", ctx) != "" {
		t.Error("empty result should remain empty")
	}
}

func TestRedactToolResult_NilContext(t *testing.T) {
	r := NewSecretRedactor()
	result := `{"data":"value"}`
	if r.RedactToolResult("any_tool", result, nil) != result {
		t.Error("nil context should return unchanged result")
	}
}

func TestRedactErrorMessage(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ctx.RegisterSecret("my-secret-token")

	errMsg := "authentication failed with token my-secret-token"
	redacted := r.RedactErrorMessage(errMsg, ctx)

	if strings.Contains(redacted, "my-secret-token") {
		t.Error("secret should be redacted from error message")
	}
	if !strings.Contains(redacted, "[SECRET_VALUE_1]") {
		t.Errorf("expected placeholder in redacted error, got %q", redacted)
	}
}

func TestRedactErrorMessage_Empty(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	if r.RedactErrorMessage("", ctx) != "" {
		t.Error("empty error should remain empty")
	}
}

func TestRedactErrorMessage_NilContext(t *testing.T) {
	r := NewSecretRedactor()
	msg := "some error"
	if r.RedactErrorMessage(msg, nil) != msg {
		t.Error("nil context should return unchanged message")
	}
}

func TestRestorePlaceholders(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ctx.RegisterSecret("real-value")

	text := "The value is [SECRET_VALUE_1]."
	restored := r.RestorePlaceholders(text, ctx)

	if restored != "The value is real-value." {
		t.Errorf("expected restoration, got %q", restored)
	}
}

func TestRestorePlaceholders_Empty(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	if r.RestorePlaceholders("", ctx) != "" {
		t.Error("empty text should remain empty")
	}
}

func TestRestorePlaceholders_NilContext(t *testing.T) {
	r := NewSecretRedactor()
	if r.RestorePlaceholders("text", nil) != "text" {
		t.Error("nil context should return unchanged text")
	}
}

func TestRestoreMapPlaceholders(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ctx.RegisterSecret("real-password")

	args := map[string]any{
		"path": "secret/data/myapp",
		"data": map[string]any{
			"password": "[SECRET_VALUE_1]",
		},
	}

	restored := r.RestoreMapPlaceholders(args, ctx)

	data, ok := restored["data"].(map[string]any)
	if !ok {
		t.Fatal("expected data to be map")
	}
	if data["password"] != "real-password" {
		t.Errorf("expected real-password, got %v", data["password"])
	}
	// Path should be unchanged.
	if restored["path"] != "secret/data/myapp" {
		t.Errorf("path should be unchanged, got %v", restored["path"])
	}
}

func TestRestoreMapPlaceholders_WithArray(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ctx.RegisterSecret("val1")
	ctx.RegisterSecret("val2")

	args := map[string]any{
		"items": []any{"[SECRET_VALUE_1]", "[SECRET_VALUE_2]", "plain"},
	}

	restored := r.RestoreMapPlaceholders(args, ctx)
	items := restored["items"].([]any)

	if items[0] != "val1" {
		t.Errorf("expected val1, got %v", items[0])
	}
	if items[1] != "val2" {
		t.Errorf("expected val2, got %v", items[1])
	}
	if items[2] != "plain" {
		t.Errorf("expected plain, got %v", items[2])
	}
}

func TestRestoreMapPlaceholders_EmptyMap(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	result := r.RestoreMapPlaceholders(nil, ctx)
	if result != nil {
		t.Error("nil map should return nil")
	}

	result = r.RestoreMapPlaceholders(map[string]any{}, ctx)
	if len(result) != 0 {
		t.Error("empty map should return empty")
	}
}

func TestRestoreMapPlaceholders_NilContext(t *testing.T) {
	r := NewSecretRedactor()
	args := map[string]any{"key": "[SECRET_VALUE_1]"}
	result := r.RestoreMapPlaceholders(args, nil)

	if result["key"] != "[SECRET_VALUE_1]" {
		t.Error("nil context should leave placeholders unchanged")
	}
}

func TestRestoreValue_NonStringTypes(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	// Non-string values should pass through unchanged.
	if restoreValue(42, ctx) != 42 {
		t.Error("int should pass through")
	}
	if restoreValue(true, ctx) != true {
		t.Error("bool should pass through")
	}
	if restoreValue(nil, ctx) != nil {
		t.Error("nil should pass through")
	}
}

func TestRedactToolResult_KVRead_ValidJSON(t *testing.T) {
	r := NewSecretRedactor()
	ctx := NewSecretContext()
	defer ctx.Destroy()

	kvResult := `{"request_id":"req-123","data":{"username":"admin","password":"hunter2"},"lease_id":"","renewable":false}`
	redacted := r.RedactToolResult("vault_kv_read", kvResult, ctx)

	// Parse the redacted JSON.
	var parsed map[string]any
	if err := json.Unmarshal([]byte(redacted), &parsed); err != nil {
		t.Fatalf("redacted result is not valid JSON: %v", err)
	}

	// Metadata keys should be preserved.
	if parsed["request_id"] != "req-123" {
		t.Errorf("request_id should be preserved, got %v", parsed["request_id"])
	}
}
