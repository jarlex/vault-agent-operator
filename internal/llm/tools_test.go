package llm

import (
	"testing"
)

func TestNormalizeToolSchema_Nil(t *testing.T) {
	result := NormalizeToolSchema(nil)

	if result["type"] != "object" {
		t.Errorf("expected type=object, got %v", result["type"])
	}
	props, ok := result["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties to be map[string]any")
	}
	if len(props) != 0 {
		t.Errorf("expected empty properties, got %v", props)
	}
}

func TestNormalizeToolSchema_MissingType(t *testing.T) {
	schema := map[string]any{
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
		},
	}
	result := NormalizeToolSchema(schema)

	if result["type"] != "object" {
		t.Errorf("expected type=object, got %v", result["type"])
	}
	// Properties should be preserved.
	props := result["properties"].(map[string]any)
	if _, ok := props["path"]; !ok {
		t.Error("expected 'path' property to be preserved")
	}
}

func TestNormalizeToolSchema_MissingProperties(t *testing.T) {
	schema := map[string]any{
		"type": "object",
	}
	result := NormalizeToolSchema(schema)

	if result["type"] != "object" {
		t.Errorf("expected type=object, got %v", result["type"])
	}
	if _, ok := result["properties"]; !ok {
		t.Error("expected properties key to be added")
	}
}

func TestNormalizeToolSchema_Complete(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
		},
		"required": []any{"path"},
	}
	result := NormalizeToolSchema(schema)

	if result["type"] != "object" {
		t.Errorf("expected type=object, got %v", result["type"])
	}
	// Required should be preserved.
	req, ok := result["required"].([]any)
	if !ok {
		t.Fatal("expected required to be preserved as []any")
	}
	if len(req) != 1 || req[0] != "path" {
		t.Errorf("expected required=[path], got %v", req)
	}
}

func TestNormalizeToolSchema_DoesNotMutateInput(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"key": map[string]any{"type": "string"},
		},
	}
	result := NormalizeToolSchema(schema)

	// Mutating result should not affect original.
	result["extra"] = "added"
	if _, ok := schema["extra"]; ok {
		t.Error("NormalizeToolSchema should return a copy, not mutate input")
	}
}

func TestMCPToolToOpenAI_Basic(t *testing.T) {
	tool := MCPToolToOpenAI(
		"vault_kv_read",
		"Read a secret from Vault KV",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string"},
			},
			"required": []any{"path"},
		},
	)

	if tool.Type != "function" {
		t.Errorf("expected type=function, got %q", tool.Type)
	}
	if tool.Function.Name != "vault_kv_read" {
		t.Errorf("expected name=vault_kv_read, got %q", tool.Function.Name)
	}
	if tool.Function.Description != "Read a secret from Vault KV" {
		t.Errorf("unexpected description: %q", tool.Function.Description)
	}
	if tool.Function.Parameters["type"] != "object" {
		t.Errorf("expected parameters.type=object, got %v", tool.Function.Parameters["type"])
	}
	props := tool.Function.Parameters["properties"].(map[string]any)
	if _, ok := props["path"]; !ok {
		t.Error("expected path property to be preserved")
	}
}

func TestMCPToolToOpenAI_NilSchema(t *testing.T) {
	tool := MCPToolToOpenAI("test_tool", "A test tool", nil)

	if tool.Function.Parameters["type"] != "object" {
		t.Errorf("expected type=object for nil schema, got %v", tool.Function.Parameters["type"])
	}
	props, ok := tool.Function.Parameters["properties"].(map[string]any)
	if !ok || len(props) != 0 {
		t.Errorf("expected empty properties for nil schema, got %v", tool.Function.Parameters["properties"])
	}
}

func TestMCPToolToOpenAI_EmptySchema(t *testing.T) {
	tool := MCPToolToOpenAI("no_args_tool", "No arguments", map[string]any{})

	if tool.Function.Parameters["type"] != "object" {
		t.Errorf("expected type=object, got %v", tool.Function.Parameters["type"])
	}
	if _, ok := tool.Function.Parameters["properties"]; !ok {
		t.Error("expected properties key to be added")
	}
}
