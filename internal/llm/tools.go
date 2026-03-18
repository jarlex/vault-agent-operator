package llm

// NormalizeToolSchema ensures a JSON Schema for tool parameters is valid for
// OpenAI's function-calling format. This applies the same normalization
// as the Python version:
//   - Ensures "type" is "object" if missing
//   - Ensures "properties" key exists (even if empty)
//   - Preserves all other schema fields
func NormalizeToolSchema(schema map[string]any) map[string]any {
	if schema == nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}

	result := make(map[string]any, len(schema))
	for k, v := range schema {
		result[k] = v
	}

	// Ensure type is "object".
	if _, ok := result["type"]; !ok {
		result["type"] = "object"
	}

	// Ensure properties exists.
	if _, ok := result["properties"]; !ok {
		result["properties"] = map[string]any{}
	}

	return result
}

// MCPToolToOpenAI converts MCP tool metadata (name, description, schema)
// to the OpenAI function-calling Tool format. This is a helper that can
// be used by the MCP client when implementing ToolsAsOpenAIFormat().
func MCPToolToOpenAI(name, description string, inputSchema map[string]any) Tool {
	return Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        name,
			Description: description,
			Parameters:  NormalizeToolSchema(inputSchema),
		},
	}
}
