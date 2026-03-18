package redaction

import (
	"fmt"
	"strings"
	"sync"
)

// SecretContext maintains a thread-safe bidirectional mapping between
// placeholder tokens and real secret values. Each request gets its own
// SecretContext, which is destroyed after the request completes.
//
// Placeholders use the format "[SECRET_VALUE_N]" where N is a monotonically
// increasing counter. This format is chosen to be unambiguous in LLM
// prompts while being easily identifiable for restoration.
type SecretContext struct {
	mu                 sync.RWMutex
	placeholderToValue map[string]string
	valueToPlaceholder map[string]string
	counter            int
	destroyed          bool
}

// NewSecretContext creates a new empty SecretContext ready for use.
func NewSecretContext() *SecretContext {
	return &SecretContext{
		placeholderToValue: make(map[string]string),
		valueToPlaceholder: make(map[string]string),
	}
}

// RegisterSecret registers a real secret value and returns its placeholder.
// If the value has already been registered, the existing placeholder is returned
// (idempotent). Returns an error if the context has been destroyed.
func (c *SecretContext) RegisterSecret(value string) (string, error) {
	if value == "" {
		return "", nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.destroyed {
		return "", fmt.Errorf("secret context has been destroyed")
	}

	// Idempotent: return existing placeholder if value was already registered.
	if placeholder, ok := c.valueToPlaceholder[value]; ok {
		return placeholder, nil
	}

	c.counter++
	placeholder := fmt.Sprintf("[SECRET_VALUE_%d]", c.counter)
	c.placeholderToValue[placeholder] = value
	c.valueToPlaceholder[value] = placeholder

	return placeholder, nil
}

// ResolvePlaceholder returns the real secret value for a given placeholder.
// The second return value indicates whether the placeholder was found.
func (c *SecretContext) ResolvePlaceholder(placeholder string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.destroyed {
		return "", false
	}

	val, ok := c.placeholderToValue[placeholder]
	return val, ok
}

// ResolveAllPlaceholders replaces all placeholder tokens in the given text
// with their corresponding real secret values. Placeholders not found in
// the context are left unchanged.
func (c *SecretContext) ResolveAllPlaceholders(text string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.destroyed || len(c.placeholderToValue) == 0 {
		return text
	}

	result := text
	for placeholder, value := range c.placeholderToValue {
		result = strings.ReplaceAll(result, placeholder, value)
	}
	return result
}

// HasPlaceholders returns true if at least one secret has been registered.
func (c *SecretContext) HasPlaceholders() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return !c.destroyed && len(c.placeholderToValue) > 0
}

// PlaceholderCount returns the number of registered secret placeholders.
func (c *SecretContext) PlaceholderCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.destroyed {
		return 0
	}
	return len(c.placeholderToValue)
}

// Destroy clears all stored secrets and marks the context as destroyed.
// After calling Destroy, all operations become no-ops or return errors.
// This should be called when the request completes to minimize the window
// during which secrets are held in memory.
func (c *SecretContext) Destroy() {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Overwrite values before clearing to reduce lingering in memory.
	for k := range c.placeholderToValue {
		c.placeholderToValue[k] = ""
	}
	for k := range c.valueToPlaceholder {
		c.valueToPlaceholder[k] = ""
	}

	c.placeholderToValue = nil
	c.valueToPlaceholder = nil
	c.destroyed = true
}

// IsDestroyed returns true if Destroy has been called.
func (c *SecretContext) IsDestroyed() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.destroyed
}

// String returns a safe representation of the SecretContext that never
// exposes secret values. This satisfies the fmt.Stringer interface.
func (c *SecretContext) String() string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.destroyed {
		return "SecretContext{destroyed=true}"
	}
	return fmt.Sprintf("SecretContext{placeholders=%d, destroyed=false}", len(c.placeholderToValue))
}

// RedactValue replaces a known secret value with its placeholder if it exists
// in the context. Returns the original string if no mapping is found.
func (c *SecretContext) RedactValue(value string) string {
	if value == "" {
		return value
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.destroyed {
		return value
	}

	if placeholder, ok := c.valueToPlaceholder[value]; ok {
		return placeholder
	}
	return value
}
