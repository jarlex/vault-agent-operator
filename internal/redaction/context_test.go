package redaction

import (
	"fmt"
	"sync"
	"testing"
)

func TestNewSecretContext(t *testing.T) {
	ctx := NewSecretContext()
	if ctx == nil {
		t.Fatal("expected non-nil context")
	}
	if ctx.HasPlaceholders() {
		t.Error("new context should have no placeholders")
	}
	if ctx.PlaceholderCount() != 0 {
		t.Errorf("expected 0, got %d", ctx.PlaceholderCount())
	}
	if ctx.IsDestroyed() {
		t.Error("new context should not be destroyed")
	}
}

func TestRegisterSecret_Basic(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	p, err := ctx.RegisterSecret("my-password")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != "[SECRET_VALUE_1]" {
		t.Errorf("expected [SECRET_VALUE_1], got %q", p)
	}
	if ctx.PlaceholderCount() != 1 {
		t.Errorf("expected count=1, got %d", ctx.PlaceholderCount())
	}
}

func TestRegisterSecret_Idempotent(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	p1, _ := ctx.RegisterSecret("secret-val")
	p2, _ := ctx.RegisterSecret("secret-val")

	if p1 != p2 {
		t.Errorf("expected same placeholder, got %q and %q", p1, p2)
	}
	if ctx.PlaceholderCount() != 1 {
		t.Errorf("expected count=1 (idempotent), got %d", ctx.PlaceholderCount())
	}
}

func TestRegisterSecret_EmptyString(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	p, err := ctx.RegisterSecret("")
	if err != nil {
		t.Errorf("unexpected error for empty string: %v", err)
	}
	if p != "" {
		t.Errorf("expected empty placeholder for empty string, got %q", p)
	}
	if ctx.PlaceholderCount() != 0 {
		t.Errorf("empty string should not create placeholder, count=%d", ctx.PlaceholderCount())
	}
}

func TestRegisterSecret_MultipleValues(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	p1, _ := ctx.RegisterSecret("val1")
	p2, _ := ctx.RegisterSecret("val2")
	p3, _ := ctx.RegisterSecret("val3")

	if p1 != "[SECRET_VALUE_1]" || p2 != "[SECRET_VALUE_2]" || p3 != "[SECRET_VALUE_3]" {
		t.Errorf("expected sequential placeholders, got %q %q %q", p1, p2, p3)
	}
	if ctx.PlaceholderCount() != 3 {
		t.Errorf("expected count=3, got %d", ctx.PlaceholderCount())
	}
}

func TestResolvePlaceholder(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ctx.RegisterSecret("my-secret")

	val, found := ctx.ResolvePlaceholder("[SECRET_VALUE_1]")
	if !found {
		t.Error("expected placeholder to be found")
	}
	if val != "my-secret" {
		t.Errorf("expected 'my-secret', got %q", val)
	}
}

func TestResolvePlaceholder_NotFound(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	_, found := ctx.ResolvePlaceholder("[SECRET_VALUE_99]")
	if found {
		t.Error("expected placeholder not to be found")
	}
}

func TestResolveAllPlaceholders_RoundTrip(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ctx.RegisterSecret("password123")
	ctx.RegisterSecret("token456")

	text := "The password is [SECRET_VALUE_1] and token is [SECRET_VALUE_2]."
	restored := ctx.ResolveAllPlaceholders(text)

	if restored != "The password is password123 and token is token456." {
		t.Errorf("round trip failed: %q", restored)
	}
}

func TestResolveAllPlaceholders_NoPlaceholders(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	text := "no secrets here"
	if ctx.ResolveAllPlaceholders(text) != text {
		t.Error("expected unchanged text")
	}
}

func TestHasPlaceholders(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	if ctx.HasPlaceholders() {
		t.Error("expected false before registering")
	}
	ctx.RegisterSecret("val")
	if !ctx.HasPlaceholders() {
		t.Error("expected true after registering")
	}
}

func TestDestroy(t *testing.T) {
	ctx := NewSecretContext()
	ctx.RegisterSecret("my-secret")
	ctx.Destroy()

	if !ctx.IsDestroyed() {
		t.Error("expected IsDestroyed=true")
	}
	if ctx.HasPlaceholders() {
		t.Error("expected HasPlaceholders=false after destroy")
	}
	if ctx.PlaceholderCount() != 0 {
		t.Errorf("expected count=0 after destroy, got %d", ctx.PlaceholderCount())
	}

	// Register should error after destroy.
	_, err := ctx.RegisterSecret("new-value")
	if err == nil {
		t.Error("expected error when registering after destroy")
	}

	// Resolve should return empty.
	_, found := ctx.ResolvePlaceholder("[SECRET_VALUE_1]")
	if found {
		t.Error("expected resolve to fail after destroy")
	}
}

func TestString(t *testing.T) {
	ctx := NewSecretContext()
	s := ctx.String()
	if s != "SecretContext{placeholders=0, destroyed=false}" {
		t.Errorf("unexpected string: %q", s)
	}

	ctx.RegisterSecret("val")
	s = ctx.String()
	if s != "SecretContext{placeholders=1, destroyed=false}" {
		t.Errorf("unexpected string: %q", s)
	}

	ctx.Destroy()
	s = ctx.String()
	if s != "SecretContext{destroyed=true}" {
		t.Errorf("unexpected string: %q", s)
	}
}

func TestRedactValue(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ctx.RegisterSecret("secret-token")

	// Known value → placeholder.
	redacted := ctx.RedactValue("secret-token")
	if redacted != "[SECRET_VALUE_1]" {
		t.Errorf("expected placeholder, got %q", redacted)
	}

	// Unknown value → unchanged.
	redacted = ctx.RedactValue("other-value")
	if redacted != "other-value" {
		t.Errorf("expected unchanged, got %q", redacted)
	}

	// Empty value → empty.
	redacted = ctx.RedactValue("")
	if redacted != "" {
		t.Errorf("expected empty, got %q", redacted)
	}
}

func TestRedactValue_AfterDestroy(t *testing.T) {
	ctx := NewSecretContext()
	ctx.RegisterSecret("secret")
	ctx.Destroy()

	// After destroy, redact should return the original value.
	if ctx.RedactValue("secret") != "secret" {
		t.Error("expected original value after destroy")
	}
}

func TestThreadSafety(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	var wg sync.WaitGroup
	const goroutines = 50

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			secret := fmt.Sprintf("secret-%d", n)
			p, err := ctx.RegisterSecret(secret)
			if err != nil {
				t.Errorf("goroutine %d: unexpected error: %v", n, err)
				return
			}
			if p == "" {
				t.Errorf("goroutine %d: empty placeholder", n)
				return
			}

			// Resolve it back.
			val, found := ctx.ResolvePlaceholder(p)
			if !found {
				t.Errorf("goroutine %d: placeholder %q not found", n, p)
				return
			}
			if val != secret {
				t.Errorf("goroutine %d: expected %q, got %q", n, secret, val)
			}
		}(i)
	}

	wg.Wait()

	if ctx.PlaceholderCount() != goroutines {
		t.Errorf("expected %d placeholders, got %d", goroutines, ctx.PlaceholderCount())
	}
}

func TestRedactKnownValues(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	ctx.RegisterSecret("mypassword")
	ctx.RegisterSecret("mytoken")

	text := "Error: invalid auth with mypassword and mytoken"
	result := ctx.redactKnownValues(text)

	if result == text {
		t.Error("expected redaction to occur")
	}
	if result != "Error: invalid auth with [SECRET_VALUE_1] and [SECRET_VALUE_2]" {
		t.Errorf("unexpected redaction result: %q", result)
	}
}

func TestRedactKnownValues_NoSecrets(t *testing.T) {
	ctx := NewSecretContext()
	defer ctx.Destroy()

	text := "no secrets here"
	if ctx.redactKnownValues(text) != text {
		t.Error("expected unchanged text")
	}
}

func TestRedactKnownValues_AfterDestroy(t *testing.T) {
	ctx := NewSecretContext()
	ctx.RegisterSecret("secret")
	ctx.Destroy()

	text := "contains secret"
	if ctx.redactKnownValues(text) != text {
		t.Error("expected unchanged text after destroy")
	}
}
