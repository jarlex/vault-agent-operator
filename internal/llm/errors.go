package llm

import "fmt"

// LLMError is the base error type for LLM-related failures.
type LLMError struct {
	// Code categorizes the error (e.g., "auth", "rate_limit", "service").
	Code string
	// Message is the human-readable error description.
	Message string
	// Err is the underlying error, if any.
	Err error
}

func (e *LLMError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("llm %s: %s: %v", e.Code, e.Message, e.Err)
	}
	return fmt.Sprintf("llm %s: %s", e.Code, e.Message)
}

// Unwrap returns the underlying error for errors.Is/errors.As support.
func (e *LLMError) Unwrap() error {
	return e.Err
}

// LLMAuthError indicates an authentication or authorization failure (HTTP 401/403).
type LLMAuthError struct {
	LLMError
}

// LLMRateLimitError indicates the LLM API rate limit was exceeded (HTTP 429).
type LLMRateLimitError struct {
	LLMError
	// RetryAfter is the suggested wait time in seconds, if provided.
	RetryAfter int
}

// LLMServiceError indicates an LLM server-side failure (HTTP 5xx).
type LLMServiceError struct {
	LLMError
	// StatusCode is the HTTP status code returned by the API.
	StatusCode int
}

// LLMToolCallUnsupportedError indicates the selected model does not support tool calling.
type LLMToolCallUnsupportedError struct {
	LLMError
	// Model is the model that was requested.
	Model string
}

// ClassifyError maps an HTTP status code and error to a typed LLM error.
func ClassifyError(err error, statusCode int) error {
	switch {
	case statusCode == 401 || statusCode == 403:
		return &LLMAuthError{
			LLMError: LLMError{
				Code:    "auth",
				Message: fmt.Sprintf("authentication failed (HTTP %d)", statusCode),
				Err:     err,
			},
		}
	case statusCode == 429:
		return &LLMRateLimitError{
			LLMError: LLMError{
				Code:    "rate_limit",
				Message: "rate limit exceeded",
				Err:     err,
			},
		}
	case statusCode >= 500:
		return &LLMServiceError{
			LLMError: LLMError{
				Code:    "service",
				Message: fmt.Sprintf("server error (HTTP %d)", statusCode),
				Err:     err,
			},
			StatusCode: statusCode,
		}
	default:
		return &LLMError{
			Code:    "unknown",
			Message: fmt.Sprintf("unexpected error (HTTP %d)", statusCode),
			Err:     err,
		}
	}
}
