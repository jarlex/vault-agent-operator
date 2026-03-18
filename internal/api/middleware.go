package api

import (
	"context"
	"fmt"
	"net/http"
	"runtime/debug"
	"time"

	"github.com/google/uuid"

	"github.com/jarlex/vault-agent-operator/internal/logging"
)

// RequestIDMiddleware generates a UUID for each request and attaches it to the
// response header and request context. Downstream handlers can retrieve it via
// the X-Request-ID header or from the context.
func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			requestID = uuid.New().String()
		}

		w.Header().Set("X-Request-ID", requestID)

		// Attach request-scoped logger with request_id to context.
		logger := logging.FromContext(r.Context())
		logger = logging.WithRequestID(logger, requestID)
		ctx := logging.NewContext(r.Context(), logger)
		ctx = context.WithValue(ctx, requestIDContextKey, requestID)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// LoggingMiddleware logs each HTTP request/response with method, path, status,
// duration, and optional client CN from mTLS.
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		logger := logging.FromContext(r.Context())

		// Wrap writer to capture status code.
		ww := &statusWriter{ResponseWriter: w, status: http.StatusOK}

		next.ServeHTTP(ww, r)

		duration := time.Since(start)
		event := logger.Info().
			Str("method", r.Method).
			Str("path", r.URL.Path).
			Int("status", ww.status).
			Dur("duration", duration).
			Int("bytes", ww.written)

		// Extract client CN from mTLS if available.
		if r.TLS != nil && len(r.TLS.PeerCertificates) > 0 {
			event = event.Str("client_cn", r.TLS.PeerCertificates[0].Subject.CommonName)
		}

		event.Msg("http.request")
	})
}

// TimeoutMiddleware enforces a per-request timeout by wrapping the request
// context with a deadline. If the deadline is exceeded, the context is cancelled.
func TimeoutMiddleware(timeout time.Duration) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx, cancel := context.WithTimeout(r.Context(), timeout)
			defer cancel()
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// RecoveryMiddleware catches panics from downstream handlers, logs the stack
// trace, and returns a structured 500 JSON error to the client.
func RecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rvr := recover(); rvr != nil {
				logger := logging.FromContext(r.Context())
				logger.Error().
					Str("panic", fmt.Sprintf("%v", rvr)).
					Str("stack", string(debug.Stack())).
					Msg("http.panic_recovered")

				writeError(w, http.StatusInternalServerError,
					"Internal server error", nil)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// statusWriter wraps http.ResponseWriter to capture the status code.
type statusWriter struct {
	http.ResponseWriter
	status  int
	written int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	n, err := w.ResponseWriter.Write(b)
	w.written += n
	return n, err
}

// contextKey type for request-scoped values.
type contextKeyType string

const requestIDContextKey contextKeyType = "request_id"

// RequestIDFromContext extracts the request ID from the context.
func RequestIDFromContext(ctx context.Context) string {
	if id, ok := ctx.Value(requestIDContextKey).(string); ok {
		return id
	}
	return ""
}
