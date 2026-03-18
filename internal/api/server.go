package api

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/logging"
)

// ServerConfig holds the configuration for the HTTP server.
type ServerConfig struct {
	Host           string
	Port           int
	RequestTimeout time.Duration
	MTLSEnabled    bool
	CACertPath     string
	ServerCertPath string
	ServerKeyPath  string
	Version        string
}

// Server wraps an http.Server with chi routing and middleware.
type Server struct {
	httpServer *http.Server
	router     chi.Router
	handler    *Handler
	config     ServerConfig
	logger     zerolog.Logger
}

// NewServer creates a configured HTTP server with all routes and middleware.
func NewServer(cfg ServerConfig, handler *Handler, logger zerolog.Logger) *Server {
	r := chi.NewRouter()

	// Middleware chain (order matters — outermost first).
	r.Use(RecoveryMiddleware)
	r.Use(RequestIDMiddleware)
	r.Use(injectLoggerMiddleware(logger))
	r.Use(LoggingMiddleware)
	if cfg.RequestTimeout > 0 {
		r.Use(TimeoutMiddleware(cfg.RequestTimeout))
	}

	// Register routes.
	r.Route("/api/v1", func(r chi.Router) {
		r.Post("/tasks", handler.CreateTask)
		r.Get("/health", handler.Health)
		r.Get("/models", handler.Models)
	})

	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)

	srv := &http.Server{
		Addr:              addr,
		Handler:           r,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	return &Server{
		httpServer: srv,
		router:     r,
		handler:    handler,
		config:     cfg,
		logger:     logger.With().Str("component", "server").Logger(),
	}
}

// Start begins listening for HTTP requests. If mTLS is enabled, TLS is
// configured with client certificate verification. This method blocks until
// the server is shut down or an error occurs.
func (s *Server) Start() error {
	if s.config.MTLSEnabled {
		tlsConfig, err := s.buildTLSConfig()
		if err != nil {
			return fmt.Errorf("tls config: %w", err)
		}
		s.httpServer.TLSConfig = tlsConfig

		s.logger.Info().
			Str("addr", s.httpServer.Addr).
			Bool("mtls", true).
			Msg("server.start")

		return s.httpServer.ListenAndServeTLS(
			s.config.ServerCertPath,
			s.config.ServerKeyPath,
		)
	}

	s.logger.Info().
		Str("addr", s.httpServer.Addr).
		Bool("mtls", false).
		Msg("server.start")

	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server, waiting for in-flight requests
// to complete up to the given context deadline.
func (s *Server) Shutdown(ctx context.Context) error {
	s.logger.Info().Msg("server.shutdown")
	return s.httpServer.Shutdown(ctx)
}

// Addr returns the server's listen address. Useful after Start() when using
// port 0 for testing.
func (s *Server) Addr() net.Addr {
	// This only works if the server has been started with a listener.
	// For production use, the address is known from config.
	return nil
}

// buildTLSConfig creates a TLS configuration for mTLS with client cert
// verification.
func (s *Server) buildTLSConfig() (*tls.Config, error) {
	caCert, err := os.ReadFile(s.config.CACertPath)
	if err != nil {
		return nil, fmt.Errorf("read CA cert %s: %w", s.config.CACertPath, err)
	}

	caCertPool := x509.NewCertPool()
	if !caCertPool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA certificate from %s", s.config.CACertPath)
	}

	return &tls.Config{
		ClientAuth: tls.RequireAndVerifyClientCert,
		ClientCAs:  caCertPool,
		MinVersion: tls.VersionTLS12,
	}, nil
}

// injectLoggerMiddleware creates a middleware that injects the base logger
// into the request context. This runs early so that downstream middleware
// (like RequestIDMiddleware) can retrieve and enhance it.
func injectLoggerMiddleware(logger zerolog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx := logging.NewContext(r.Context(), logger)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}
