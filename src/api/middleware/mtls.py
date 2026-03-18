"""mTLS certificate validation middleware for vault-operator-agent.

Extracts the client certificate from the TLS connection, validates it against
the configured CA, and extracts the Common Name (CN) for logging and audit.

The health endpoint (``/api/v1/health``) is excluded from mTLS enforcement
to allow infrastructure health probes without client certificates.

When ``MTLSConfig.enabled`` is ``False`` (development mode), the middleware
is a pass-through that logs a warning.

Implementation notes:
    Uvicorn handles the TLS handshake and ``ssl.CERT_REQUIRED`` enforcement
    at the transport layer. By the time a request reaches this middleware,
    the TLS handshake has already succeeded — meaning the client cert is
    already validated by the SSL context. This middleware performs the
    *application-level* extraction of the client identity (CN) from the
    cert for logging/audit purposes.

    When mTLS is enabled at the Uvicorn layer, requests without a valid
    client cert are rejected at the TLS handshake — they never reach
    FastAPI or this middleware at all.
"""

from __future__ import annotations

from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.config.models import MTLSConfig
from src.logging import get_logger

logger = get_logger(__name__)

# Paths exempt from mTLS identity extraction logging (health probes)
_EXEMPT_PATHS: set[str] = {"/api/v1/health"}


class MTLSMiddleware(BaseHTTPMiddleware):
    """Middleware that extracts client certificate identity for audit logging.

    When mTLS is enabled at the Uvicorn level (``ssl.CERT_REQUIRED``),
    the TLS layer already validates the client certificate. This middleware
    extracts the CN (Common Name) from the validated cert and attaches it
    to the request state for downstream logging.

    When mTLS is disabled (dev mode), the middleware passes through without
    checking certificates and sets ``client_cn`` to ``"anonymous"``.

    Parameters
    ----------
    app:
        The ASGI application.
    config:
        mTLS configuration.
    """

    def __init__(self, app: Any, config: MTLSConfig) -> None:
        super().__init__(app)
        self._config = config

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request, extracting client identity if available."""
        # Skip certificate extraction for exempt paths (health endpoint)
        if request.url.path in _EXEMPT_PATHS:
            request.state.client_cn = "health-probe"
            return await call_next(request)

        if not self._config.enabled:
            # Dev mode — no mTLS enforcement
            request.state.client_cn = "anonymous"
            return await call_next(request)

        # Extract client certificate from the TLS transport
        client_cn = self._extract_client_cn(request)
        request.state.client_cn = client_cn

        logger.debug(
            "mtls.client_identified",
            client_cn=client_cn,
            path=request.url.path,
            method=request.method,
        )

        return await call_next(request)

    @staticmethod
    def _extract_client_cn(request: Request) -> str:
        """Extract the Common Name (CN) from the client's TLS certificate.

        The certificate is available through the ASGI transport's
        ``get_extra_info("peercert")`` or ``get_extra_info("ssl_object")``.

        Returns ``"unknown"`` if the CN cannot be extracted (this should
        not happen when ``ssl.CERT_REQUIRED`` is set at the Uvicorn level,
        because the handshake would have failed).
        """
        try:
            # Uvicorn exposes the transport via scope["transport"]
            transport = request.scope.get("transport")
            if transport is None:
                return "unknown"

            # Get the peer certificate dict
            ssl_object = transport.get_extra_info("ssl_object")
            if ssl_object is None:
                return "unknown"

            peercert = ssl_object.getpeercert()
            if peercert is None:
                return "unknown"

            # Extract CN from the subject
            subject = peercert.get("subject", ())
            for rdn in subject:
                for attr_type, attr_value in rdn:
                    if attr_type == "commonName":
                        return attr_value

        except Exception as exc:
            logger.warning(
                "mtls.cn_extraction_failed",
                error=str(exc),
                exc_type=type(exc).__name__,
            )

        return "unknown"
