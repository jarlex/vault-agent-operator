"""POST /api/v1/tasks — submit a natural-language task to the agent.

This is the primary endpoint. It validates the request, delegates to
``AgentCore.execute()``, and builds a structured ``TaskResponse`` including
unredacted secret data for the consumer.

Every request is assigned a unique ``X-Request-ID`` header that is included
in all log entries and returned in the response headers.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from src.agent.core import AgentCore
from src.api.dependencies import get_agent
from src.api.schemas import (
    ErrorResponse,
    TaskRequest,
    TaskResponse,
    ToolCallDetail,
)
from src.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/tasks",
    response_model=TaskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error or invalid model"},
        503: {"model": ErrorResponse, "description": "Agent dependency unavailable"},
    },
    summary="Submit a task to the agent",
    description="Send a natural-language prompt and receive the agent's response.",
)
async def create_task(
    body: TaskRequest,
    request: Request,
    response: Response,
    agent: AgentCore = Depends(get_agent),
) -> TaskResponse | ErrorResponse:
    """Execute a task through the agent reasoning loop."""
    # Generate and bind request ID
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    # Bind request_id to structlog context for all downstream logs
    structlog.contextvars.bind_contextvars(request_id=request_id)

    client_cn = getattr(request.state, "client_cn", "unknown")

    logger.info(
        "api.task.received",
        prompt_length=len(body.prompt),
        model=body.model,
        has_secret_data=body.secret_data is not None,
        client_cn=client_cn,
    )

    start_time = time.monotonic()

    try:
        result = await agent.execute(
            prompt=body.prompt,
            model=body.model,
            secret_data=body.secret_data,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.error(
            "api.task.unhandled_error",
            error=str(exc),
            exc_type=type(exc).__name__,
            duration_ms=duration_ms,
        )
        error_body = ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred while processing the task.",
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content=error_body.model_dump(),
            headers={"X-Request-ID": request_id},
        )
    finally:
        structlog.contextvars.unbind_contextvars("request_id")

    duration_ms = int((time.monotonic() - start_time) * 1000)

    # Map AgentResult error codes to HTTP status codes
    if result.status == "error":
        http_status = _error_code_to_http_status(result.error_code)
        response.status_code = http_status

    # Build tool call audit trail
    tool_calls = [
        ToolCallDetail(
            tool_name=tc.tool_name,
            arguments=tc.arguments,
            result=tc.result,
            is_error=tc.is_error,
            duration_ms=tc.duration_ms,
        )
        for tc in result.tool_calls
    ]

    # Build unredacted data for consumer (real secret values)
    unredacted_data: list[dict[str, Any]] | None = None
    if result.unredacted_responses:
        unredacted_data = result.unredacted_responses

    logger.info(
        "api.task.completed",
        status=result.status,
        model_used=result.model_used,
        tool_call_count=len(tool_calls),
        duration_ms=duration_ms,
        error_code=result.error_code,
        client_cn=client_cn,
    )

    return TaskResponse(
        status=result.status,
        result=result.result,
        model_used=result.model_used,
        tool_calls=tool_calls,
        duration_ms=duration_ms,
        error=result.result if result.status == "error" else None,
        unredacted_data=unredacted_data,
    )


def _error_code_to_http_status(error_code: str | None) -> int:
    """Map agent error codes to HTTP status codes.

    Returns
    -------
    int
        The HTTP status code for the given error code.
    """
    mapping: dict[str, int] = {
        "llm_auth": 502,
        "llm_rate_limit": 429,
        "llm_service": 503,
        "llm_tool_unsupported": 400,
        "llm_timeout": 504,
        "mcp_error": 503,
        "mcp_connection": 503,
        "empty_response": 502,
        "internal_error": 500,
        "max_iterations": 200,  # Partial result — still 200
    }
    return mapping.get(error_code or "", 500)
