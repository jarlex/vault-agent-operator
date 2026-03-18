"""FastAPI dependency injection for vault-operator-agent.

Provides singleton accessors for the core components (AgentCore, MCPClient,
LLMProvider, Settings) via FastAPI's ``Depends()`` system. Components are
initialised during the application lifespan and stored in ``app.state``.

Usage in route handlers::

    from src.api.dependencies import get_agent

    @router.post("/tasks")
    async def create_task(agent: AgentCore = Depends(get_agent)):
        result = await agent.execute(...)
"""

from __future__ import annotations

from fastapi import Request

from src.agent.core import AgentCore
from src.config.settings import Settings
from src.llm.provider import LLMProvider
from src.mcp.client import MCPClient


def get_settings(request: Request) -> Settings:
    """Retrieve the application settings from app state."""
    return request.app.state.settings


def get_agent(request: Request) -> AgentCore:
    """Retrieve the AgentCore singleton from app state."""
    return request.app.state.agent


def get_mcp_client(request: Request) -> MCPClient:
    """Retrieve the MCPClient singleton from app state."""
    return request.app.state.mcp_client


def get_llm_provider(request: Request) -> LLMProvider:
    """Retrieve the LLMProvider singleton from app state."""
    return request.app.state.llm_provider


def get_start_time(request: Request) -> float:
    """Retrieve the application start time (monotonic) from app state."""
    return request.app.state.start_time
