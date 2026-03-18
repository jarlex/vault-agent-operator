"""GET /api/v1/models — list available LLM models.

Returns the configured LLM models from the provider, indicating which is the
default and whether each supports tool calling.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_llm_provider
from src.api.schemas import ModelDetail, ModelsResponse
from src.llm.provider import LLMProvider
from src.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available models",
    description="Returns the LLM models configured for this agent.",
)
async def list_models(
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> ModelsResponse:
    """Return the list of available LLM models."""
    models = llm_provider.get_available_models()
    default_model = llm_provider._config.default_model

    available = [
        ModelDetail(
            name=m.name,
            provider=m.provider,
            model_id=m.model_id,
            supports_tool_calling=m.supports_tool_calling,
            is_default=(m.name == default_model),
        )
        for m in models
    ]

    logger.debug(
        "api.models.listed",
        model_count=len(available),
        default_model=default_model,
    )

    return ModelsResponse(
        default_model=default_model,
        available_models=available,
    )
