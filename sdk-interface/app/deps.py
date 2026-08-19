from .openrouter_client import openrouter_client
from .models import (
    ChatCompletionRequest,
    ModelsResponse,
)


async def list_models() -> ModelsResponse:
    return ModelsResponse(data=await openrouter_client.list_models())


async def get_client(request: ChatCompletionRequest):
    """Get the appropriate client for the requested model."""

    return openrouter_client
