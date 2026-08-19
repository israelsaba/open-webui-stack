from .provider_gateway import provider_gateway
from .models import (
    ChatCompletionRequest,
    ModelsResponse,
)


async def list_models() -> ModelsResponse:
    return ModelsResponse(data=await provider_gateway.list_models())


async def get_client(request: ChatCompletionRequest):
    """Get the appropriate client for the requested model."""

    return provider_gateway
