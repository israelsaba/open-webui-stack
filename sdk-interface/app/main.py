import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.anthropic_client import anthropic_client
from app.auth import BearerTokenMiddleware, parse_api_keys
from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
    ModelInfo,
)

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anthropic to OpenAI API Bridge",
    description="OpenAI-compatible API for Anthropic models",
    version="1.0.0"
)

# Add bearer token authentication middleware
valid_tokens = parse_api_keys(settings.api_keys)
if valid_tokens:
    app.add_middleware(BearerTokenMiddleware, valid_tokens=valid_tokens)
    logger.info(f"Bearer token authentication enabled with {len(valid_tokens)} valid tokens")
else:
    logger.warning("No API keys configured - authentication is DISABLED")


# Cache for model validation (fetched from Anthropic API on first request)
_model_cache: set[str] | None = None


async def get_available_models() -> set[str]:
    """Get available model IDs from Anthropic API (cached)."""
    global _model_cache
    if _model_cache is None:
        models = await anthropic_client.list_models()
        _model_cache = {model.id for model in models}
    return _model_cache


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Anthropic to OpenAI API Bridge",
        "docs": "/docs",
        "models": "/v1/models"
    }


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models from Anthropic API in OpenAI format."""
    try:
        models = await anthropic_client.list_models()
        logger.info(f"Fetched {len(models)} models from Anthropic API")
        return ModelsResponse(data=models)
    except Exception as e:
        logger.error(f"Error fetching models from Anthropic API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch models from Anthropic API: {str(e)}"
        )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    """Get a specific model by ID from Anthropic API in OpenAI format."""
    try:
        model = await anthropic_client.get_model(model_id)
        logger.info(f"Fetched model: {model_id}")
        return model
    except Exception as e:
        logger.error(f"Error fetching model {model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found: {str(e)}"
        )


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest
) -> ChatCompletionResponse | StreamingResponse:
    """
    Create a chat completion using Anthropic's API.
    
    Supports both streaming and non-streaming responses.
    """
    logger.info(
        f"Chat completion request: model={request.model}, "
        f"messages={len(request.messages)}, stream={request.stream}"
    )
    
    # Validate model against Anthropic API
    available_models = await get_available_models()
    if request.model not in available_models:
        logger.warning(f"Unknown model requested: {request.model}")
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not found. Use /v1/models to see available models."
        )
    
    try:
        if request.stream:
            return StreamingResponse(
                anthropic_client.create_stream_completion(request),
                media_type="text/event-stream"
            )
        else:
            response = await anthropic_client.create_completion(request)
            logger.info(
                f"Completion successful: tokens={response.usage.total_tokens}, "
                f"finish_reason={response.choices[0].finish_reason}"
            )
            return response
    except Exception as e:
        logger.error(f"Error creating completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level
    )
