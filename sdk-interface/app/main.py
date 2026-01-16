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


AVAILABLE_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


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
    """List available models in OpenAI format."""
    created = int(time.time())
    models = [
        ModelInfo(
            id=model_id,
            created=created,
            owned_by="anthropic"
        )
        for model_id in AVAILABLE_MODELS
    ]
    
    logger.info(f"Listing {len(models)} available models")
    return ModelsResponse(data=models)


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
    
    # Validate model
    if request.model not in AVAILABLE_MODELS:
        logger.warning(f"Unknown model requested: {request.model}")
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not found. Available models: {', '.join(AVAILABLE_MODELS)}"
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
