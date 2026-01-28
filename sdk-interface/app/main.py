import time
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.anthropic_client import anthropic_client
from app.gemini_client import gemini_client
from app.grok_client import grok_client
from app.auth import BearerTokenMiddleware, parse_api_keys
from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
    ModelInfo,
)
from app.deep_research import create_deep_research_stream

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests with appropriate log levels based on status code."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Log with appropriate level based on status code
        client_host = request.client.host if request.client else "unknown"
        client_port = request.client.port if request.client else "unknown"
        log_msg = f'{client_host}:{client_port} - "{request.method} {request.url.path} HTTP/1.1" {response.status_code}'
        
        if response.status_code >= 500:
            logger.error(log_msg)
        elif response.status_code >= 400:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        return response

app = FastAPI(
    title="Anthropic, Gemini & Grok to OpenAI API Bridge",
    description="OpenAI-compatible API for Anthropic, Gemini, and Grok models",
    version="1.2.0"
)

# Add access log middleware
app.add_middleware(AccessLogMiddleware)

# Add bearer token authentication middleware
valid_tokens = parse_api_keys(settings.api_keys)
if valid_tokens:
    app.add_middleware(BearerTokenMiddleware, valid_tokens=valid_tokens)
    logger.info(f"Bearer token authentication enabled with {len(valid_tokens)} valid tokens")
else:
    logger.warning("No API keys configured - authentication is DISABLED")


# Cache for model validation (fetched from APIs on first request)
_model_cache: set[str] | None = None


async def get_available_models() -> set[str]:
    """Get available model IDs from APIs (cached)."""
    global _model_cache
    if _model_cache is None:
        anthropic_models = await anthropic_client.list_models()
        gemini_models = await gemini_client.list_models()
        grok_models = await grok_client.list_models()
        _model_cache = {model.id for model in anthropic_models} | \
                       {model.id for model in gemini_models} | \
                       {model.id for model in grok_models}
    return _model_cache


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Anthropic, Gemini & Grok to OpenAI API Bridge",
        "docs": "/docs",
        "models": "/v1/models"
    }


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models from all supported APIs in OpenAI format."""
    try:
        anthropic_models = await anthropic_client.list_models()
        gemini_models = await gemini_client.list_models()
        grok_models = await grok_client.list_models()
        
        all_models = anthropic_models + gemini_models + grok_models
        logger.info(f"Fetched {len(all_models)} models "
                   f"({len(anthropic_models)} Anthropic, "
                   f"{len(gemini_models)} Gemini, "
                   f"{len(grok_models)} Grok)")
        return ModelsResponse(data=all_models)
    except Exception as e:
        logger.error(f"Error fetching models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch models: {str(e)}"
        )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    """Get a specific model by ID from any provider in OpenAI format."""
    try:
        # Check cache or prefixes to decide where to look first if possible
        try:
            return await anthropic_client.get_model(model_id)
        except ValueError:
            try:
                return await gemini_client.get_model(model_id)
            except ValueError:
                try:
                    return await grok_client.get_model(model_id)
                except ValueError:
                    raise ValueError(f"Model {model_id} not found in any provider")

    except ValueError as e:
        logger.warning(f"Model {model_id} not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching model {model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching model {model_id}: {str(e)}"
        )


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest
) -> ChatCompletionResponse | StreamingResponse:
    """
    Create a chat completion using Anthropic, Gemini, or Grok API.
    
    Supports both streaming and non-streaming responses.
    """
    logger.info(
        f"Chat completion request: model={request.model}, "
        f"messages={len(request.messages)}, stream={request.stream}"
    )
    
    # Validate model
    available_models = await get_available_models()
    if request.model not in available_models:
        global _model_cache
        _model_cache = None
        available_models = await get_available_models()
        
        if request.model not in available_models:
            logger.warning(f"Unknown model requested: {request.model}")
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} not found. Use /v1/models to see available models."
            )
    
    # Determine provider
    model_lower = request.model.lower()
    if "gemini" in model_lower:
        client = gemini_client
    elif "grok" in model_lower:
        client = grok_client
    else:
        client = anthropic_client
    
    try:
        if request.stream:
            return StreamingResponse(
                client.create_stream_completion(request),
                media_type="text/event-stream"
            )
        else:
            response = await client.create_completion(request)
            logger.info(
                f"Completion successful: tokens={response.usage.total_tokens}, "
                f"finish_reason={response.choices[0].finish_reason}"
            )
            return response
    except Exception as e:
        logger.error(f"Error creating completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/deep-research")
async def create_deep_research(
    request: ChatCompletionRequest
) -> StreamingResponse:
    """
    Deep Research endpoint using Gemini Thinking models.
    
    Provides comprehensive, research-style responses with extensive reasoning.
    Only works with Gemini thinking models (gemini-2.0-flash-thinking-exp, etc.).
    
    Always streams responses with detailed reasoning_content showing the research process.
    """
    logger.info(
        f"Deep Research request: model={request.model}, "
        f"messages={len(request.messages)}"
    )
    
    # Validate that a Gemini thinking model is being used
    if "gemini" not in request.model.lower() or "thinking" not in request.model.lower():
        raise HTTPException(
            status_code=400,
            detail=f"Deep Research requires a Gemini Thinking model (e.g., gemini-2.0-flash-thinking-exp). "
                   f"Got: {request.model}"
        )
    
    # Validate model availability
    available_models = await get_available_models()
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not found. Use /v1/models to see available models."
        )
    
    try:
        return StreamingResponse(
            create_deep_research_stream(request),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in deep research: {e}", exc_info=True)
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
        log_level=settings.log_level,
        access_log=False  # Disable default access logs (we use custom middleware)
    )
