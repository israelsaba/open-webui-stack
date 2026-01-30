import logging
import re
import sqlite3
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from app.anthropic_client import AnthropicClient
from app.gemini_client import GeminiClient
from app.grok_client import GrokClient
from app.auth import BearerTokenMiddleware, parse_api_keys
from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
    PreviousCompletion
)

from .deps import cached_list_models, get_client, get_previous_completion, get_db

class RedactSecrets(logging.Filter):
    _patterns: list[re.Pattern[str]] = [
        re.compile(r'(?i)(\bauthorization\s*:\s*)([^\r\n]+)'),
        re.compile(r'(?i)(["\']authorization["\']\s*:\s*["\'])([^"\']+)(["\'])'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()

        msg = self._patterns[0].sub(r'\1[REDACTED]', msg)
        msg = self._patterns[1].sub(r'\1[REDACTED]\3', msg)

        record.msg = msg
        record.args = ()  # prevent old args from being formatted again
        return True

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

for h in logging.getLogger().handlers:
    h.addFilter(RedactSecrets())

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Anthropic, Gemini & Grok to OpenAI API Bridge",
    description="OpenAI-compatible API for Anthropic, Gemini, and Grok models",
    version="1.2.0",
)

valid_tokens = parse_api_keys(settings.api_keys)
if valid_tokens:
    app.add_middleware(BearerTokenMiddleware, valid_tokens=valid_tokens)
    logger.debug(f"Bearer token authentication enabled with {len(valid_tokens)} valid tokens")
else:
    logger.warning("No API keys configured - authentication is DISABLED")


_model_cache: set[str] | None = None

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
    
    all_models = await cached_list_models()

    if not all_models:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch models from any provider"
        )
    
    return all_models


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    client: Annotated[GeminiClient | AnthropicClient | GrokClient, Depends(get_client)],
    db: Annotated[sqlite3.Connection | None, Depends(get_db)],
    previous_completion: Annotated[PreviousCompletion| None, Depends(get_previous_completion)]
) -> ChatCompletionResponse | StreamingResponse:
    """
    Create a chat completion using Anthropic, Gemini, or Grok API.
    
    Supports both streaming and non-streaming responses.
    """
    logger.info(
        f"Chat completion request: model={request.model}, "
        f"messages={len(request.messages)}, stream={request.stream}"
    )
    
    try:
        if request.stream:
            return StreamingResponse(
                client.create_stream_completion(request, db, previous_completion),
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
        raise HTTPException(status_code=500, detail=str("error during the execution, check server logs"))




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
        access_log=False,
        timeout_graceful_shutdown=0
    )
