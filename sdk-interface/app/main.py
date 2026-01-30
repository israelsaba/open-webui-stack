import logging
import re
from async_lru import alru_cache
import json
import hashlib
import sqlite3
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.anthropic_client import AnthropicClient, anthropic_client
from app.gemini_client import GeminiClient, gemini_client
from app.grok_client import GrokClient, grok_client
from app.auth import BearerTokenMiddleware, parse_api_keys
from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
    PreviousCompletion
)

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


async def get_available_models() -> set[str]:
    """Get available model IDs from APIs"""
    all_model_ids = set()
    
    try:
        anthropic_models = await anthropic_client.list_models()
        all_model_ids.update(model.id for model in anthropic_models)
    except Exception as e:
        logger.warning(f"Unable to fetch anthropic models: {e}")
    
    try:
        gemini_models = await gemini_client.list_models()
        all_model_ids.update(model.id for model in gemini_models)
    except Exception as e:
        logger.warning(f"Unable to fetch gemini models: {e}")
    
    try:
        grok_models = await grok_client.list_models()
        all_model_ids.update(model.id for model in grok_models)
    except Exception as e:
        logger.warning(f"Unable to fetch grok models: {e}")
    
    _model_cache = all_model_ids
    return _model_cache


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Anthropic, Gemini & Grok to OpenAI API Bridge",
        "docs": "/docs",
        "models": "/v1/models"
    }


@alru_cache() 
async def _cached_list_models() -> ModelsResponse:
    all_models = []
    
    try:
        anthropic_models = await anthropic_client.list_models()
        all_models.extend(anthropic_models)
        logger.debug(f"Fetched {len(anthropic_models)} Anthropic models")
    except Exception as e:
        logger.warning(f"Failed to fetch Anthropic models: {e}")
    
    try:
        gemini_models = await gemini_client.list_models()
        all_models.extend(gemini_models)
        logger.debug(f"Fetched {len(gemini_models)} Gemini models")
    except Exception as e:
        logger.warning(f"Failed to fetch Gemini models: {e}")
    
    try:
        grok_models = await grok_client.list_models()
        all_models.extend(grok_models)
        logger.debug(f"Fetched {len(grok_models)} Grok models")
    except Exception as e:
        logger.warning(f"Failed to fetch Grok models: {e}")

    return ModelsResponse(data=all_models)

@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models from all supported APIs in OpenAI format."""
    
    all_models = await _cached_list_models()

    if not all_models:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch models from any provider"
        )
    
    return all_models


@alru_cache() 
async def _cached_get_client(model_id: str):
    client = None
    
    try:
        anthropic_models = await anthropic_client.list_models()
        if any(m.id.lower() == model_id for m in anthropic_models):
            client = anthropic_client
    except Exception as e:
        logger.debug(f"Anthropic client unavailable: {e}")
    
    if not client:
        try:
            gemini_models = await gemini_client.list_models()
            if any(m.id.lower() == model_id for m in gemini_models):
                client = gemini_client
        except Exception as e:
            logger.debug(f"Gemini client unavailable: {e}")
    
    if not client:
        try:
            grok_models = await grok_client.list_models()
            if any(m.id.lower() == model_id for m in grok_models):
                client = grok_client
        except Exception as e:
            logger.debug(f"Grok client unavailable: {e}")
    
    if not client:
        raise ValueError(f"Model {model_id} not found in any available provider")
    return client

async def get_client(request: ChatCompletionRequest):
    """Get the appropriate client for the requested model."""

    return await _cached_get_client(request.model.lower())
    
def needs_previous_checking(
    request: ChatCompletionRequest,
    client: Annotated[GeminiClient | AnthropicClient | GrokClient, Depends(get_client)]
):
    if isinstance(client, GeminiClient) and "deep-research" in request.model.lower():
        return True
    return False

def get_db(
    api_request: Request,
    check_previous_completion: Annotated[bool, Depends(needs_previous_checking)]
):
    if check_previous_completion:  
        from app.db import get_db
        return get_db(api_request)
    return None

def get_previous_completion(
    check_previous_completion: Annotated[bool, Depends(needs_previous_checking)],
    db: Annotated[sqlite3.Connection | None, Depends(get_db)],
    request: ChatCompletionRequest
):
    if check_previous_completion and db:
        s = json.dumps([msg.model_dump() for msg in request.messages], sort_keys=True, separators=(',', ':'))
        h = hashlib.md5(s.encode('utf-8')).hexdigest()
        row = db.execute("""
            SELECT * 
            FROM research_hashes 
            WHERE md5 = ? AND deleted_at IS NULL 
            LIMIT 1
        """,(h,),).fetchone()

        if not row:
            import time
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            db.execute("INSERT INTO research_hashes (md5, created_at, created_by) VALUES (?, ?, ?)", (h, now, "system"))
            db.commit()
            
            row = db.execute("""
                SELECT * 
                FROM research_hashes 
                WHERE md5 = ? AND deleted_at IS NULL 
                LIMIT 1
            """,(h,),).fetchone()

        return PreviousCompletion.model_validate(dict(row))
    return None

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
