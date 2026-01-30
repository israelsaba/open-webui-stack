import logging
import json
import hashlib
import sqlite3
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
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

from openai.types.model import Model as ModelSchema 

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    
    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host if request.client else "unknown"
        client_port = request.client.port if request.client else "unknown"
        
        if settings.detailed_request_logging:
            body_bytes = await request.body()
            
            log_details = {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client": f"{client_host}:{client_port}",
            }
            
            if body_bytes:
                try:
                    body_str = body_bytes.decode('utf-8')
                    try:
                        log_details["body"] = json.loads(body_str)
                    except json.JSONDecodeError:
                        log_details["body"] = body_str
                except UnicodeDecodeError:
                    log_details["body"] = f"<binary data: {len(body_bytes)} bytes>"
        response = await call_next(request)
        
        log_msg = f'{client_host}:{client_port} - "{request.method} {request.url.path} HTTP/1.1" {response.status_code}'
        
        if response.status_code >= 500:
            logger.error(log_msg)
        elif response.status_code >= 400:
            logger.warning(log_msg)
        else:
            logger.debug(log_msg)
        
        return response

app = FastAPI(
    title="Anthropic, Gemini & Grok to OpenAI API Bridge",
    description="OpenAI-compatible API for Anthropic, Gemini, and Grok models",
    version="1.2.0",
)

app.add_middleware(AccessLogMiddleware)

valid_tokens = parse_api_keys(settings.api_keys)
if valid_tokens:
    app.add_middleware(BearerTokenMiddleware, valid_tokens=valid_tokens)
    logger.debug(f"Bearer token authentication enabled with {len(valid_tokens)} valid tokens")
else:
    logger.warning("No API keys configured - authentication is DISABLED")


_model_cache: set[str] | None = None


async def get_available_models() -> set[str]:
    """Get available model IDs from APIs (cached)."""
    global _model_cache
    if _model_cache is None:
        all_model_ids = set()
        
        try:
            anthropic_models = await anthropic_client.list_models()
            all_model_ids.update(model.id for model in anthropic_models)
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models for cache: {e}")
        
        try:
            gemini_models = await gemini_client.list_models()
            all_model_ids.update(model.id for model in gemini_models)
        except Exception as e:
            logger.warning(f"Failed to fetch Gemini models for cache: {e}")
        
        try:
            grok_models = await grok_client.list_models()
            all_model_ids.update(model.id for model in grok_models)
        except Exception as e:
            logger.warning(f"Failed to fetch Grok models for cache: {e}")
        
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


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models from all supported APIs in OpenAI format."""
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
    
    if not all_models:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch models from any provider"
        )
    
    logger.debug(f"Total models fetched: {len(all_models)}")
    return ModelsResponse(data=all_models)


async def get_client(request: ChatCompletionRequest):
    """Get the appropriate client for the requested model."""
    model_id = request.model.lower()
    
    # Check cache first - use model_id as cache key
    # Since @cache doesn't work with async, we'll use a simple dict cache
    global _client_cache
    if not hasattr(get_client, '_cache'):
        get_client._cache = {}
    
    # Return cached client if available
    if model_id in get_client._cache:
        logger.debug(f"Cache hit for model: {model_id}")
        return get_client._cache[model_id]
    
    logger.debug(f"Cache miss for model: {model_id}, determining client...")
    
    # Get available models from cache
    available_models = await get_available_models()
    
    # Check if model exists in available models
    if model_id not in available_models:
        # Refresh cache and try again
        global _model_cache
        _model_cache = None
        available_models = await get_available_models()
        
        if model_id not in available_models:
            raise ValueError(f"Model {model_id} not found in any provider")
    
    # Try each client to see which one has this model
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
    
    # Cache the result
    get_client._cache[model_id] = client
    logger.debug(f"Cached client for model: {model_id}")
    
    return client



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
        access_log=False,
        timeout_graceful_shutdown=0
    )
