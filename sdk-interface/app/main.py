from contextlib import asynccontextmanager
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
    ModelInfo,
    PreviousCompletion
)


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Anthropic, Gemini & Grok to OpenAI API Bridge",
    description="OpenAI-compatible API for Anthropic, Gemini, and Grok models",
    version="1.2.0",
    lifespan=lifespan
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
        logger.debug(f"Fetched {len(all_models)} models "
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


def get_client(request: ChatCompletionRequest):
    model_lower = request.model.lower()
    if "gemini" in model_lower or "gemma" in model_lower or "deep-research" in model_lower:
        return gemini_client
    elif "grok" in model_lower:
        return grok_client
    else:
        return anthropic_client

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
