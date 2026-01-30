import logging
import sqlite3
from typing_extensions import Annotated
from async_lru import alru_cache
import json
import hashlib

from fastapi import Depends, Request
from app.anthropic_client import AnthropicClient, anthropic_client
from app.gemini_client import GeminiClient, gemini_client
from app.grok_client import GrokClient, grok_client
from app.models import (
    ChatCompletionRequest,
    ModelsResponse,
    PreviousCompletion
)

logger = logging.getLogger(__name__)


@alru_cache() 
async def cached_list_models() -> ModelsResponse:
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

