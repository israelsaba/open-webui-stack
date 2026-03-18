import logging
import time
import sqlite3
from collections.abc import AsyncIterator
from typing import override

from openai import AsyncOpenAI

from app.config import settings
from app.connection_client import ConnectionClient
from app.models import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    Message as InterfaceMessage,
    ModelInfo,
    PreviousCompletion,
    Usage,
)

logger = logging.getLogger(__name__)


class GrokClient(ConnectionClient):
    """Client for interacting with xAI Grok API."""

    def __init__(self) -> None:
        super().__init__(provider="xAI")
        if settings.grok_api_key:
            logger.debug("Initializing Grok client with API key")
            self.client = AsyncOpenAI(
                api_key=settings.grok_api_key.get_secret_value(),
                base_url="https://api.x.ai/v1"
            )
        else:
            logger.warning("Grok API key not configured. Grok models will be unavailable.")
            self.client = None

    @override
    async def list_models(self, limit: int = 100) -> list[ModelInfo]:
        """
        Fetch available models from Grok API.
        
        Args:
            limit: Maximum number of models to fetch (default: 100)
        
        Returns:
            List of ModelInfo objects in OpenAI-compatible format
        """
        if not self.client:
            raise ValueError("Grok API key not configured")

        try:
            response = await self.client.models.list()
            
            models = []
            for model in response.data:
                models.append(ModelInfo(
                    id=model.id,
                    owned_by=model.owned_by
                ))
            
            logger.debug(f"Successfully fetched {len(models)} models from Grok API")
            return models
        except Exception as e:
            logger.error(f"Failed to fetch models from Grok API: {e}", exc_info=True)
            raise

    @override
    async def get_model(self, model_id: str) -> ModelInfo:
        """
        Fetch a specific model by ID from Grok API.
        
        Args:
            model_id: The model identifier
        
        Returns:
            ModelInfo object in OpenAI-compatible format
        """
        if not self.client:
            raise ValueError("Grok API key not configured")

        try:
            model = await self.client.models.retrieve(model_id)
            
            return ModelInfo(
                id=model.id,
                created=model.created,
                owned_by=model.owned_by
            )
        except Exception as e:
            logger.error(f"Failed to fetch model {model_id} from API: {e}", exc_info=True)
            raise ValueError(f"Model {model_id} not found")
    
    @override
    async def create_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion."""
        if not self.client:
            raise ValueError("Grok API key not configured")

        try:
            
            kwargs = {
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content} 
                    for msg in request.messages
                ],
                "stream": False,
            }
            
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
            if request.stop:
                kwargs["stop"] = request.stop
                
            response = await self.client.chat.completions.create(**kwargs)
            
            # Map response back to our internal model (though they should be identical)
            
            choice = response.choices[0]
            
            return ChatCompletionResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    ChatCompletionChoice(
                        index=choice.index,
                        message=ChatMessage(
                            role=choice.message.role, 
                            content=choice.message.content
                        ),
                        finish_reason=choice.finish_reason
                    )
                ],
                usage=Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            )
            
        except Exception as e:
            logger.error(f"Error creating Grok completion: {e}", exc_info=True)
            raise
    
    @override
    async def create_stream_completion(
        self,
        request: ChatCompletionRequest,
        db: sqlite3.Connection | None = None,
        previous_completion: PreviousCompletion | None = None
    ) -> AsyncIterator[str]:
        """Create a streaming chat completion."""
        if not self.client:
            raise ValueError("Grok API key not configured")

        try:
            kwargs = {
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content} 
                    for msg in request.messages ],
                "stream": True,
            }
            
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
            if request.stop:
                kwargs["stop"] = request.stop

            # Generate completion ID for meta-reasoning
            completion_id_meta = f"chatcmpl-{int(time.time() * 1000)}"
            created_meta = int(time.time())
            
            # Send meta-reasoning: Initiating connection
            yield f"data: {self.reasoning_content(
                request, 
                completion_id_meta, 
                InterfaceMessage.PRE_CONNECTION(params=f"{request.model}")
            ).model_dump_json()}\n\n"
            
            stream = await self.client.chat.completions.create(**kwargs)
            
            yield f"data: {self.reasoning_content(
                request, 
                completion_id_meta,
                InterfaceMessage.STARTING()
            ).model_dump_json()}\n\n"
            
            first_chunk = True
            async for chunk in stream:
                if first_chunk:
                    first_chunk = False
                
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                
                delta_dict = {}
                if hasattr(delta, "role") and delta.role:
                    delta_dict["role"] = delta.role
                if hasattr(delta, "content") and delta.content:
                    delta_dict["content"] = delta.content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    delta_dict["reasoning_content"] = delta.reasoning_content
                
                response_chunk = ChatCompletionChunk(
                    id=chunk.id,
                    created=chunk.created,
                    model=chunk.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=chunk.choices[0].index,
                            delta=delta_dict,
                            finish_reason=finish_reason
                        )
                    ]
                )
                
                yield f"data: {response_chunk.model_dump_json(exclude_none=True)}\n\n"
            
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error streaming Grok completion: {e}", exc_info=True)
            raise


grok_client = GrokClient()
