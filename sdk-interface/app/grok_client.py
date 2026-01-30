import logging
import time
import sqlite3
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, APIError

from app.config import settings
from app.models import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    ModelInfo,
    PreviousCompletion,
    Usage,
)

logger = logging.getLogger(__name__)


class GrokClient:
    """Client for interacting with xAI Grok API."""

    def __init__(self) -> None:
        if settings.grok_api_key:
            logger.debug("Initializing Grok client with API key")
            self.client = AsyncOpenAI(
                api_key=settings.grok_api_key.get_secret_value(),
                base_url="https://api.x.ai/v1"
            )
        else:
            logger.warning("Grok API key not configured. Grok models will be unavailable.")
            self.client = None

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
                    created=model.created,
                    owned_by=model.owned_by
                ))
            
            logger.debug(f"Successfully fetched {len(models)} models from Grok API")
            return models
        except Exception as e:
            logger.error(f"Failed to fetch models from Grok API: {e}", exc_info=True)
            raise


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

    async def create_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion."""
        if not self.client:
            raise ValueError("Grok API key not configured")

        try:
            # Pass request directly to OpenAI client
            # We convert our internal Pydantic model to dict or args
            
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
                    for msg in request.messages
                ],
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
            yield f"data: {ChatCompletionChunk(id=completion_id_meta, created=created_meta, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'[SDK] Connecting to Grok API with model {request.model}...'}, finish_reason=None)]).model_dump_json()}\n\n"
            
            stream = await self.client.chat.completions.create(**kwargs)
            
            # Send meta-reasoning: Stream started
            yield f"data: {ChatCompletionChunk(id=completion_id_meta, created=created_meta, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[SDK] Stream established, awaiting response...'}, finish_reason=None)]).model_dump_json()}\n\n"
            
            first_chunk = True
            async for chunk in stream:
                if first_chunk:
                    # Send meta-reasoning: First response received
                    yield f"data: {ChatCompletionChunk(id=completion_id_meta, created=created_meta, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[SDK] Response received, streaming content...'}, finish_reason=None)]).model_dump_json()}\n\n"
                    first_chunk = False
                # We need to convert the OpenAI chunk to our internal chunk format and then stringify
                # But wait, our internal format IS OpenAI format.
                # However, the chunk object from openai library needs to be dumped to json.
                
                # The openai library chunk object has a .model_dump_json() method if using pydantic V2 under the hood,
                # or we can construct our own.
                
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                
                # Build delta dict with all available content
                delta_dict = {}
                if hasattr(delta, "role") and delta.role:
                    delta_dict["role"] = delta.role
                if hasattr(delta, "content") and delta.content:
                    delta_dict["content"] = delta.content
                # Support reasoning_content if present (for future Grok reasoning models)
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
                
                # Yield the chunk (exclude_none handles None values automatically)
                yield f"data: {response_chunk.model_dump_json(exclude_none=True)}\n\n"
            
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error streaming Grok completion: {e}", exc_info=True)
            raise


# Global client instance
grok_client = GrokClient()
