import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, MessageStreamEvent

from app.config import settings
from app.models import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    ModelInfo,
    Usage,
)

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Client for interacting with Anthropic API."""
    
    def __init__(self) -> None:
        api_key = settings.anthropic_api_key.get_secret_value()
        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
    
    async def list_models(self, limit: int = 100) -> list[ModelInfo]:
        """
        Fetch available models from Anthropic API.
        
        Args:
            limit: Maximum number of models to fetch (default: 100)
        
        Returns:
            List of ModelInfo objects in OpenAI-compatible format
        """
        try:
            # Try to use the models API if available
            if hasattr(self.async_client, 'models'):
                response = await self.async_client.models.list(limit=limit)
                
                # Convert Anthropic format to OpenAI-compatible format
                models = []
                for model in response.data:
                    # Parse ISO 8601 datetime and convert to Unix timestamp
                    try:
                        created_at_str = str(model.created_at)
                        # Handle both datetime objects and strings
                        if 'T' in created_at_str or '-' in created_at_str:
                            created_timestamp = int(datetime.fromisoformat(created_at_str.replace('Z', '+00:00')).timestamp())
                        else:
                            # If it's already a timestamp
                            created_timestamp = int(created_at_str)
                    except (ValueError, AttributeError):
                        # Fallback to current time if parsing fails
                        created_timestamp = int(time.time())
                    
                    models.append(ModelInfo(
                        id=model.id,
                        created=created_timestamp,
                        owned_by="anthropic"
                    ))
                
                logger.info(f"Successfully fetched {len(models)} models from Anthropic API")
                return models
            else:
                logger.warning("Models API not available in this SDK version, using hardcoded list")
                return self._get_hardcoded_models()
        except Exception as e:
            logger.warning(f"Failed to fetch models from Anthropic API: {e}, using hardcoded list")
            return self._get_hardcoded_models()
    
    @staticmethod
    def _get_hardcoded_models() -> list[ModelInfo]:
        """
        Return a hardcoded list of available models as fallback.
        
        Returns:
            List of ModelInfo objects for known Anthropic models
        """
        base_timestamp = int(time.time())
        
        # Hardcoded list of known Anthropic models (updated from actual API)
        model_ids = [
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
        ]
        
        return [
            ModelInfo(
                id=model_id,
                created=base_timestamp,
                owned_by="anthropic"
            )
            for model_id in model_ids
        ]
    
    async def get_model(self, model_id: str) -> ModelInfo:
        """
        Fetch a specific model by ID from Anthropic API.
        
        Args:
            model_id: The model identifier or alias
        
        Returns:
            ModelInfo object in OpenAI-compatible format
        """
        try:
            if hasattr(self.async_client, 'models'):
                response = await self.async_client.models.retrieve(model_id)
                
                # Parse ISO 8601 datetime and convert to Unix timestamp
                created_timestamp = int(datetime.fromisoformat(response.created_at.replace('Z', '+00:00')).timestamp())
                
                return ModelInfo(
                    id=response.id,
                    created=created_timestamp,
                    owned_by="anthropic"
                )
            else:
                # Fallback: return model info from hardcoded list
                hardcoded_models = self._get_hardcoded_models()
                for model in hardcoded_models:
                    if model.id == model_id:
                        return model
                raise ValueError(f"Model {model_id} not found")
        except Exception as e:
            logger.warning(f"Failed to fetch model {model_id} from API: {e}")
            # Fallback: return model info from hardcoded list
            hardcoded_models = self._get_hardcoded_models()
            for model in hardcoded_models:
                if model.id == model_id:
                    return model
            raise ValueError(f"Model {model_id} not found")
    
    @staticmethod
    def _convert_messages(messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, str]]]:
        """
        Convert OpenAI-style messages to Anthropic format.
        
        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_message: str | None = None
        anthropic_messages: list[dict[str, str]] = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system_message, anthropic_messages
    
    async def create_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion."""
        system_message, anthropic_messages = self._convert_messages(request.messages)
        
        # Build kwargs for Anthropic API
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or 4096,
        }
        
        if system_message:
            kwargs["system"] = system_message
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
        
        response: Message = await self.async_client.messages.create(**kwargs)
        
        # Convert to OpenAI format
        completion_id = f"chatcmpl-{response.id}"
        created = int(time.time())
        
        content = ""
        if response.content:
            content = " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )
        
        # Map Anthropic stop_reason to OpenAI finish_reason
        stop_reason_str = str(response.stop_reason) if response.stop_reason else "end_turn"
        if stop_reason_str == "max_tokens":
            mapped_finish_reason: str = "length"
        elif stop_reason_str in ("end_turn", "stop_sequence"):
            mapped_finish_reason = "stop"
        else:
            mapped_finish_reason = "stop"
        
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=mapped_finish_reason  # type: ignore
                )
            ],
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
        )
    
    async def create_stream_completion(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Create a streaming chat completion."""
        system_message, anthropic_messages = self._convert_messages(request.messages)
        
        # Build kwargs for Anthropic API
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or 4096,
        }
        
        if system_message:
            kwargs["system"] = system_message
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
        
        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        
        async with self.async_client.messages.stream(**kwargs) as stream:
            async for event in stream:
                chunk = self._convert_stream_event(
                    event, completion_id, created, request.model
                )
                if chunk:
                    yield f"data: {chunk.model_dump_json()}\n\n"
        
        yield "data: [DONE]\n\n"
    
    @staticmethod
    def _convert_stream_event(
        event: MessageStreamEvent,
        completion_id: str,
        created: int,
        model: str
    ) -> ChatCompletionChunk | None:
        """Convert Anthropic streaming event to OpenAI format."""
        if event.type == "content_block_delta":
            if hasattr(event.delta, "text"):
                return ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"role": "assistant", "content": event.delta.text},
                            finish_reason=None
                        )
                    ]
                )
        elif event.type == "message_stop":
            return ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={},
                        finish_reason="stop"
                    )
                ]
            )
        
        return None


# Global client instance
anthropic_client = AnthropicClient()
