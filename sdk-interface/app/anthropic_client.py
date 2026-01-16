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
        response = await self.async_client.models.list(limit=limit)
        
        # Convert Anthropic format to OpenAI-compatible format
        models = []
        for model in response.data:
            # Parse ISO 8601 datetime and convert to Unix timestamp
            created_timestamp = int(datetime.fromisoformat(model.created_at.replace('Z', '+00:00')).timestamp())
            
            models.append(ModelInfo(
                id=model.id,
                created=created_timestamp,
                owned_by="anthropic"
            ))
        
        return models
    
    async def get_model(self, model_id: str) -> ModelInfo:
        """
        Fetch a specific model by ID from Anthropic API.
        
        Args:
            model_id: The model identifier or alias
        
        Returns:
            ModelInfo object in OpenAI-compatible format
        """
        response = await self.async_client.models.retrieve(model_id)
        
        # Parse ISO 8601 datetime and convert to Unix timestamp
        created_timestamp = int(datetime.fromisoformat(response.created_at.replace('Z', '+00:00')).timestamp())
        
        return ModelInfo(
            id=response.id,
            created=created_timestamp,
            owned_by="anthropic"
        )
    
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
        
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop" if response.stop_reason == "end_turn" else response.stop_reason
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
