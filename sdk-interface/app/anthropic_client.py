import logging
import re
import time
import sqlite3
from datetime import datetime
from collections.abc import AsyncIterator
from typing import Any
from typing_extensions import override

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message
from anthropic.types.message_stream_event import MessageStreamEvent

from .config import settings
from .connection_client import ConnectionClient
from .models import (
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


class AnthropicClient(ConnectionClient):
    """Client for interacting with Anthropic API."""
    
    def __init__(self) -> None:
        super().__init__(provider="Anthropic")
        if settings.anthropic_api_key:
            api_key = settings.anthropic_api_key.get_secret_value()
            self.client = Anthropic(api_key=api_key)
            self.async_client = AsyncAnthropic(api_key=api_key)
        else:
            logger.warning("Anthropic API key not configured. Anthropic models will be unavailable.")
            self.client = None
            self.async_client = None
    
    @override
    async def list_models(self, limit: int = 100) -> list[ModelInfo]:
        """
        Fetch available models from Anthropic API.
        
        Args:
            limit: Maximum number of models to fetch (default: 100)
        
        Returns:
            List of ModelInfo objects in OpenAI-compatible format
        """

        if hasattr(self.async_client, 'models'):
            response = await self.async_client.models.list(limit=limit)
            
            models = []
            for model in response.data:
                
                models.append(ModelInfo(
                    id=model.id,
                    owned_by="anthropic"
                ))
            
            logger.debug(f"Successfully fetched {len(models)} models from Anthropic API")
            return models
        else:
            raise ValueError("Anthropic models' list came empty")
    
    
    @override
    async def get_model(self, model_id: str) -> ModelInfo:
        """
        Fetch a specific model by ID from Anthropic API.
        
        Args:
            model_id: The model identifier or alias
        
        Returns:
            ModelInfo object in OpenAI-compatible format
        """

        if hasattr(self.async_client, 'models'):
            response = await self.async_client.models.retrieve(model_id)
            
            created_timestamp = int(datetime.fromisoformat(response.created_at.replace('Z', '+00:00')).timestamp())
            
            return ModelInfo(
                id=response.id,
                created=created_timestamp,
                owned_by="anthropic"
            )
        raise ValueError(f"Model {model_id} not found")

    
    @staticmethod
    def _supports_extended_thinking(model: str) -> bool:
        """
        Check whether a model gets an extended-thinking block at all.

        This gate decides *whether* thinking is applied, not *how* (legacy
        vs adaptive — see ``_uses_adaptive_thinking``). It intentionally
        preserves today's working behavior:
        - Sonnet: claude-3-7-sonnet and claude-sonnet-4-x get thinking;
          newer/other Sonnet (e.g. claude-sonnet-5) do NOT — they work today
          without a thinking block and must stay that way.
        - Opus: any claude-opus-4 and every newer Opus release (5.x, ...)
          get thinking. Future Opus launches are covered without a code
          change.
        - Anything else (e.g. Haiku) is unchanged: no thinking block.
        """
        m = model.lower().replace(".", "-")

        if "sonnet" in m:
            if "3-7" in m:
                return True
            if re.search(r"sonnet-4(?:-|$)", m):
                return True
            return False

        opus = re.search(r"opus-(\d+)", m)
        if opus and int(opus.group(1)) >= 4:
            return True

        return False
    
    @staticmethod
    def _uses_adaptive_thinking(model: str) -> bool:
        """
        For a thinking-capable model, decide the control API to use.

        Anthropic dropped the legacy ``thinking.type=enabled`` +
        ``budget_tokens`` control starting with Opus 4.7. Those models 400
        with ``"thinking.type.enabled" is not supported`` and instead expect
        ``thinking.type=adaptive`` plus ``output_config.effort``.

        Rule (only consulted when ``_supports_extended_thinking`` is True):
        - Sonnet stays on the legacy/current path forever — its models work
          today and must not change.
        - Opus is version-gated: legacy through 4.6, adaptive for 4.7 and
          EVERY newer release (4.9, 5.x, ...). Plain ``claude-opus-4`` with
          no minor is treated as legacy (matches current behavior).
        - Any other thinking-capable model we did not map explicitly defaults
          to the new adaptive API, so future Anthropic families following the
          new rule work without a code change.
        """
        m = model.lower().replace(".", "-")

        if "sonnet" in m:
            return False

        match = re.search(r"opus-(\d+)(?:-(\d+))?", m)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            return (major, minor) > (4, 6)

        return True

    @staticmethod
    def _apply_thinking_kwargs(
        request: "ChatCompletionRequest",
        kwargs: dict[str, Any],
        effort_to_budget: dict[str, int],
    ) -> int:
        """
        Populate ``kwargs`` with the correct thinking controls for the model.

        Returns the resolved thinking budget (0 when not applicable), used only
        for human-facing status strings. For adaptive-thinking models the
        budget is informational; Anthropic sizes thinking from ``effort``.
        """
        effort = (request.reasoning_effort or "medium").lower()

        if AnthropicClient._uses_adaptive_thinking(request.model):
            kwargs["thinking"] = {"type": "adaptive"}
            extra_body = kwargs.setdefault("extra_body", {})
            extra_body["output_config"] = {"effort": effort}
            return effort_to_budget.get(effort, 5000)

        thinking_budget = effort_to_budget.get(effort, 5000)
        thinking_budget = max(thinking_budget, 1024)
        if request.max_tokens and request.max_tokens >= 1024:
            thinking_budget = min(thinking_budget, request.max_tokens)
        else:
            kwargs["max_tokens"] = int(thinking_budget * 1.5)
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        return thinking_budget

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
    
    @override
    async def create_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion."""

        system_message, anthropic_messages = self._convert_messages(request.messages)
        
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or 4096,
        }
        
        if self._supports_extended_thinking(request.model):
            effort_to_budget = {
                "low": 2000,
                "medium": 5000,
                "high": 10000,
            }
            self._apply_thinking_kwargs(request, kwargs, effort_to_budget)
        
        if system_message:
            kwargs["system"] = system_message
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
        
        response: Message = await self.async_client.messages.create(**kwargs)
        
        completion_id = f"chatcmpl-{response.id}"
        created = int(time.time())
        
        content = ""
        if response.content:
            content = " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )
        
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
    
    @override
    async def create_stream_completion(
        self,
        request: ChatCompletionRequest,
        db: sqlite3.Connection | None = None,
        previous_completion: PreviousCompletion | None = None
    ) -> AsyncIterator[str]:
        """Create a streaming chat completion with reasoning support."""

        system_message, anthropic_messages = self._convert_messages(request.messages)
        
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or 4096,
        }
        
        thinking_budget = 0
        if self._supports_extended_thinking(request.model):
            effort_to_budget = {
                "low": 2000,
                "medium": 5000,
                "high": 16000,
            }
            thinking_budget = self._apply_thinking_kwargs(request, kwargs, effort_to_budget)
        
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
        
        thinking_status = f" (extended thinking: {request.reasoning_effort or 'medium'} effort, {thinking_budget} tokens)" if self._supports_extended_thinking(request.model) else ""

        yield f"data: {self.reasoning_content(
            request, 
            completion_id, 
            InterfaceMessage.PRE_CONNECTION(params=f"{request.model}{thinking_status}")
        ).model_dump_json()}\n\n"
          

        async with self.async_client.messages.stream(**kwargs) as stream:
            yield f"data: {self.reasoning_content(
                request, 
                completion_id,
                InterfaceMessage.STARTING()
            ).model_dump_json()}\n\n"

            first_event = True
            async for event in stream:
                if first_event:
                    first_event = False
                
                chunk = self._convert_stream_event_new(
                    event, completion_id, created, request.model
                )
                if chunk:
                    yield f"data: {chunk.model_dump_json()}\n\n"
        
        yield "data: [DONE]\n\n"
    
    @staticmethod
    def _convert_stream_event_new(
        event: Any,
        completion_id: str,
        created: int,
        model: str
    ) -> ChatCompletionChunk | None:
        """Convert Anthropic streaming event to OpenAI format with reasoning support."""
        from anthropic.lib.streaming._types import ThinkingEvent, TextEvent, MessageStopEvent
        
        if isinstance(event, ThinkingEvent):
            return ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={"reasoning_content": event.thinking},
                        finish_reason=None
                    )
                ]
            )
        
        elif isinstance(event, TextEvent):
            return ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={"role": "assistant", "content": event.text},
                        finish_reason=None
                    )
                ]
            )
        
        elif isinstance(event, MessageStopEvent):
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
    
    @staticmethod
    def _convert_stream_event(
        event: MessageStreamEvent,
        completion_id: str,
        created: int,
        model: str
    ) -> ChatCompletionChunk | None:
        """Convert Anthropic streaming event to OpenAI format (legacy support)."""
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


anthropic_client = AnthropicClient()
