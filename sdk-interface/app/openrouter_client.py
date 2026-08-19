import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from .config import settings
from .models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    ModelInfo,
    Usage,
)

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """OpenAI-compatible gateway for OpenRouter's provider routing layer."""

    def __init__(self) -> None:
        if not settings.openrouter_api_key:
            logger.warning("OpenRouter API key not configured")
            self.client = None
        else:
            self.client = AsyncOpenAI(
                api_key=settings.openrouter_api_key.get_secret_value(),
                base_url=settings.openrouter_base_url,
                default_headers={
                    "HTTP-Referer": settings.openrouter_site_url,
                    "X-Title": settings.openrouter_site_name,
                },
            )
        self._models: tuple[float, list[ModelInfo]] | None = None

    def _require_client(self) -> AsyncOpenAI:
        if self.client is None:
            raise RuntimeError("SDK__OPENROUTER_API_KEY is not configured")
        return self.client

    @staticmethod
    def _owner(model_id: str) -> str:
        return model_id.split("/", 1)[0] if "/" in model_id else "openrouter"

    async def list_models(self, *, force_refresh: bool = False) -> list[ModelInfo]:
        """Return OpenRouter's live catalog, cached only for one minute."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._models
            and now - self._models[0] < settings.models_cache_ttl
        ):
            return self._models[1]

        response = await self._require_client().models.list()
        models = [
            ModelInfo(id=model.id, owned_by=self._owner(model.id))
            for model in response.data
        ]
        self._models = (now, models)
        logger.info("Loaded %d models from OpenRouter", len(models))
        return models

    @staticmethod
    def _request_kwargs(
        request: ChatCompletionRequest, *, stream: bool
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [
                message.model_dump(exclude_none=True) for message in request.messages
            ],
            "stream": stream,
        }
        for field in (
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "provider",
        ):
            value = getattr(request, field, None)
            if value is not None:
                kwargs[field] = value
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort
        return kwargs

    async def create_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        response = await self._require_client().chat.completions.create(
            **self._request_kwargs(request, stream=False)
        )
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        usage = response.usage
        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    index=choice.index,
                    message=ChatMessage.model_validate(message),
                    finish_reason=choice.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    async def create_stream_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        stream = await self._require_client().chat.completions.create(
            **self._request_kwargs(request, stream=True)
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta.model_dump(exclude_none=True)
            if "reasoning" in delta:
                delta["reasoning_content"] = delta.pop("reasoning")
            converted = ChatCompletionChunk(
                id=chunk.id,
                created=chunk.created,
                model=chunk.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=choice.index,
                        delta=delta,
                        finish_reason=choice.finish_reason,
                    )
                ],
            )
            yield f"data: {converted.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


openrouter_client = OpenRouterClient()
