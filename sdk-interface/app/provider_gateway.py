import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Provider:
    name: str
    client: AsyncOpenAI


class ProviderGateway:
    """Self-hosted OpenAI-compatible gateway for direct provider APIs."""

    def __init__(self) -> None:
        self.providers = tuple(self._configured_providers())
        self._models: tuple[float, list[ModelInfo]] | None = None
        self._routes: dict[str, Provider] = {}

    @staticmethod
    def _configured_providers() -> list[Provider]:
        configs = (
            ("openai", settings.openai_api_key, settings.openai_base_url),
            ("anthropic", settings.anthropic_api_key, settings.anthropic_base_url),
            ("google", settings.google_api_key, settings.google_base_url),
            ("xai", settings.grok_api_key, settings.grok_base_url),
        )
        return [
            Provider(
                name=name,
                client=AsyncOpenAI(api_key=key.get_secret_value(), base_url=base_url),
            )
            for name, key, base_url in configs
            if key
        ]

    async def list_models(self, *, force_refresh: bool = False) -> list[ModelInfo]:
        """Fetch models from every configured provider without a hardcoded list."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._models
            and now - self._models[0] < settings.models_cache_ttl
        ):
            return self._models[1]
        if not self.providers:
            raise RuntimeError("No AI provider API keys are configured")

        results = await asyncio.gather(
            *(self._list_provider_models(provider) for provider in self.providers),
            return_exceptions=True,
        )
        models: list[ModelInfo] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Provider model listing failed: %s", result)
                continue
            models.extend(result)
        if not models:
            raise RuntimeError("No configured provider returned models")
        self._models = (now, models)
        return models

    async def _list_provider_models(self, provider: Provider) -> list[ModelInfo]:
        response = await provider.client.models.list()
        models = [
            ModelInfo(id=model.id, owned_by=provider.name) for model in response.data
        ]
        self._routes.update({model.id.lower(): provider for model in models})
        logger.info("Loaded %d models from %s", len(models), provider.name)
        return models

    async def get_client(self, request: ChatCompletionRequest) -> Provider:
        if request.provider:
            for provider in self.providers:
                if provider.name == request.provider.lower():
                    return provider
            raise ValueError(f"Provider {request.provider!r} is not configured")
        route = self._routes.get(request.model.lower())
        if route:
            return route
        await self.list_models(force_refresh=True)
        route = self._routes.get(request.model.lower())
        if not route:
            raise ValueError(f"Model {request.model} not found in configured providers")
        return route

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
        provider = await self.get_client(request)
        response = await provider.client.chat.completions.create(
            **self._request_kwargs(request, stream=False)
        )
        choice = response.choices[0]
        usage = response.usage
        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    index=choice.index,
                    message=ChatMessage.model_validate(
                        choice.message.model_dump(exclude_none=True)
                    ),
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
        provider = await self.get_client(request)
        stream = await provider.client.chat.completions.create(
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


provider_gateway = ProviderGateway()
