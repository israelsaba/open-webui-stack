from __future__ import annotations
from datetime import UTC, datetime
from typing import Literal
from typing_extensions import Any
from pydantic import BaseModel, Field, field_serializer


class ChatMessage(BaseModel):
    """A single chat message."""

    model_config = {"extra": "allow"}
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = {"extra": "allow"}
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    reasoning_effort: str | None = Field(
        default=None, description="Reasoning effort level: low, medium, high"
    )
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    provider: str | None = Field(
        default=None,
        description="Optional local provider route: openai, anthropic, google, or xai",
    )


class ModelInfo(BaseModel):
    """Information about a single model."""

    id: str
    object: Literal["model"] = "model"
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    owned_by: str

    @field_serializer("created")
    def return_str(self, field: datetime) -> str:
        return str(field)


class ModelsResponse(BaseModel):
    """Response containing list of available models."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] | None

    class Config:
        # Allow extra fields like reasoning_content
        extra = "allow"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatCompletionStreamChoice(BaseModel):
    """A single streaming completion choice."""

    index: int
    delta: Any
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] | None = (
        None
    )


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str
    choices: list[ChatCompletionStreamChoice]

    @field_serializer("created")
    def return_str(self, field: datetime) -> str:
        return str(field)
