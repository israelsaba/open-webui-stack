from __future__ import annotations
from datetime import UTC, datetime
from enum import Enum
from typing import Literal, overload
from typing_extensions import Any, Self
from pydantic import BaseModel, Field, field_serializer


class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    reasoning_effort: str | None = Field(default=None, description="Reasoning effort level: low, medium, high")


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
    finish_reason: Literal["stop", "length", "content_filter"] | None
    
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

class Message(str, Enum):
    STARTING = "Streaming established. Awaiting model's response"
    DEEP_RESEARCH_RECONNECTING = "Reconnecting to stream..."
    PRE_CONNECTION = "Connecting to {params}"
    DEEP_RESEARCH_ID = "Continuing interaction with {params}" 
    DEEP_RESEARCH_STATUS = "Interaction status is {params}" 

    @overload
    def __call__(self: Literal[Message.STARTING, Message.DEEP_RESEARCH_RECONNECTING]) -> str: ...
    @overload
    def __call__(
        self: Literal[
            Message.PRE_CONNECTION, 
            Message.DEEP_RESEARCH_ID, 
            Message.DEEP_RESEARCH_STATUS
        ], 
        params: str
    ) -> str: ...


    def __call__(self:Self, params: str | None = None) -> str:
        prepend = f"\n\n[log {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC] "
        postpend = "\n\n"
        if self in {Message.STARTING, Message.DEEP_RESEARCH_RECONNECTING}:
            return f"{prepend}{str(self.value)}{postpend}" 
        if not params:
            raise ValueError("missing params")
        return f"{prepend}{str(self.value).format(params=params)}{postpend}"


class ReasoningDelta(BaseModel):
    reasoning_content: str

class ChatCompletionStreamChoice(BaseModel):
    """A single streaming completion choice."""
    index: int
    delta: Any
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


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

class CompletionChunkResponse(BaseModel):
    data: ChatCompletionChunk

class PreviousCompletion (BaseModel):
    id: int
    md5: str
    interaction_id: str | None = None
    created_at: str
    created_by: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None
    deleted_at: str | None = None
    deleted_by: str | None = None
