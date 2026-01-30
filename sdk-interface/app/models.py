from datetime import UTC, datetime
from typing import Literal
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
    def return_str(self, field: datetime) -> str | None:
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
    reasoning_tokens: int = 0  # For models with reasoning capabilities


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


class ChatCompletionStreamChoice(BaseModel):
    """A single streaming completion choice."""
    index: int
    delta: dict[str, str]  # Can include "content", "reasoning_content", "role"
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]

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
