import logging
import re
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.auth import BearerTokenMiddleware, parse_api_keys
from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
)

from . import deps


class RedactSecrets(logging.Filter):
    _patterns: list[re.Pattern[str]] = [
        re.compile(r"(?i)(\bauthorization\s*:\s*)([^\r\n]+)"),
        re.compile(r'(?i)(["\']authorization["\']\s*:\s*["\'])([^"\']+)(["\'])'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()

        msg = self._patterns[0].sub(r"\1[REDACTED]", msg)
        msg = self._patterns[1].sub(r"\1[REDACTED]\3", msg)

        record.msg = msg
        record.args = ()  # prevent old args from being formatted again
        return True


logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

for h in logging.getLogger().handlers:
    h.addFilter(RedactSecrets())

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenRouter to OpenAI API Bridge",
    description="OpenAI-compatible API with OpenRouter model routing",
    version="2.0.0",
)

# Configure CORS
cors_origins = [
    origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled for origins: {cors_origins}")
else:
    logger.warning("No CORS origins configured")

# Configure authentication
valid_tokens = parse_api_keys(settings.api_keys)
if settings.environment.upper() in {"PROD", "PRODUCTION"}:
    if not valid_tokens:
        raise RuntimeError("SDK__API_KEYS is required when SDK__ENVIRONMENT is PROD")
    if not cors_origins or "*" in cors_origins:
        raise RuntimeError(
            "SDK__CORS_ORIGINS must contain explicit origins when SDK__ENVIRONMENT is PROD"
        )
if valid_tokens:
    app.add_middleware(BearerTokenMiddleware, valid_tokens=valid_tokens)
    logger.debug(
        f"Bearer token authentication enabled with {len(valid_tokens)} valid tokens"
    )
else:
    logger.warning("No API keys configured - authentication is DISABLED")


_model_cache: set[str] | None = None


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "OpenRouter to OpenAI API Bridge",
        "docs": "/docs",
        "models": "/v1/models",
    }


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List the live OpenRouter model catalog in OpenAI format."""

    all_models = await deps.list_models()

    if not all_models:
        raise HTTPException(
            status_code=500, detail="Failed to fetch models from OpenRouter"
        )

    return all_models


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    client: Annotated[object, Depends(deps.get_client)],
) -> ChatCompletionResponse | StreamingResponse:
    """
    Create a chat completion using OpenRouter's provider routing layer.

    Supports both streaming and non-streaming responses.
    """
    logger.info(
        f"Chat completion request: model={request.model}, "
        f"messages={len(request.messages)}, stream={request.stream}"
    )

    if request.stream:
        content = client.create_stream_completion(request)
        return StreamingResponse(content=content, media_type="text/event-stream")
    response = await client.create_completion(request)
    logger.info(
        f"Completion successful: tokens={response.usage.total_tokens}, "
        f"finish_reason={response.choices[0].finish_reason}"
    )
    return response


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        access_log=False,
        timeout_graceful_shutdown=0,
    )
