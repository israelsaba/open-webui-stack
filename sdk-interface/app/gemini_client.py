import logging
import time
from collections.abc import AsyncIterator
from typing import Any
import httpx

from google import genai
from google.genai import types

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


class GeminiClient:
    """Client for interacting with Google Gemini API using the new google-genai SDK."""

    def __init__(self) -> None:
        if settings.google_api_key:
            self.client = genai.Client(api_key=settings.google_api_key.get_secret_value())
            self.available = True
        else:
            logger.warning("Google API key not configured. Gemini models will be unavailable.")
            self.available = False
            self.client = None

    async def list_models(self, limit: int = 100) -> list[ModelInfo]:
        """
        Fetch available models from Google Gemini API using REST API.
        
        Args:
            limit: Maximum number of models to fetch (default: 100)
        
        Returns:
            List of ModelInfo objects in OpenAI-compatible format
        """
        if not self.available or not settings.google_api_key:
            return []

        try:
            models = []
            
            # Use the REST API directly to list models
            # Documentation: https://ai.google.dev/api/rest/v1beta/models/list
            api_key = settings.google_api_key.get_secret_value()
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            
            logger.info("Calling Gemini REST API to list models...")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                data = response.json()
            
            # Parse the response
            total_models_from_api = len(data.get("models", []))
            logger.info(f"Total models returned by API: {total_models_from_api}")
            
            for model in data.get("models", []):
                model_name = model.get("name", "")
                supported_methods = model.get("supportedGenerationMethods", [])
                
                # Log model details
                logger.debug(f"Model {model_name}: supported_methods={supported_methods}")
                
                # Filter for models that support generateContent
                if "generateContent" in supported_methods:
                    # Remove 'models/' prefix from name
                    model_id = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
                    
                    if model_id:
                        models.append(ModelInfo(
                            id=model_id,
                            created=int(time.time()),
                            owned_by="google"
                        ))
            
            # If no models were returned, fall back to hardcoded list
            if len(models) == 0:
                logger.warning(
                    f"Gemini API returned {total_models_from_api} total models but 0 support generateContent. "
                    f"Falling back to hardcoded model list."
                )
                return self._get_hardcoded_models()
            
            logger.info(f"Successfully fetched {len(models)} models from Gemini API")
            return models[:limit]
        except Exception as e:
            logger.warning(f"Failed to fetch models from Gemini API: {e}, using hardcoded list")
            return self._get_hardcoded_models()

    @staticmethod
    def _get_hardcoded_models() -> list[ModelInfo]:
        """
        Return a hardcoded list of available models as fallback.
        
        This list should be updated periodically to include new models.
        The REST API call above should dynamically fetch all available models,
        but this serves as a fallback if the API is unavailable.
        
        Returns:
            List of ModelInfo objects for known Gemini models
        """
        base_timestamp = int(time.time())
        
        model_ids = [
            # Gemini 2.0 models (experimental)
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.0-flash-thinking-exp",
            "gemini-2.0-pro-exp",
            # Gemini 1.5 models
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-002",
            "gemini-1.5-pro-001",
            "gemini-1.5-pro",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b-latest",
            "gemini-1.5-flash-8b-001",
            "gemini-1.5-flash-8b",
            # Gemini 1.0 models
            "gemini-1.0-pro-latest",
            "gemini-1.0-pro-001",
            "gemini-1.0-pro",
            "gemini-1.0-pro-vision-latest",
        ]
        
        return [
            ModelInfo(
                id=model_id,
                created=base_timestamp,
                owned_by="google"
            )
            for model_id in model_ids
        ]

    async def get_model(self, model_id: str) -> ModelInfo:
        """
        Fetch a specific model by ID from Gemini API.
        
        Args:
            model_id: The model identifier
        
        Returns:
            ModelInfo object in OpenAI-compatible format
        """
        if not self.available or not self.client:
            raise ValueError("Google API key not configured")

        try:
            # Handle "gemini-" prefix if passed without "models/"
            full_model_name = f"models/{model_id}" if not model_id.startswith("models/") else model_id
            
            model = self.client.models.get(model=full_model_name)
            
            return ModelInfo(
                id=getattr(model, 'name', '').replace("models/", "") if getattr(model, 'name', None) else '',
                created=int(time.time()),
                owned_by="google"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch model {model_id} from API: {e}")
            # Fallback
            hardcoded_models = self._get_hardcoded_models()
            for model in hardcoded_models:
                if model.id == model_id:
                    return model
            raise ValueError(f"Model {model_id} not found")

    @staticmethod
    def _convert_messages(messages: list[ChatMessage]) -> list[Any]: # Changed return type to list[Any]
        """
        Convert OpenAI-style messages to Gemini format.
        
        Returns:
            List of Content objects for Gemini
        """
        gemini_contents = []
        
        for msg in messages:
            if msg.role == "system":
                # System messages are handled separately in the config
                continue
            elif msg.role == "user":
                gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=msg.content)]
                    )
                )
            elif msg.role == "assistant":
                gemini_contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=msg.content)]
                    )
                )
                
        return gemini_contents

    @staticmethod
    def _extract_system_message(messages: list[ChatMessage]) -> str | None:
        """Extract system message from the message list."""
        for msg in messages:
            if msg.role == "system":
                return msg.content
        return None

    async def create_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion."""
        if not self.available or not self.client:
            raise ValueError("Google API key not configured")

        system_message = self._extract_system_message(request.messages)
        contents = self._convert_messages(request.messages)
        
        # Build generation config
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens,
            system_instruction=system_message,
        )
        
        # Add stop sequences if provided
        if request.stop:
            stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop
            config.stop_sequences = stop_sequences

        try:
            response = self.client.models.generate_content(
                model=request.model,
                contents=contents, # Keep as is, list[Any] should be compatible
                config=config
            )
            
            completion_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            
            # Extract text from response
            content = response.text if hasattr(response, 'text') and response.text is not None else ""
            
            # Map finish reason
            finish_reason: str = "stop"
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    # Map Google finish reasons to OpenAI
                    reason_str = str(candidate.finish_reason)
                    if "MAX_TOKENS" in reason_str:
                        finish_reason = "length"
                    elif "SAFETY" in reason_str or "RECITATION" in reason_str:
                        finish_reason = "content_filter"
                    else:
                        finish_reason = "stop"

            # Extract usage metadata
            usage = Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = Usage(
                    prompt_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata and response.usage_metadata.prompt_token_count is not None else 0,
                    completion_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata and response.usage_metadata.candidates_token_count is not None else 0,
                    total_tokens=response.usage_metadata.total_token_count if response.usage_metadata and response.usage_metadata.total_token_count is not None else 0
                )

            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=content),
                        finish_reason=finish_reason 
                    )
                ],
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Error creating Gemini completion: {e}", exc_info=True)
            raise

    async def create_stream_completion(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Create a streaming chat completion."""
        if not self.available or not self.client:
            raise ValueError("Google API key not configured")

        system_message = self._extract_system_message(request.messages)
        contents = self._convert_messages(request.messages)
        
        # Build generation config
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens,
            system_instruction=system_message,
        )
        
        # Add stop sequences if provided
        if request.stop:
            stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop
            config.stop_sequences = stop_sequences
        
        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        try:
            # Send meta-reasoning: Initiating connection
            yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'[SDK] Connecting to Google Gemini API with model {request.model}...'}, finish_reason=None)]).model_dump_json()}\n\n"
            
            # Stream the response
            response_stream = self.client.models.generate_content_stream(
                model=request.model,
                contents=contents, # Keep as is, list[Any] should be compatible
                config=config
            )
            
            # Send meta-reasoning: Stream started
            yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[SDK] Stream established, awaiting response...'}, finish_reason=None)]).model_dump_json()}\n\n"
            
            # Track if we're in a thinking block for thinking models
            in_thinking_block = False
            thinking_buffer = []
            first_chunk = True
            
            for chunk in response_stream:
                if first_chunk:
                    # Send meta-reasoning: First response received
                    yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[SDK] Response received, streaming content...'}, finish_reason=None)]).model_dump_json()}\n\n"
                    first_chunk = False
                # Extract text delta from chunk
                if hasattr(chunk, 'text') and chunk.text:
                    content_delta = chunk.text
                    
                    # Check if this is a thinking model (contains "thinking" in model name)
                    is_thinking_model = "thinking" in request.model.lower()
                    
                    if is_thinking_model:
                        # Parse thinking content from <thinking> tags
                        if "<thinking>" in content_delta:
                            in_thinking_block = True
                            # Extract any text before <thinking>
                            before_thinking = content_delta.split("<thinking>")[0]
                            if before_thinking:
                                response_chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0,
                                            delta={"role": "assistant", "content": before_thinking},
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {response_chunk.model_dump_json()}\n\n"
                            # Start collecting thinking content
                            after_thinking = content_delta.split("<thinking>")[1] if len(content_delta.split("<thinking>")) > 1 else ""
                            if "</thinking>" in after_thinking:
                                thinking_content = after_thinking.split("</thinking>")[0]
                                in_thinking_block = False
                                # Send thinking content
                                response_chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0,
                                            delta={"reasoning_content": thinking_content},
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {response_chunk.model_dump_json()}\n\n"
                                # Send any content after </thinking>
                                after_closing = after_thinking.split("</thinking>")[1] if len(after_thinking.split("</thinking>")) > 1 else ""
                                if after_closing:
                                    response_chunk = ChatCompletionChunk(
                                        id=completion_id,
                                        created=created,
                                        model=request.model,
                                        choices=[
                                            ChatCompletionStreamChoice(
                                                index=0,
                                                delta={"role": "assistant", "content": after_closing},
                                                finish_reason=None
                                            )
                                        ]
                                    )
                                    yield f"data: {response_chunk.model_dump_json()}\n\n"
                            else:
                                thinking_buffer.append(after_thinking)
                            continue
                        elif "</thinking>" in content_delta and in_thinking_block:
                            before_closing = content_delta.split("</thinking>")[0]
                            thinking_buffer.append(before_closing)
                            thinking_content = "".join(thinking_buffer)
                            thinking_buffer = []
                            in_thinking_block = False
                            # Send accumulated thinking content
                            response_chunk = ChatCompletionChunk(
                                id=completion_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    ChatCompletionStreamChoice(
                                        index=0,
                                        delta={"reasoning_content": thinking_content},
                                        finish_reason=None
                                    )
                                ]
                            )
                            yield f"data: {response_chunk.model_dump_json()}\n\n"
                            # Send any content after </thinking>
                            after_closing = content_delta.split("</thinking>")[1] if len(content_delta.split("</thinking>")) > 1 else ""
                            if after_closing:
                                response_chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0,
                                            delta={"role": "assistant", "content": after_closing},
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {response_chunk.model_dump_json()}\n\n"
                            continue
                        elif in_thinking_block:
                            # Accumulate thinking content
                            thinking_buffer.append(content_delta)
                            continue
                    
                    # Regular content (not in thinking block or not a thinking model)
                    response_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"role": "assistant", "content": content_delta},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {response_chunk.model_dump_json()}\n\n"
            
            # Send final stop chunk
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={},
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error streaming Gemini completion: {e}", exc_info=True)
            raise


gemini_client = GeminiClient()
