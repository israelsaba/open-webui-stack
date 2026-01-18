import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import google.generativeai as genai
from google.generativeai.types import (
    GenerateContentResponse,
    HarmCategory,
    HarmBlockThreshold,
)

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
    """Client for interacting with Google Gemini API."""

    def __init__(self) -> None:
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key.get_secret_value())
            self.available = True
        else:
            logger.warning("Google API key not configured. Gemini models will be unavailable.")
            self.available = False

    async def list_models(self, limit: int = 100) -> list[ModelInfo]:
        """
        Fetch available models from Google Gemini API.
        
        Args:
            limit: Maximum number of models to fetch (default: 100)
        
        Returns:
            List of ModelInfo objects in OpenAI-compatible format
        """
        if not self.available:
            return []

        try:
            # List models using the synchronous API (it's fast enough)
            # or wrap it if we strictly need async, but for now we'll call it directly
            # as genai.list_models is a generator
            
            # Note: genai.list_models() returns an iterator
            models = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    # Parse name to get ID (e.g. "models/gemini-pro" -> "gemini-pro")
                    model_id = m.name.replace("models/", "")
                    
                    # Google API doesn't provide creation time, use current time or hardcoded fallback
                    created_timestamp = int(time.time())
                    
                    models.append(ModelInfo(
                        id=model_id,
                        created=created_timestamp,
                        owned_by="google"
                    ))
            
            logger.info(f"Successfully fetched {len(models)} models from Gemini API")
            return models
        except Exception as e:
            logger.warning(f"Failed to fetch models from Gemini API: {e}, using hardcoded list")
            return self._get_hardcoded_models()

    @staticmethod
    def _get_hardcoded_models() -> list[ModelInfo]:
        """
        Return a hardcoded list of available models as fallback.
        
        Returns:
            List of ModelInfo objects for known Gemini models
        """
        base_timestamp = int(time.time())
        
        model_ids = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
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
        if not self.available:
            raise ValueError("Google API key not configured")

        try:
            # Handle "gemini-" prefix if passed without "models/"
            full_model_name = f"models/{model_id}" if not model_id.startswith("models/") else model_id
            
            model = genai.get_model(full_model_name)
            
            return ModelInfo(
                id=model.name.replace("models/", ""),
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
    def _convert_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Returns:
            List of content dictionaries for Gemini
        """
        gemini_messages = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                # Gemini 1.5 supports system instructions separately, 
                # but for simplicity in chat history we might need to handle it differently
                # depending on how we initialize the model. 
                # For now, let's treat it as a user message or handle it at the model init level if possible.
                # However, the generate_content API takes a 'contents' list.
                # System prompts in Gemini are typically passed to GenerativeModel(system_instruction=...)
                # But here we are processing a list of messages for a stateless call.
                # We'll extract it and return it separately if we were creating the model object here.
                # Since we are just converting messages for history, we might skip it here and handle it 
                # in create_completion where we instantiate the model.
                pass 
            elif msg.role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg.content}]
                })
                
        return gemini_messages

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
        if not self.available:
            raise ValueError("Google API key not configured")

        system_message = self._extract_system_message(request.messages)
        contents = self._convert_messages(request.messages)
        
        # Configure model
        model_name = request.model
        if not model_name.startswith("models/") and not model_name.startswith("gemini-"):
             # Assuming simple ID is passed, Google usually expects 'gemini-...'
             # But if the user passed 'claude-...' it shouldn't be here.
             pass

        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=[request.stop] if isinstance(request.stop, str) else request.stop if request.stop else None
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_message
        )

        try:
            response = await model.generate_content_async(
                contents,
                generation_config=generation_config,
                stream=False
            )
            
            completion_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            
            content = response.text
            
            # Map finish reason
            finish_reason = "stop"
            if response.candidates and response.candidates[0].finish_reason:
                # Map Google finish reasons to OpenAI
                # 1: STOP, 2: MAX_TOKENS, 3: SAFETY, 4: RECITATION, 5: OTHER
                reason_map = {
                    1: "stop",
                    2: "length",
                    3: "content_filter",
                    4: "content_filter" 
                }
                finish_reason = reason_map.get(response.candidates[0].finish_reason.value, "stop")

            usage = Usage(
                prompt_tokens=0, # Google doesn't always return token counts easily in the simple response object without extra calls
                completion_tokens=0,
                total_tokens=0
            )
            
            if response.usage_metadata:
                usage = Usage(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count
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
        if not self.available:
            raise ValueError("Google API key not configured")

        system_message = self._extract_system_message(request.messages)
        contents = self._convert_messages(request.messages)
        
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=[request.stop] if isinstance(request.stop, str) else request.stop if request.stop else None
        )

        model = genai.GenerativeModel(
            model_name=request.model,
            system_instruction=system_message
        )
        
        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        try:
            response_stream = await model.generate_content_async(
                contents,
                generation_config=generation_config,
                stream=True
            )
            
            async for chunk in response_stream:
                content_delta = chunk.text
                
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


# Global client instance
gemini_client = GeminiClient()
