import sqlite3
import logging
import time
import asyncio
from datetime import datetime
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
    PreviousCompletion,
    Usage,
)

logger = logging.getLogger(__name__)

# Polling interval for checking interaction status (in seconds)
INTERACTION_POLL_INTERVAL = 30


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
            
            api_key = settings.google_api_key.get_secret_value()
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                data = response.json()
            
            total_models_from_api = len(data.get("models", []))
            logger.debug(f"Total models returned by API: {total_models_from_api}")
            
            for model in data.get("models", []):
                model_name = model.get("name", "")
                supported_methods = model.get("supportedGenerationMethods", [])
                
                logger.debug(f"Model {model_name}: supported_methods={supported_methods}")
                
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
            
            logger.debug(f"Successfully fetched {len(models)} models from Gemini API")
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
            # Deep Research models (virtual models that use deep research endpoint)
            "gemini-2.0-flash-thinking-deep-research",
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
        """Create a chat completion."""

        system_message = self._extract_system_message(request.messages)
        contents = self._convert_messages(request.messages)
        
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens,
            system_instruction=system_message,
        )
        
        if request.stop:
            stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop
            config.stop_sequences = stop_sequences

        try:
            response = self.client.models.generate_content(
                model=request.model,
                contents=contents,
                config=config
            )
            
            completion_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            
            content = response.text if hasattr(response, 'text') and response.text is not None else ""
            
            finish_reason: str = "stop"
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    reason_str = str(candidate.finish_reason)
                    if "MAX_TOKENS" in reason_str:
                        finish_reason = "length"
                    elif "SAFETY" in reason_str or "RECITATION" in reason_str:
                        finish_reason = "content_filter"
                    else:
                        finish_reason = "stop"

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

    @staticmethod
    def _convert_messages_to_turns(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """
        Convert messages to Interaction turns (dicts).
        
        Args:
            messages: List of ChatMessage objects
            
        Returns:
            List of TurnParam dicts for Interactions API
        """
        turns = []
        for msg in messages:
            if msg.role == "system":
                continue
            
            role = "user" if msg.role == "user" else "model"
            turns.append({
                "role": role,
                "content": msg.content
            })
        return turns


    async def _create_interaction_stream(
        self,
        request: ChatCompletionRequest,
        db: sqlite3.Connection | None = None,
        previous_completion: PreviousCompletion | None = None,
        md5_hash: str | None = None
    ) -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        
        system_message = self._extract_system_message(request.messages)
        input_turns = self._convert_messages_to_turns(request.messages)
        
        if system_message:
            if input_turns and input_turns[0]["role"] == "user":
                input_turns[0]["content"] = f"System Instruction: {system_message}\n\n{input_turns[0]['content']}"
            else:
                input_turns.insert(0, {
                    "role": "user",
                    "content": f"System Instruction: {system_message}"
                })
        
        interaction_id = previous_completion.interaction_id if previous_completion and previous_completion.interaction_id else None
        last_event_id = None
        is_complete = False
        last_poll_time = time.time()
        
        if not interaction_id:
            ts = datetime.now().strftime("%H:%M:%S")
            yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'[SDK {ts}] Connecting to Deep Research Agent ({request.model})...\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
            
            kwargs = {
                "agent": request.model,
                "input": input_turns,
                "stream": True,
                "background": True,
                "agent_config":{
                    "type": "deep-research",
                    "thinking_summaries": "auto"
                }
            }
            
            stream = await self.client.aio.interactions.create(**kwargs)
            ts = datetime.now().strftime("%H:%M:%S")
            yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'[SDK {ts}] Interaction started...\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
        else:
            ts = datetime.now().strftime("%H:%M:%S")
            yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'[SDK {ts}] Continuing interaction with id {interaction_id}\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
            stream = await self.client.aio.interactions.get(id=interaction_id, stream=True)
        
        # Convert stream to async iterator we can control
        stream_iter = stream.__aiter__()
        
        while not is_complete:
            try:
                # Wait for next event with timeout equal to polling interval
                event = await asyncio.wait_for(stream_iter.__anext__(), timeout=INTERACTION_POLL_INTERVAL)
                
                logger.debug(f"interaction event: {event}")
                
                if event.event_type == "interaction.start":
                    interaction_id = event.interaction.id
                    logger.debug(f"Interaction ID is: {interaction_id}, md5_hash: {md5_hash}")
                    
                    if md5_hash:
                        import sqlite3 as sqlite_module
                        from app.config import settings
                        conn = sqlite_module.connect(str(settings.db_path))
                        conn.execute("UPDATE research_hashes SET interaction_id = ? WHERE md5 = ?", (interaction_id, md5_hash))
                        conn.commit()
                        conn.close()
                    
                    yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content':f'Gemini started reasoning with id: {interaction_id}'}, finish_reason=None)]).model_dump_json()}\n\n"
                
                if event.event_id:
                    last_event_id = event.event_id
                
                if event.event_type == "content.delta":
                    if event.delta.type == "text":
                        yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'content':event.delta.text}, finish_reason=None)]).model_dump_json()}\n\n"
                    elif event.delta.type == "thought_summary":
                        yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': event.delta.content.text}, finish_reason=None)]).model_dump_json()}\n\n"
                
                if event.event_type in ['interaction.complete', 'error']:
                    is_complete = True
                    yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason='stop')]).model_dump_json()}\n\n"
                
                # Reset poll timer when we receive events
                last_poll_time = time.time()
                
            except asyncio.TimeoutError:
                # Timeout reached - poll for status and send update, then reconnect
                if interaction_id:
                    try:
                        logger.info(f"Polling interaction {interaction_id} status after {INTERACTION_POLL_INTERVAL}s timeout")
                        interaction_status = await self.client.aio.interactions.get(id=interaction_id)
                        
                        if interaction_status and hasattr(interaction_status, 'status'):
                            status = interaction_status.status
                            logger.info(f"Interaction status from API: {status}")
                            
                            # Map status to user-friendly messages
                            status_messages = {
                                "in_progress": "still in progress - researching and generating response",
                                "requires_action": "requires action",
                                "completed": "completed",
                                "failed": "failed",
                                "cancelled": "cancelled"
                            }
                            
                            status_msg = status_messages.get(status, f"status: {status}")
                            
                            # Only show polling message if still running
                            if status in ["in_progress", "requires_action"]:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                msg = f'\n\n[SDK {timestamp}] Interaction {status_msg}\n\n'
                                logger.info(f"Yielding status message: {msg.strip()}")
                                chunk_data = ChatCompletionChunk(
                                    id=completion_id, 
                                    created=created, 
                                    model=request.model, 
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0, 
                                            delta={'reasoning_content': msg}, 
                                            finish_reason=None
                                        )
                                    ]
                                ).model_dump_json()
                                logger.info(f"Chunk JSON: {chunk_data}")
                                yield f"data: {chunk_data}\n\n"
                                
                                # Small delay to ensure message is delivered before reconnecting
                                await asyncio.sleep(0.1)
                                
                                # Reconnect after showing status since stream likely timed out
                                logger.info("Yielding reconnecting message")
                                reconnect_timestamp = datetime.now().strftime("%H:%M:%S")
                                yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'\n\n[SDK {reconnect_timestamp}] Reconnecting to stream...\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
                                
                                # Small delay before reconnecting
                                await asyncio.sleep(0.5)
                                kwargs = {"id": interaction_id, "stream": True}
                                if last_event_id:
                                    kwargs["last_event_id"] = last_event_id
                                stream = await self.client.aio.interactions.get(**kwargs)
                                stream_iter = stream.__aiter__()
                                # Continue the loop to start listening to the new stream
                                continue
                            elif status in ["completed", "failed", "cancelled"]:
                                yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'[SDK] Interaction {status_msg}\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
                                is_complete = True
                        else:
                            logger.warning(f"No status available in interaction object: {interaction_status}")
                            yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'[SDK] Interaction status unknown - continuing to monitor\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
                    except Exception as poll_error:
                        logger.warning(f"Failed to poll interaction status: {poll_error}")
                        yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[SDK] Unable to check status\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
                
            except StopAsyncIteration:
                # Stream ended unexpectedly
                logger.warning(f"[EDGE CASE] Stream ended with StopAsyncIteration. is_complete={is_complete}, interaction_id={interaction_id}")
                if not is_complete:
                    logger.warning("[EDGE CASE] Stream ended but interaction not complete - this shouldn't happen as timeout should handle reconnection")
                break
        
        yield "data: [DONE]\n\n"

    async def create_stream_completion(
        self,
        request: ChatCompletionRequest,
        db: sqlite3.Connection | None = None,
        previous_completion: PreviousCompletion | None = None
    ) -> AsyncIterator[str]:
        """Create a streaming chat completion."""

        if "deep-research" in request.model.lower():
            md5_hash = previous_completion.md5 if previous_completion else None
            async for chunk in self._create_interaction_stream(request, db, previous_completion, md5_hash):
                yield chunk
            return

        system_message = self._extract_system_message(request.messages)
        contents = self._convert_messages(request.messages)
        
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens,
            system_instruction=system_message,
        )
        
        if request.stop:
            stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop
            config.stop_sequences = stop_sequences
        
        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        try:
            yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': f'[SDK] Connecting to Google Gemini API with model {request.model}...\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
            
            response_stream = self.client.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=config
            )
            
            # Send meta-reasoning: Stream started
            yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[SDK] Stream established, awaiting response...\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
            
            # Track if we're in a thinking block for thinking models
            in_thinking_block = False
            thinking_buffer = []
            first_chunk = True
            
            for chunk in response_stream:
                if first_chunk:
                    yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[SDK] Response received, streaming content...\n\n'}, finish_reason=None)]).model_dump_json()}\n\n"
                    first_chunk = False

                if hasattr(chunk, 'text') and chunk.text:
                    content_delta = chunk.text
                    
                    # Check if this is a thinking model (contains "thinking" in model name)
                    is_thinking_model = "thinking" in request.model.lower()
                    
                    if is_thinking_model:
                        if "<thinking>" in content_delta:
                            in_thinking_block = True
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
                            thinking_buffer.append(content_delta)
                            continue
                    
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
