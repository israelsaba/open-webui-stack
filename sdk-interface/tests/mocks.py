"""
Mock responses for provider APIs used in testing.
"""

import json
from typing import Any


class MockResponses:
    """Container for all mock API responses."""
    
    @staticmethod
    def anthropic_models_list() -> dict[str, Any]:
        """Mock response for Anthropic models list."""
        return {
            "data": [
                {
                    "id": "claude-opus-4-5-20251101",
                    "type": "model",
                    "display_name": "Claude Opus 4.5",
                    "created_at": "2025-11-01T00:00:00Z"
                },
                {
                    "id": "claude-sonnet-4-5-20250929",
                    "type": "model",
                    "display_name": "Claude Sonnet 4.5",
                    "created_at": "2025-09-29T00:00:00Z"
                }
            ]
        }
    
    @staticmethod
    def anthropic_completion() -> dict[str, Any]:
        """Mock response for Anthropic chat completion."""
        return {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello from Anthropic test!"
                }
            ],
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        }
    
    @staticmethod
    def anthropic_streaming_chunk(text: str, finish: bool = False) -> str:
        """Generate SSE chunk for Anthropic streaming."""
        if finish:
            event = {
                "type": "message_stop"
            }
        else:
            event = {
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": text
                }
            }
        return f"data: {json.dumps(event)}\n\n"
    
    @staticmethod
    def google_models_list() -> dict[str, Any]:
        """Mock response for Google Gemini models list."""
        return {
            "models": [
                {
                    "name": "models/gemini-2.0-flash-exp",
                    "displayName": "Gemini 2.0 Flash",
                    "supportedGenerationMethods": ["generateContent"]
                },
                {
                    "name": "models/gemini-2.0-flash-thinking-exp",
                    "displayName": "Gemini 2.0 Flash Thinking",
                    "supportedGenerationMethods": ["generateContent"]
                },
                {
                    "name": "models/deep-research-pro-preview-12-2025",
                    "displayName": "Deep Research Pro",
                    "supportedGenerationMethods": ["generateContent"]
                }
            ]
        }
    
    @staticmethod
    def google_completion() -> dict[str, Any]:
        """Mock response for Google Gemini chat completion."""
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Hello from Gemini test!"
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30
            }
        }
    
    @staticmethod
    def google_deep_research_interaction() -> dict[str, Any]:
        """Mock response for Google Deep Research interaction creation."""
        return {
            "id": "v1_mock_interaction_id",
            "status": "in_progress",
            "agent": "deep-research-pro-preview-12-2025",
            "input": [],
            "outputs": []
        }
    
    @staticmethod
    def google_deep_research_complete() -> dict[str, Any]:
        """Mock response for completed Deep Research interaction."""
        return {
            "id": "v1_mock_interaction_id",
            "status": "completed",
            "agent": "deep-research-pro-preview-12-2025",
            "input": [],
            "outputs": [
                {
                    "type": "text",
                    "text": "The answer is 4."
                }
            ]
        }
    
    @staticmethod
    def google_streaming_chunk(text: str, finish: bool = False) -> str:
        """Generate SSE chunk for Google streaming."""
        if finish:
            event = {
                "event_type": "interaction.complete"
            }
        else:
            event = {
                "event_type": "content.delta",
                "delta": {
                    "type": "text",
                    "text": text
                }
            }
        return f"data: {json.dumps(event)}\n\n"
    
    @staticmethod
    def xai_models_list() -> dict[str, Any]:
        """Mock response for xAI Grok models list."""
        return {
            "data": [
                {
                    "id": "grok-2-vision-1212",
                    "object": "model",
                    "created": 1702339200,
                    "owned_by": "xai"
                },
                {
                    "id": "grok-code-fast-1",
                    "object": "model",
                    "created": 1702339200,
                    "owned_by": "xai"
                }
            ],
            "object": "list"
        }
    
    @staticmethod
    def xai_completion() -> dict[str, Any]:
        """Mock response for xAI Grok chat completion."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1702339200,
            "model": "grok-code-fast-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from Grok test!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    @staticmethod
    def xai_streaming_chunk(text: str, finish: bool = False) -> str:
        """Generate SSE chunk for xAI streaming."""
        if finish:
            chunk = {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1702339200,
                "model": "grok-code-fast-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
        else:
            chunk = {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1702339200,
                "model": "grok-code-fast-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": None
                    }
                ]
            }
        return f"data: {json.dumps(chunk)}\n\n"
