"""
Deep Research implementation for Gemini models.

Deep Research is a feature of Gemini 2.0 Flash Thinking models that provides
extended, research-style thinking with detailed reasoning about complex topics.
"""
import logging
import time
from collections.abc import AsyncIterator

from app.gemini_client import gemini_client
from app.models import ChatCompletionRequest, ChatCompletionChunk, ChatCompletionStreamChoice

logger = logging.getLogger(__name__)


DEEP_RESEARCH_SYSTEM_PROMPT = """You are a deep research assistant. When given a question or topic:

1. Break down the topic into key components and subtopics
2. Research each component thoroughly, showing your thinking process
3. Consider multiple perspectives and approaches
4. Synthesize information from different angles
5. Provide detailed reasoning and analysis
6. Draw well-supported conclusions

Use the <thinking> tags to show your research process, analysis, and reasoning.
Be thorough, methodical, and comprehensive in your research."""


async def create_deep_research_stream(
    request: ChatCompletionRequest
) -> AsyncIterator[str]:
    """
    Create a deep research streaming completion using Gemini Thinking models.
    
    This wraps the user's query with a deep research system prompt and uses
    Gemini's thinking capabilities to provide extensive reasoning.
    
    Args:
        request: Chat completion request
    
    Yields:
        SSE-formatted streaming chunks with reasoning_content and content
    """
    # Inject deep research system prompt
    messages = request.messages.copy()
    
    # Add or prepend system message
    system_msg_exists = any(msg.role == "system" for msg in messages)
    if not system_msg_exists:
        from app.models import ChatMessage
        messages.insert(0, ChatMessage(
            role="system",
            content=DEEP_RESEARCH_SYSTEM_PROMPT
        ))
    else:
        # Enhance existing system message
        for msg in messages:
            if msg.role == "system":
                msg.content = f"{DEEP_RESEARCH_SYSTEM_PROMPT}\n\n{msg.content}"
                break
    
    # Create modified request with enhanced prompts
    research_request = ChatCompletionRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens or 8000,  # Allow longer responses for research
        stream=True,
        top_p=request.top_p,
        stop=request.stop,
        reasoning_effort="high",  # Always use high reasoning for deep research
    )
    
    completion_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    
    # Send meta-reasoning: Starting deep research
    yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[Deep Research] Initializing comprehensive research mode...'}, finish_reason=None)]).model_dump_json()}\n\n"
    
    yield f"data: {ChatCompletionChunk(id=completion_id, created=created, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta={'reasoning_content': '[Deep Research] Analyzing topic and identifying key research areas...'}, finish_reason=None)]).model_dump_json()}\n\n"
    
    # Stream through Gemini client
    async for chunk in gemini_client.create_stream_completion(research_request):
        yield chunk
