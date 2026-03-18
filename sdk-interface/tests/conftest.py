"""
Pytest configuration and fixtures for SDK Interface tests.
"""

import os
import pytest
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Load test environment variables from ROOT directory
# Priority: root/.env.test > root/.env > environment variables
root_dir = Path(__file__).parent.parent.parent  # Go up to project root
env_test_path = root_dir / ".env.test"
env_path = root_dir / ".env"

if env_test_path.exists():
    load_dotenv(env_test_path, override=True)
elif env_path.exists():
    load_dotenv(env_path, override=False)

# Configure test environment for maximum performance
os.environ.setdefault("SDK__DB_PATH", ":memory:")  # In-memory SQLite for speed
os.environ.setdefault("SDK__TEST_MODE", "mock")
os.environ.setdefault("SDK__API_KEYS", "")  # Disable auth
os.environ.setdefault("SDK__GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("SDK__ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("SDK__GROK_API_KEY", "test-grok-key")
os.environ.setdefault("SDK__LOG_LEVEL", "ERROR")  # Reduce logging overhead


@pytest.fixture(scope="session")
def test_mode() -> str:
    """Get test mode: 'mock' or 'real'."""
    return os.getenv("SDK__TEST_MODE", "mock")


@pytest.fixture(scope="session")
def is_mock_mode(test_mode: str) -> bool:
    """Check if tests should use mocked responses."""
    return test_mode == "mock"


@pytest.fixture(scope="session")
def sdk_base_url() -> str:
    """Get SDK base URL from environment or use default."""
    return os.getenv("SDK__BASE_URL", "http://localhost:8060")


@pytest.fixture(scope="session")
def api_key() -> str:
    """Get API key from environment."""
    return os.getenv("SDK__API_KEY", "")


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    return os.getenv("SDK__ANTHROPIC_API_KEY", "")


@pytest.fixture(scope="session")
def google_api_key() -> str:
    """Get Google API key from environment."""
    return os.getenv("SDK__GOOGLE_API_KEY", "")


@pytest.fixture(scope="session")
def grok_api_key() -> str:
    """Get Grok API key from environment."""
    return os.getenv("SDK__GROK_API_KEY", "")


@pytest.fixture(scope="session")
def test_models() -> dict[str, str]:
    """Test models for each provider."""
    return {
        "anthropic": os.getenv("SDK__TEST_MODEL_ANTHROPIC", "claude-sonnet-4-5-20250929"),
        "gemini": os.getenv("SDK__TEST_MODEL_GEMINI", "gemini-2.0-flash-exp"),
        "gemini_deep_research": os.getenv(
            "SDK__TEST_MODEL_GEMINI_DEEP_RESEARCH", "deep-research-pro-preview-12-2025"
        ),
        "grok": os.getenv("SDK__TEST_MODEL_GROK", "grok-code-fast-1"),
    }


# Session-scoped client for maximum performance - created once and reused
@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    Create a FastAPI TestClient once per test session.
    
    Performance optimizations:
    - Session scope: App created once, not per test
    - In-memory SQLite: No disk I/O
    - Monkeypatched methods: No network calls
    - Cached mock data: Data created once
    """
    from app.models import ModelInfo, ChatCompletionResponse, ChatCompletionChunk
    
    from app.models import ChatMessage, ChatCompletionChoice, Usage, ChatCompletionStreamChoice
    import time
    
    # Pre-create simple mock data - bypass MockResponses to avoid schema mismatches
    _anthropic_models = [
        ModelInfo(id="claude-sonnet-4-5-20250929", owned_by="anthropic"),
        ModelInfo(id="claude-opus-4-5-20251101", owned_by="anthropic"),
    ]
    _google_models = [
        ModelInfo(id="gemini-2.0-flash-exp", owned_by="google"),
        ModelInfo(id="gemini-2.0-flash-thinking-exp", owned_by="google"),
        ModelInfo(id="deep-research-pro-preview-12-2025", owned_by="google"),
    ]
    _xai_models = [
        ModelInfo(id="grok-2-vision-1212", owned_by="xai"),
        ModelInfo(id="grok-code-fast-1", owned_by="xai"),
    ]
    
    # Create simple completion responses
    _timestamp = int(time.time())
    _anthropic_response = ChatCompletionResponse(
        id="chatcmpl-test",
        object="chat.completion",
        created=_timestamp,
        model="claude-sonnet-4-5-20250929",
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Test response"),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    _google_response = ChatCompletionResponse(
        id="chatcmpl-test",
        object="chat.completion",
        created=_timestamp,
        model="gemini-2.0-flash-exp",
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Test response"),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    _xai_response = ChatCompletionResponse(
        id="chatcmpl-test",
        object="chat.completion",
        created=_timestamp,
        model="grok-2-vision-1212",
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Test response"),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    
    # Create streaming chunks (delta is Any type, use dict)
    def _chunk_template(model):
        return ChatCompletionChunk(id="chatcmpl-test", object="chat.completion.chunk", created=_timestamp, model=model, choices=[ChatCompletionStreamChoice(index=0, delta={"content": "Test", "role": "assistant"}, finish_reason="stop")])
    _anthropic_chunk = _chunk_template("claude-sonnet-4-5-20250929")
    _google_chunk = _chunk_template("gemini-2.0-flash-exp")
    _xai_chunk = _chunk_template("grok-2-vision-1212")
    
    # Fast mock functions using pre-created data
    async def mock_anthropic_list_models(self):
        return _anthropic_models
    
    async def mock_gemini_list_models(self):
        return _google_models
    
    async def mock_grok_list_models(self):
        return _xai_models
    
    async def mock_anthropic_completion(self, request):
        return _anthropic_response
    
    async def mock_anthropic_streaming(self, request, db=None, previous_completion=None):
        # Yield SSE-formatted strings (same format as real clients)
        import json
        yield f"data: {json.dumps(_anthropic_chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"
    
    async def mock_gemini_completion(self, request):
        return _google_response
    
    async def mock_gemini_streaming(self, request, db=None, previous_completion=None):
        import json
        yield f"data: {json.dumps(_google_chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"
    
    async def mock_grok_completion(self, request):
        return _xai_response
    
    async def mock_grok_streaming(self, request, db=None, previous_completion=None):
        import json
        yield f"data: {json.dumps(_xai_chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"
    
    # Patch at import time before app loads
    import app.anthropic_client
    import app.gemini_client
    import app.grok_client
    
    app.anthropic_client.AnthropicClient.list_models = mock_anthropic_list_models
    app.anthropic_client.AnthropicClient.create_completion = mock_anthropic_completion
    app.anthropic_client.AnthropicClient.create_stream_completion = mock_anthropic_streaming
    
    app.gemini_client.GeminiClient.list_models = mock_gemini_list_models
    app.gemini_client.GeminiClient.create_completion = mock_gemini_completion
    app.gemini_client.GeminiClient.create_stream_completion = mock_gemini_streaming
    
    app.grok_client.GrokClient.list_models = mock_grok_list_models
    app.grok_client.GrokClient.create_completion = mock_grok_completion
    app.grok_client.GrokClient.create_stream_completion = mock_grok_streaming
    
    # Now import and create the app (it will use mocked methods)
    from app.main import app
    
    # Create TestClient once for entire session
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def skip_if_no_api_key(is_mock_mode: bool):
    """Skip test if required API keys are missing (only in real mode)."""
    def _skip(provider: str, api_key: str):
        if not is_mock_mode and not api_key:
            pytest.skip(f"{provider} API key not configured (set {provider.upper()}_API_KEY or TEST_MODE=mock)")
    return _skip


# Cache mock data at module level for performance
_cached_mock_responses = None

def get_cached_mock_responses():
    """Get cached mock responses to avoid recreating them."""
    global _cached_mock_responses
    if _cached_mock_responses is None:
        from tests.mocks import MockResponses
        _cached_mock_responses = MockResponses
    return _cached_mock_responses


# Remove unused respx mocks - we're using monkeypatch instead for better performance
