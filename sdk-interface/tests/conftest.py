"""
Pytest configuration and fixtures for SDK Interface tests.
"""

import os
import pytest
import httpx
import respx
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv

# Load test environment variables from ROOT directory
# Priority: root/.env.test > root/.env > environment variables
root_dir = Path(__file__).parent.parent.parent  # Go up to project root
env_test_path = root_dir / ".env.test"
env_path = root_dir / ".env"

if env_test_path.exists():
    load_dotenv(env_test_path, override=True)
elif env_path.exists():
    load_dotenv(env_path, override=False)


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


@pytest.fixture
async def http_client(sdk_base_url: str, api_key: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an HTTP client for testing."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    async with httpx.AsyncClient(
        base_url=sdk_base_url,
        headers=headers,
        timeout=30.0
    ) as client:
        yield client


@pytest.fixture(scope="session")
def skip_if_no_api_key(is_mock_mode: bool):
    """Skip test if required API keys are missing (only in real mode)."""
    def _skip(provider: str, api_key: str):
        if not is_mock_mode and not api_key:
            pytest.skip(f"{provider} API key not configured (set {provider.upper()}_API_KEY or TEST_MODE=mock)")
    return _skip


@pytest.fixture
def mock_anthropic_api(is_mock_mode: bool):
    """Mock Anthropic API endpoints."""
    if not is_mock_mode:
        yield None
        return
    
    from tests.mocks import MockResponses
    
    with respx.mock:
        # Mock models endpoint
        respx.get("https://api.anthropic.com/v1/models").mock(
            return_value=httpx.Response(200, json=MockResponses.anthropic_models_list())
        )
        
        # Mock chat completions endpoint
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MockResponses.anthropic_completion())
        )
        
        yield respx


@pytest.fixture
def mock_google_api(is_mock_mode: bool):
    """Mock Google API endpoints."""
    if not is_mock_mode:
        yield None
        return
    
    from tests.mocks import MockResponses
    
    with respx.mock:
        # Mock models endpoint
        respx.get(url__startswith="https://generativelanguage.googleapis.com/v1beta/models").mock(
            return_value=httpx.Response(200, json=MockResponses.google_models_list())
        )
        
        # Mock chat completions endpoint  
        respx.post(url__startswith="https://generativelanguage.googleapis.com/v1beta/models/").mock(
            return_value=httpx.Response(200, json=MockResponses.google_completion())
        )
        
        # Mock interactions endpoint (Deep Research)
        respx.post("https://generativelanguage.googleapis.com/v1beta/interactions").mock(
            return_value=httpx.Response(200, json=MockResponses.google_deep_research_interaction())
        )
        
        respx.get(url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/interactions/.*").mock(
            return_value=httpx.Response(200, json=MockResponses.google_deep_research_complete())
        )
        
        yield respx


@pytest.fixture
def mock_xai_api(is_mock_mode: bool):
    """Mock xAI API endpoints."""
    if not is_mock_mode:
        yield None
        return
    
    from tests.mocks import MockResponses
    
    with respx.mock:
        # Mock models endpoint
        respx.get("https://api.x.ai/v1/models").mock(
            return_value=httpx.Response(200, json=MockResponses.xai_models_list())
        )
        
        # Mock chat completions endpoint
        respx.post("https://api.x.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MockResponses.xai_completion())
        )
        
        yield respx
