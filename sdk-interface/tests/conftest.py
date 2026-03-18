"""
Pytest configuration and fixtures for SDK Interface tests.
"""

import os
import pytest
import httpx
from typing import AsyncGenerator


@pytest.fixture(scope="session")
def sdk_base_url() -> str:
    """Get SDK base URL from environment or use default."""
    return os.getenv("SDK_BASE_URL", "http://localhost:8060")


@pytest.fixture(scope="session")
def api_key() -> str:
    """Get API key from environment."""
    return os.getenv("SDK_API_KEY", "")


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    return os.getenv("ANTHROPIC_API_KEY", "")


@pytest.fixture(scope="session")
def google_api_key() -> str:
    """Get Google API key from environment."""
    return os.getenv("GOOGLE_API_KEY", "")


@pytest.fixture(scope="session")
def grok_api_key() -> str:
    """Get Grok API key from environment."""
    return os.getenv("GROK_API_KEY", "")


@pytest.fixture(scope="session")
def test_models() -> dict[str, str]:
    """Test models for each provider."""
    return {
        "anthropic": os.getenv("TEST_MODEL_ANTHROPIC", "claude-sonnet-4-5-20250929"),
        "gemini": os.getenv("TEST_MODEL_GEMINI", "gemini-2.0-flash-exp"),
        "gemini_deep_research": os.getenv(
            "TEST_MODEL_GEMINI_DEEP_RESEARCH", "deep-research-pro-preview-12-2025"
        ),
        "grok": os.getenv("TEST_MODEL_GROK", "grok-code-fast-1"),
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
def skip_if_no_api_key():
    """Skip test if required API keys are missing."""
    def _skip(provider: str, api_key: str):
        if not api_key:
            pytest.skip(f"{provider} API key not configured (set {provider.upper()}_API_KEY)")
    return _skip
