"""
Tests for /v1/chat/completions endpoint across all providers.
"""

import json
import pytest
import httpx


class TestNonStreamingCompletions:
    """Test non-streaming completions for all providers."""

    @pytest.mark.asyncio
    async def test_anthropic_completion(
        self,
        http_client: httpx.AsyncClient,
        test_models: dict[str, str],
        anthropic_api_key: str,
        skip_if_no_api_key,
    ):
        """Test Anthropic (Claude) non-streaming completion."""
        skip_if_no_api_key("anthropic", anthropic_api_key)
        
        payload = {
            "model": test_models["anthropic"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Anthropic test!' and nothing else."}
            ],
            "stream": False,
            "max_tokens": 50,
        }
        
        response = await http_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        
        # Verify content type is JSON
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, f"Expected JSON but got: {content_type}"
        
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert data["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_gemini_completion(
        self,
        http_client: httpx.AsyncClient,
        test_models: dict[str, str],
        google_api_key: str,
        skip_if_no_api_key,
    ):
        """Test Gemini non-streaming completion."""
        skip_if_no_api_key("google", google_api_key)
        
        payload = {
            "model": test_models["gemini"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Gemini test!' and nothing else."}
            ],
            "stream": False,
            "max_tokens": 50,
        }
        
        response = await http_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        
        # Verify content type is JSON
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, f"Expected JSON but got: {content_type}"
        
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

    @pytest.mark.asyncio
    async def test_grok_completion(
        self,
        http_client: httpx.AsyncClient,
        test_models: dict[str, str],
        grok_api_key: str,
        skip_if_no_api_key,
    ):
        """Test Grok non-streaming completion."""
        skip_if_no_api_key("grok", grok_api_key)
        
        payload = {
            "model": test_models["grok"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Grok test!' and nothing else."}
            ],
            "stream": False,
            "max_tokens": 50,
        }
        
        response = await http_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        
        # Verify content type is JSON
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, f"Expected JSON but got: {content_type}"
        
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]


class TestStreamingCompletions:
    """Test streaming completions for all providers."""

    @pytest.mark.asyncio
    async def test_anthropic_streaming(
        self,
        http_client: httpx.AsyncClient,
        test_models: dict[str, str],
        anthropic_api_key: str,
        skip_if_no_api_key,
    ):
        """Test Anthropic (Claude) streaming completion."""
        skip_if_no_api_key("anthropic", anthropic_api_key)
        
        payload = {
            "model": test_models["anthropic"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Anthropic streaming test!' and nothing else."}
            ],
            "stream": True,
            "max_tokens": 50,
        }
        
        async with http_client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            
            # Verify content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE but got: {content_type}"
            
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)
            
            assert len(chunks) > 0, "Expected at least one chunk"

    @pytest.mark.asyncio
    async def test_gemini_streaming(
        self,
        http_client: httpx.AsyncClient,
        test_models: dict[str, str],
        google_api_key: str,
        skip_if_no_api_key,
    ):
        """Test Gemini streaming completion."""
        skip_if_no_api_key("google", google_api_key)
        
        payload = {
            "model": test_models["gemini"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Gemini streaming test!' and nothing else."}
            ],
            "stream": True,
            "max_tokens": 50,
        }
        
        async with http_client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            
            # Verify content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE but got: {content_type}"
            
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)
            
            assert len(chunks) > 0, "Expected at least one chunk"

    @pytest.mark.asyncio
    async def test_grok_streaming(
        self,
        http_client: httpx.AsyncClient,
        test_models: dict[str, str],
        grok_api_key: str,
        skip_if_no_api_key,
    ):
        """Test Grok streaming completion."""
        skip_if_no_api_key("grok", grok_api_key)
        
        payload = {
            "model": test_models["grok"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Grok streaming test!' and nothing else."}
            ],
            "stream": True,
            "max_tokens": 50,
        }
        
        async with http_client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            
            # Verify content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE but got: {content_type}"
            
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)
            
            assert len(chunks) > 0, "Expected at least one chunk"


class TestDeepResearch:
    """Test deep research functionality (Gemini)."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_deep_research_streaming(
        self,
        http_client: httpx.AsyncClient,
        test_models: dict[str, str],
        google_api_key: str,
        skip_if_no_api_key,
    ):
        """Test deep research with streaming."""
        skip_if_no_api_key("google", google_api_key)
        
        payload = {
            "model": test_models["gemini_deep_research"],
            "messages": [
                {"role": "user", "content": "What is 2+2? Just give the answer."}
            ],
            "stream": True,
            "max_tokens": 100,
        }
        
        async with http_client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=60.0  # Deep research takes longer
        ) as response:
            assert response.status_code == 200
            
            # Verify content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE but got: {content_type}"
            
            chunks = []
            reasoning_chunks = []
            content_chunks = []
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        chunks.append(chunk)
                        
                        # Track reasoning vs content
                        if chunk.get("choices", [{}])[0].get("delta", {}).get("reasoning_content"):
                            reasoning_chunks.append(chunk)
                        if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                            content_chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Failed to parse JSON chunk: {data_str[:100]}... Error: {e}")
            
            assert len(chunks) > 0, "Expected at least one chunk"
            # Deep research should have reasoning chunks
            assert len(reasoning_chunks) > 0, "Expected reasoning chunks from deep research"
