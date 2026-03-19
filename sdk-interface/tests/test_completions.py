"""
Tests for /v1/chat/completions endpoint across all providers.
"""

import json
import pytest
from fastapi.testclient import TestClient


class TestNonStreamingCompletions:
    """Test non-streaming completions for all providers."""

    def test_anthropic_completion(
        self,
        client: TestClient,
        test_models: dict[str, str],
    ):
        """Test Anthropic (Claude) non-streaming completion."""
        
        payload = {
            "model": test_models["anthropic"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Anthropic test!' and nothing else."}
            ],
            "stream": False,
            "max_tokens": 50,
        }
        
        response = client.post("/v1/chat/completions", json=payload)
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

    def test_gemini_completion(
        self,
        client: TestClient,
        test_models: dict[str, str],
    ):
        """Test Gemini non-streaming completion."""
        
        payload = {
            "model": test_models["gemini"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Gemini test!' and nothing else."}
            ],
            "stream": False,
            "max_tokens": 50,
        }
        
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        
        # Verify content type is JSON
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, f"Expected JSON but got: {content_type}"
        
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

    def test_grok_completion(
        self,
        client: TestClient,
        test_models: dict[str, str],
    ):
        """Test Grok non-streaming completion."""
        
        payload = {
            "model": test_models["grok"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Grok test!' and nothing else."}
            ],
            "stream": False,
            "max_tokens": 50,
        }
        
        response = client.post("/v1/chat/completions", json=payload)
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

    def test_anthropic_streaming(
        self,
        client: TestClient,
        test_models: dict[str, str],
    ):
        """Test Anthropic (Claude) streaming completion."""
        
        payload = {
            "model": test_models["anthropic"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Anthropic streaming test!' and nothing else."}
            ],
            "stream": True,
            "max_tokens": 50,
        }
        
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            
            # Verify content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE but got: {content_type}"
            
            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)
            
            assert len(chunks) > 0, "Expected at least one chunk"

    def test_gemini_streaming(
        self,
        client: TestClient,
        test_models: dict[str, str],
    ):
        """Test Gemini streaming completion."""
        
        payload = {
            "model": test_models["gemini"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Gemini streaming test!' and nothing else."}
            ],
            "stream": True,
            "max_tokens": 50,
        }
        
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            
            # Verify content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE but got: {content_type}"
            
            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)
            
            assert len(chunks) > 0, "Expected at least one chunk"

    def test_grok_streaming(
        self,
        client: TestClient,
        test_models: dict[str, str],
    ):
        """Test Grok streaming completion."""
        
        payload = {
            "model": test_models["grok"],
            "messages": [
                {"role": "user", "content": "Say 'Hello from Grok streaming test!' and nothing else."}
            ],
            "stream": True,
            "max_tokens": 50,
        }
        
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            
            # Verify content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE but got: {content_type}"
            
            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)
            
            assert len(chunks) > 0, "Expected at least one chunk"


class TestDeepResearch:
    """Test deep research functionality (Gemini)."""

    @pytest.mark.slow
    def test_deep_research_streaming(
        self,
        client: TestClient,
        test_models: dict[str, str],
    ):
        """Test deep research with streaming."""
        
        payload = {
            "model": test_models["gemini_deep_research"],
            "messages": [
                {"role": "user", "content": "What is 2+2? Just give the answer."}
            ],
            "stream": True,
            "max_tokens": 100,
        }
        
        # TestClient doesn't support timeout parameter, but handles long-running requests fine
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            
            # Verify content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE but got: {content_type}"
            
            chunks = []
            reasoning_chunks = []
            content_chunks = []
            
            for line in response.iter_lines():
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
