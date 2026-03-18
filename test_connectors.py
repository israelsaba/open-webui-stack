#!/usr/bin/env python3
"""
Test script to verify all SDK interface connectors are working.
Tests Anthropic, Gemini (including deep-research), and Grok models.
"""

import asyncio
import httpx
import json
import os
from typing import Any

# Configuration
SDK_BASE_URL = os.getenv("SDK_BASE_URL", "http://192.168.2.4:8060")
API_KEY = os.getenv("SDK_API_KEY", "")

# Test models for each provider
TEST_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-2.0-flash-exp",
    "gemini_deep_research": "deep-research-pro-preview-12-2025",
    "grok": "grok-code-fast-1",
}

async def test_models_list(client: httpx.AsyncClient) -> bool:
    """Test /v1/models endpoint"""
    print("\n=== Testing /v1/models ===")
    try:
        response = await client.get(f"{SDK_BASE_URL}/v1/models")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Successfully fetched {len(data.get('data', []))} models")
        
        # Check if our test models are available
        model_ids = {model['id'] for model in data.get('data', [])}
        for provider, model_id in TEST_MODELS.items():
            if model_id in model_ids:
                print(f"  ✓ {provider}: {model_id} available")
            else:
                print(f"  ✗ {provider}: {model_id} NOT FOUND")
        
        return True
    except Exception as e:
        print(f"✗ Failed to fetch models: {e}")
        return False


async def test_completion(
    client: httpx.AsyncClient, 
    model: str, 
    provider_name: str,
    stream: bool = False
) -> bool:
    """Test chat completion for a model"""
    print(f"\n=== Testing {provider_name} ({model}) {'[STREAMING]' if stream else '[NON-STREAMING]'} ===")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello from SDK interface test!' and nothing else."}
        ],
        "stream": stream,
        "max_tokens": 50
    }
    
    try:
        if stream:
            async with client.stream("POST", f"{SDK_BASE_URL}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'text/event-stream' not in content_type:
                    print(f"✗ Wrong content-type: {content_type} (expected text/event-stream)")
                    return False
                
                chunks = []
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            print(f"✗ Failed to parse JSON chunk: {data_str}")
                            return False
                
                print(f"✓ Received {len(chunks)} chunks")
                print(f"✓ Content-Type: {content_type}")
                return True
        else:
            response = await client.post(f"{SDK_BASE_URL}/v1/chat/completions", json=payload)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type:
                print(f"✗ Wrong content-type: {content_type} (expected application/json)")
                print(f"✗ Response body: {response.text[:200]}")
                return False
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"✓ Response: {content[:100]}")
            print(f"✓ Content-Type: {content_type}")
            return True
            
    except httpx.HTTPStatusError as e:
        print(f"✗ HTTP {e.response.status_code}: {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_deep_research(client: httpx.AsyncClient) -> bool:
    """Test deep research with streaming"""
    model = TEST_MODELS["gemini_deep_research"]
    print(f"\n=== Testing Deep Research ({model}) ===")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 2+2? Just give the answer."}
        ],
        "stream": True,
        "max_tokens": 100
    }
    
    try:
        async with client.stream("POST", f"{SDK_BASE_URL}/v1/chat/completions", json=payload, timeout=30.0) as response:
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'text/event-stream' not in content_type:
                print(f"✗ Wrong content-type: {content_type} (expected text/event-stream)")
                return False
            
            chunks = []
            reasoning_chunks = []
            content_chunks = []
            
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        chunks.append(chunk)
                        
                        # Track reasoning vs content
                        if chunk.get('choices', [{}])[0].get('delta', {}).get('reasoning_content'):
                            reasoning_chunks.append(chunk)
                        if chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                            content_chunks.append(chunk)
                            
                    except json.JSONDecodeError:
                        print(f"✗ Failed to parse JSON: {data_str}")
                        return False
            
            print(f"✓ Total chunks: {len(chunks)}")
            print(f"✓ Reasoning chunks: {len(reasoning_chunks)}")
            print(f"✓ Content chunks: {len(content_chunks)}")
            print(f"✓ Content-Type: {content_type}")
            return True
            
    except httpx.HTTPStatusError as e:
        print(f"✗ HTTP {e.response.status_code}: {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def main():
    """Run all tests"""
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        print(f"Testing SDK Interface at {SDK_BASE_URL}")
        
        # Test health endpoint
        print("\n=== Testing /health ===")
        try:
            response = await client.get(f"{SDK_BASE_URL}/health")
            response.raise_for_status()
            print(f"✓ Health check passed: {response.json()}")
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return
        
        # Test models list
        await test_models_list(client)
        
        # Test each provider (non-streaming)
        results = {}
        for provider, model in TEST_MODELS.items():
            if provider == "gemini_deep_research":
                continue  # Skip for now, will test separately
            results[f"{provider}_non_stream"] = await test_completion(
                client, model, provider, stream=False
            )
        
        # Test streaming
        for provider, model in TEST_MODELS.items():
            if provider == "gemini_deep_research":
                continue
            results[f"{provider}_stream"] = await test_completion(
                client, model, provider, stream=True
            )
        
        # Test deep research
        results["deep_research"] = await test_deep_research(client)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
