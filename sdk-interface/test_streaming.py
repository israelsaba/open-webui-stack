#!/usr/bin/env python3
"""
Test script for streaming completions with reasoning support.
Tests the main model for each provider: Anthropic, Gemini, and Grok.
"""
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Test configuration
TESTS = [
    {
        "name": "Anthropic Claude Sonnet",
        "model": "claude-3-7-sonnet-20250219",
        "has_reasoning": True,  # Claude has extended thinking
    },
    {
        "name": "Gemini 2.0 Flash Thinking",
        "model": "gemini-2.0-flash-thinking-exp",
        "has_reasoning": True,
    },
    {
        "name": "Grok 2",
        "model": "grok-2-latest",
        "has_reasoning": False,  # Grok doesn't have reasoning yet
    },
]

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8060/v1")


async def test_streaming(model: str, name: str, has_reasoning: bool):
    """Test streaming for a specific model."""
    import httpx
    
    print(f"\n{'='*60}")
    print(f"Testing: {name} ({model})")
    print(f"{'='*60}\n")
    
    # Use a simple test prompt
    prompt = "Explain quantum entanglement in simple terms."
    if has_reasoning:
        prompt = "Think step by step: What is 15 * 24?"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "max_tokens": 500,
    }
    
    # Get API key from environment
    api_key = os.getenv("API_KEYS", "").split(":")[1] if ":" in os.getenv("API_KEYS", "") else ""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                return False
            
            # Process streaming response
            content_chunks = []
            reasoning_chunks = []
            
            async for line in response.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                
                if line.startswith("data: "):
                    try:
                        chunk_data = json.loads(line[6:])
                        if chunk_data.get("choices"):
                            delta = chunk_data["choices"][0].get("delta", {})
                            
                            # Check for regular content
                            if "content" in delta and delta["content"]:
                                content_chunks.append(delta["content"])
                                print(delta["content"], end="", flush=True)
                            
                            # Check for reasoning content
                            if "reasoning_content" in delta and delta["reasoning_content"]:
                                reasoning_chunks.append(delta["reasoning_content"])
                                print(f"[THINKING: {delta['reasoning_content']}]", end="", flush=True)
                    except json.JSONDecodeError as e:
                        print(f"\n⚠️  JSON decode error: {e}")
                        continue
            
            print("\n")
            
            # Summary
            full_content = "".join(content_chunks)
            full_reasoning = "".join(reasoning_chunks)
            
            print(f"\n✅ Streaming completed successfully!")
            print(f"   Content length: {len(full_content)} chars")
            if has_reasoning:
                print(f"   Reasoning length: {len(full_reasoning)} chars")
                if len(full_reasoning) == 0:
                    print(f"   ⚠️  Expected reasoning content but got none")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all streaming tests."""
    print("🧪 Testing Streaming with Reasoning Support")
    print("=" * 60)
    
    results = {}
    for test in TESTS:
        success = await test_streaming(
            test["model"],
            test["name"],
            test["has_reasoning"]
        )
        results[test["name"]] = success
        await asyncio.sleep(2)  # Rate limiting
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")


if __name__ == "__main__":
    asyncio.run(main())
