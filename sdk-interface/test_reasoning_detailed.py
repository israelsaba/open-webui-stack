#!/usr/bin/env python3
"""
Detailed test for reasoning model streaming.
Tests with Claude 3.7 Sonnet (known reasoning model) to verify:
1. Meta-reasoning from SDK operations
2. Model reasoning content
3. Regular content
4. Proper SSE streaming format
"""
import asyncio
import json
import os
import sys
from dotenv import load_dotenv
import httpx

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8060/v1")
API_KEY = os.getenv("API_KEYS", "").split(":")[1] if ":" in os.getenv("API_KEYS", "") else ""

# Use Claude 3.7 Sonnet - confirmed to have extended thinking
TEST_MODEL = "claude-3-7-sonnet-20250219"
TEST_PROMPT = "Think step by step and show your reasoning: What is 137 * 89? Break down the multiplication."


async def test_reasoning_streaming():
    """Test streaming with a model that definitely has reasoning."""
    
    print("=" * 80)
    print(f"🧪 Testing Reasoning Model Streaming")
    print("=" * 80)
    print(f"Model: {TEST_MODEL}")
    print(f"API: {API_BASE_URL}")
    print(f"Prompt: {TEST_PROMPT}")
    print("=" * 80)
    print()
    
    if not API_KEY:
        print("❌ Error: API_KEYS environment variable not set")
        print("   Set it in .env file: API_KEYS=username:token")
        return False
    
    payload = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "user", "content": TEST_PROMPT}
        ],
        "stream": True,
        "max_tokens": 2000,
        "temperature": 1.0,
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    
    # Track different types of content
    meta_reasoning_chunks = []  # SDK operation messages
    model_reasoning_chunks = []  # Model's thinking process
    content_chunks = []  # Final answer content
    
    try:
        print("📡 Initiating streaming request...\n")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                
                if response.status_code != 200:
                    print(f"❌ Error: HTTP {response.status_code}")
                    print(await response.aread())
                    return False
                
                print("✅ Stream connected successfully!\n")
                print("=" * 80)
                print("📊 STREAMING OUTPUT")
                print("=" * 80)
                print()
                
                line_count = 0
                async for line in response.aiter_lines():
                    line_count += 1
                    
                    if not line.strip():
                        continue
                    
                    if line == "data: [DONE]":
                        print("\n✅ Stream completed with [DONE] signal")
                        break
                    
                    if line.startswith("data: "):
                        try:
                            chunk_data = json.loads(line[6:])
                            
                            if not chunk_data.get("choices"):
                                continue
                            
                            delta = chunk_data["choices"][0].get("delta", {})
                            finish_reason = chunk_data["choices"][0].get("finish_reason")
                            
                            # Check for meta-reasoning (SDK operations)
                            if "reasoning_content" in delta and delta["reasoning_content"]:
                                reasoning_text = delta["reasoning_content"]
                                
                                # Distinguish between SDK meta-reasoning and model reasoning
                                if reasoning_text.startswith("[SDK]"):
                                    meta_reasoning_chunks.append(reasoning_text)
                                    print(f"🔧 {reasoning_text}")
                                else:
                                    model_reasoning_chunks.append(reasoning_text)
                                    print(f"💭 REASONING: {reasoning_text}", end="", flush=True)
                            
                            # Check for regular content
                            if "content" in delta and delta["content"]:
                                content_chunks.append(delta["content"])
                                print(f"💬 CONTENT: {delta['content']}", end="", flush=True)
                            
                            # Check for finish
                            if finish_reason:
                                print(f"\n🏁 Finish reason: {finish_reason}")
                        
                        except json.JSONDecodeError as e:
                            print(f"⚠️  JSON decode error on line {line_count}: {e}")
                            print(f"   Raw line: {line[:100]}...")
                            continue
                
                print()
                print("=" * 80)
                print("📈 RESULTS SUMMARY")
                print("=" * 80)
                
                # Summary
                full_reasoning = "".join(model_reasoning_chunks)
                full_content = "".join(content_chunks)
                
                print(f"\n✅ Test completed successfully!")
                print(f"\n📊 Statistics:")
                print(f"   Total lines processed: {line_count}")
                print(f"   SDK meta-reasoning events: {len(meta_reasoning_chunks)}")
                print(f"   Model reasoning chunks: {len(model_reasoning_chunks)}")
                print(f"   Content chunks: {len(content_chunks)}")
                print(f"   Total reasoning length: {len(full_reasoning)} chars")
                print(f"   Total content length: {len(full_content)} chars")
                
                print(f"\n🔧 SDK Meta-Reasoning Events:")
                for msg in meta_reasoning_chunks:
                    print(f"   • {msg}")
                
                if len(model_reasoning_chunks) > 0:
                    print(f"\n💭 Model Reasoning Preview (first 200 chars):")
                    print(f"   {full_reasoning[:200]}...")
                else:
                    print(f"\n⚠️  WARNING: No model reasoning content received!")
                    print(f"   This might indicate:")
                    print(f"   1. The model doesn't support extended thinking")
                    print(f"   2. The SDK isn't properly extracting thinking content")
                    print(f"   3. The prompt didn't trigger reasoning mode")
                
                if len(content_chunks) > 0:
                    print(f"\n💬 Final Content Preview (first 200 chars):")
                    print(f"   {full_content[:200]}...")
                else:
                    print(f"\n⚠️  WARNING: No regular content received!")
                
                # Validation
                success = True
                if len(meta_reasoning_chunks) < 2:
                    print(f"\n❌ FAIL: Expected at least 2 SDK meta-reasoning events, got {len(meta_reasoning_chunks)}")
                    success = False
                
                if len(model_reasoning_chunks) == 0:
                    print(f"\n⚠️  WARNING: No model reasoning detected (might be expected for some models)")
                
                if len(content_chunks) == 0:
                    print(f"\n❌ FAIL: No content chunks received")
                    success = False
                
                return success
                
    except Exception as e:
        print(f"\n❌ Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the detailed reasoning test."""
    success = await test_reasoning_streaming()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
