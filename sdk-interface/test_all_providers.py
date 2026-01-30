#!/usr/bin/env python3
"""Comprehensive test for all three providers."""

import requests
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get credentials from environment
API_KEY = os.getenv("API_KEYS", "").split(":")[1] if ":" in os.getenv("API_KEYS", "") else None
BASE_URL = "http://localhost:8060/v1"

if not API_KEY:
    print("❌ ERROR: No API key found in .env file")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_non_streaming(model_name: str, provider: str):
    """Test non-streaming completion."""
    print(f"\n{'='*60}")
    print(f"Testing {provider}: {model_name} (non-streaming)")
    print(f"{'='*60}")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say 'OK' only."}],
        "max_tokens": 50,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", "N/A")
            print(f"✅ SUCCESS")
            print(f"   Response: {content[:100]}")
            print(f"   Tokens: {tokens}")
            return True
        else:
            print(f"❌ FAILED - HTTP {response.status_code}")
            error = response.json().get("detail", response.text)
            print(f"   Error: {error[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def test_streaming(model_name: str, provider: str):
    """Test streaming completion."""
    print(f"\n{'='*60}")
    print(f"Testing {provider}: {model_name} (streaming)")
    print(f"{'='*60}")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say 'OK' only."}],
        "max_tokens": 50,
        "stream": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            chunks_received = 0
            content_chunks = []
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            chunks_received += 1
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            if 'content' in delta:
                                content_chunks.append(delta['content'])
                        except json.JSONDecodeError:
                            pass
            
            full_content = ''.join(content_chunks)
            print(f"✅ SUCCESS")
            print(f"   Chunks received: {chunks_received}")
            print(f"   Content: {full_content[:100]}")
            return True
        else:
            print(f"❌ FAILED - HTTP {response.status_code}")
            error = response.json().get("detail", response.text)
            print(f"   Error: {error[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def main():
    print("="*60)
    print("COMPREHENSIVE PROVIDER TESTING")
    print("="*60)
    
    # Get available models
    print("\nFetching available models...")
    try:
        response = requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to fetch models: {response.status_code}")
            sys.exit(1)
        
        models = response.json()["data"]
        print(f"✅ Found {len(models)} models")
    except Exception as e:
        print(f"❌ Exception fetching models: {e}")
        sys.exit(1)
    
    # Select one model from each provider
    gemini_model = next((m["id"] for m in models if "gemini" in m["id"].lower() and "flash" in m["id"].lower()), None)
    grok_model = next((m["id"] for m in models if "grok" in m["id"].lower()), None)
    anthropic_model = next((m["id"] for m in models if "claude" in m["id"].lower() and "sonnet" in m["id"].lower()), None)
    
    print(f"\nSelected models:")
    print(f"  Anthropic: {anthropic_model}")
    print(f"  Gemini: {gemini_model}")
    print(f"  Grok: {grok_model}")
    
    results = {}
    
    # Test Anthropic (Claude)
    if anthropic_model:
        print(f"\n\n{'#'*60}")
        print("# TESTING ANTHROPIC CLAUDE")
        print(f"{'#'*60}")
        results["Anthropic (non-streaming)"] = test_non_streaming(anthropic_model, "Anthropic")
        results["Anthropic (streaming)"] = test_streaming(anthropic_model, "Anthropic")
    else:
        print("\n⚠️  No Anthropic model found")
    
    # Test Gemini
    if gemini_model:
        print(f"\n\n{'#'*60}")
        print("# TESTING GOOGLE GEMINI")
        print(f"{'#'*60}")
        results["Gemini (non-streaming)"] = test_non_streaming(gemini_model, "Gemini")
        results["Gemini (streaming)"] = test_streaming(gemini_model, "Gemini")
    else:
        print("\n⚠️  No Gemini model found")
    
    # Test Grok
    if grok_model:
        print(f"\n\n{'#'*60}")
        print("# TESTING xAI GROK")
        print(f"{'#'*60}")
        results["Grok (non-streaming)"] = test_non_streaming(grok_model, "Grok")
        results["Grok (streaming)"] = test_streaming(grok_model, "Grok")
    else:
        print("\n⚠️  No Grok model found")
    
    # Print summary
    print("\n\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
