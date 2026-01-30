#!/usr/bin/env python3
"""Test main models from each provider."""

import os
import requests
import json
import sys

# Get API key from env file
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEYS", "").split(":")[1] if ":" in os.getenv("API_KEYS", "") else None
if not API_KEY:
    print("ERROR: No API key found in .env file")
    sys.exit(1)

BASE_URL = "http://localhost:8000/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_model(model_name: str, prompt: str = "Hello! Please respond with just 'OK' if you can hear me."):
    """Test a specific model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ SUCCESS")
            print(f"Response: {content[:200]}")
            return True
        else:
            print(f"❌ FAILED with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def list_models():
    """List all available models."""
    print("\nFetching available models...")
    try:
        response = requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            models = response.json()["data"]
            print(f"Found {len(models)} models")
            return models
        else:
            print(f"Failed to fetch models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception fetching models: {e}")
        return []

if __name__ == "__main__":
    print("Starting model tests...")
    
    # List all models first
    all_models = list_models()
    
    # Test main models from each provider
    # Pick models from the available list
    anthropic_model = next((m["id"] for m in all_models if "claude" in m["id"].lower() and "sonnet" in m["id"].lower()), None)
    gemini_model = next((m["id"] for m in all_models if "gemini" in m["id"].lower() and "flash" in m["id"].lower()), None)
    grok_model = next((m["id"] for m in all_models if "grok" in m["id"].lower()), None)
    
    test_models = []
    if anthropic_model:
        test_models.append((anthropic_model, f"Anthropic ({anthropic_model})"))
    if gemini_model:
        test_models.append((gemini_model, f"Google Gemini ({gemini_model})"))
    if grok_model:
        test_models.append((grok_model, f"xAI Grok ({grok_model})"))
    
    if not test_models:
        print("ERROR: No test models found!")
        sys.exit(1)
    
    results = {}
    for model_id, description in test_models:
        print(f"\n\nTesting {description}...")
        results[description] = test_model(model_id)
    
    # Print summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for description, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")
    
    # Exit with error code if any test failed
    if not all(results.values()):
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)
