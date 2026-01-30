#!/usr/bin/env python3
"""Simple test for each provider."""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEYS", "").split(":")[1] if ":" in os.getenv("API_KEYS", "") else None
BASE_URL = "http://localhost:8000/v1"

def test_model(model_name: str):
    """Test a specific model with a simple request."""
    print(f"\nTesting {model_name}...")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say 'OK' only."}],
        "max_tokens": 50,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"✅ {model_name}: {content[:100]}")
            return True
        else:
            print(f"❌ {model_name}: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ {model_name}: {e}")
        return False

# Get available models
resp = requests.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"})
models = resp.json()["data"]

# Find one model from each provider
gemini = next((m["id"] for m in models if "gemini" in m["id"].lower() and "2.5" in m["id"]), None)
grok = next((m["id"] for m in models if "grok" in m["id"].lower()), None)
anthropic = next((m["id"] for m in models if "claude" in m["id"].lower() and "sonnet" in m["id"].lower()), None)

print(f"\nFound models:")
print(f"  Gemini: {gemini}")
print(f"  Grok: {grok}")
print(f"  Anthropic: {anthropic}")

results = {}
if gemini:
    results["Gemini"] = test_model(gemini)
if grok:
    results["Grok"] = test_model(grok)
if anthropic:
    results["Anthropic"] = test_model(anthropic)

print("\n" + "="*50)
print("RESULTS:")
for name, success in results.items():
    print(f"  {'✅' if success else '❌'} {name}")
