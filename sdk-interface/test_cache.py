#!/usr/bin/env python3
"""Test that get_client caching works."""

import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEYS", "").split(":")[1] if ":" in os.getenv("API_KEYS", "") else None
BASE_URL = "http://localhost:8060/v1"

def make_request(model_name: str):
    """Make a simple request."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
        "stream": False
    }
    
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        timeout=30
    )
    elapsed = time.time() - start
    
    return response.status_code == 200, elapsed

print("Testing get_client cache...")
print("="*60)

# Get a grok model
resp = requests.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"})
grok_model = next((m["id"] for m in resp.json()["data"] if "grok" in m["id"].lower()), None)

if not grok_model:
    print("❌ No Grok model found")
    exit(1)

print(f"Using model: {grok_model}\n")

# First request - should be slower (cache miss)
print("Request 1 (cache miss expected)...")
success1, time1 = make_request(grok_model)
print(f"  Status: {'✅' if success1 else '❌'}")
print(f"  Time: {time1:.3f}s")

# Second request - should be faster (cache hit)
print("\nRequest 2 (cache hit expected)...")
success2, time2 = make_request(grok_model)
print(f"  Status: {'✅' if success2 else '❌'}")
print(f"  Time: {time2:.3f}s")

# Third request - should also be fast (cache hit)
print("\nRequest 3 (cache hit expected)...")
success3, time3 = make_request(grok_model)
print(f"  Status: {'✅' if success3 else '❌'}")
print(f"  Time: {time3:.3f}s")

print("\n" + "="*60)
print("RESULTS:")
print(f"  Request 1: {time1:.3f}s")
print(f"  Request 2: {time2:.3f}s (speedup: {time1/time2:.2f}x)")
print(f"  Request 3: {time3:.3f}s (speedup: {time1/time3:.2f}x)")

if success1 and success2 and success3:
    print("\n✅ All requests successful")
    if time2 < time1 * 0.9 and time3 < time1 * 0.9:
        print("✅ Cache is working! Subsequent requests are faster")
    else:
        print("⚠️  Requests completed but no significant speedup detected")
        print("   (This might be normal if network/API is very fast)")
else:
    print("\n❌ Some requests failed")
