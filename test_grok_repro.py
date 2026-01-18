import asyncio
import os
import sys

# Add sdk-interface to sys.path
sys.path.append(os.path.join(os.getcwd(), "sdk-interface"))

# Mock environment variables for testing if .env is missing or for override
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
os.environ["GROK_API_KEY"] = "xai-test"

from app.grok_client import grok_client
from app.config import settings

async def test_grok():
    print(f"Grok API Key Configured: {settings.grok_api_key is not None}")
    
    # Test get_model
    print("\nTesting get_model('grok-2-latest')...")
    try:
        model = await grok_client.get_model("grok-2-latest")
        print(f"Success: {model}")
    except Exception as e:
        print(f"Error: {e}")

    # Test list_models
    print("\nTesting list_models()...")
    try:
        models = await grok_client.list_models()
        print(f"Found {len(models)} models")
        for m in models:
            print(f"- {m.id}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_grok())
