import asyncio
import os
import sys

# Add sdk-interface to sys.path
sys.path.append(os.path.join(os.getcwd(), "sdk-interface"))

# Mock environment variables
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
os.environ["GROK_API_KEY"] = "xai-test" # Invalid key, but enough to initialize client

from app.grok_client import grok_client
from app.models import ChatCompletionRequest, ChatMessage
from app.main import create_chat_completion, app

async def test_completion():
    print("\nTesting create_chat_completion with 'grok-2-latest'...")
    
    # Mocking list_models to ensure validation passes even if API fails
    # actually getting the hardcoded list
    
    req = ChatCompletionRequest(
        model="grok-2-latest",
        messages=[ChatMessage(role="user", content="Hello")]
    )
    
    try:
        # This will try to call the real API with the test key
        # We expect an Authentication Error (401/400) from xAI, 
        # NOT a "Model not found" (404/400 from our app validation)
        await create_chat_completion(req)
        print("Success (unexpected with invalid key)")
    except Exception as e:
        print(f"Caught exception: {e}")
        # If it's HTTPException
        if hasattr(e, "status_code"):
            print(f"Status Code: {e.status_code}")
            print(f"Detail: {e.detail}")

if __name__ == "__main__":
    asyncio.run(test_completion())
