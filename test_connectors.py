"""Small manual smoke test for the OpenAI-compatible SDK boundary.

Run with the stack up and a valid SDK token when authentication is enabled.
This intentionally tests the gateway, not provider SDKs directly.
"""

import asyncio
import os

import httpx


async def main() -> None:
    base_url = os.getenv("SDK__BASE_URL", "http://localhost:8060")
    model = os.getenv("SDK__TEST_MODEL", "anthropic/claude-sonnet-4.5")
    token = os.getenv("SDK__API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=120) as client:
        models = await client.get("/v1/models")
        models.raise_for_status()
        model_ids = {item["id"] for item in models.json()["data"]}
        print(f"Loaded {len(model_ids)} models")
        if model not in model_ids:
            print(f"Warning: {model} is not in the current provider catalog")

        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with: gateway ok"}],
                "max_tokens": 32,
            },
        )
        response.raise_for_status()
        print(response.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
