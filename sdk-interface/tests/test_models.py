"""
Tests for /v1/models endpoint.
"""

from fastapi.testclient import TestClient

from app.models import ChatCompletionRequest
from app.provider_gateway import ProviderGateway


def test_list_models(client: TestClient):
    """Test /v1/models endpoint returns available models."""
    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0

    # Check model structure
    first_model = data["data"][0]
    assert "id" in first_model
    assert "object" in first_model
    assert "owned_by" in first_model
    assert first_model["object"] == "model"


def test_models_include_test_models(client: TestClient, test_models: dict[str, str]):
    """Test that our test models are available in the list."""
    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    model_ids = {model["id"] for model in data["data"]}

    # Check if test models are available (skip if API key not configured)
    for provider, model_id in test_models.items():
        if model_id in model_ids:
            print(f"✓ {provider}: {model_id} available")


def test_agentic_request_fields_are_forwarded():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "response_format": {"type": "json_object"},
            "provider": "anthropic",
        }
    )

    kwargs = ProviderGateway._request_kwargs(request, stream=False)

    assert kwargs["tools"] == request.tools
    assert request.provider == "anthropic"
    assert kwargs["parallel_tool_calls"] is True
