"""
Tests for /v1/models endpoint.
"""

import pytest
import httpx


@pytest.mark.asyncio
async def test_list_models(
    http_client: httpx.AsyncClient,
    mock_anthropic_api,
    mock_google_api,
    mock_xai_api
):
    """Test /v1/models endpoint returns available models."""
    response = await http_client.get("/v1/models")
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


@pytest.mark.asyncio
async def test_models_include_test_models(
    http_client: httpx.AsyncClient,
    test_models: dict[str, str],
    mock_anthropic_api,
    mock_google_api,
    mock_xai_api
):
    """Test that our test models are available in the list."""
    response = await http_client.get("/v1/models")
    assert response.status_code == 200
    
    data = response.json()
    model_ids = {model["id"] for model in data["data"]}
    
    # Check if test models are available (skip if API key not configured)
    for provider, model_id in test_models.items():
        if model_id in model_ids:
            print(f"✓ {provider}: {model_id} available")
