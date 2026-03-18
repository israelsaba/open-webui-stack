"""
Tests for health and basic endpoints.
"""

import pytest
import httpx


@pytest.mark.asyncio
async def test_health_check(http_client: httpx.AsyncClient):
    """Test /health endpoint returns healthy status."""
    response = await http_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint(http_client: httpx.AsyncClient):
    """Test root endpoint returns API information."""
    response = await http_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert "models" in data
