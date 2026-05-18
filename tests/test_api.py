"""API integration tests (require trained model for full coverage)."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or response.text.startswith("#")


def test_predict_validation_error():
    response = client.post("/predict", json={"prices": [], "steps": 1})
    assert response.status_code == 422


def test_health_includes_model_symbol():
    response = client.get("/health")
    data = response.json()
    assert "model_symbol" in data
