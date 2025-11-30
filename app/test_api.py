"""
API Tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["status"] == "online"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")
    # Might be 503 if model not loaded, or 200 if loaded
    assert response.status_code in [200, 503]


def test_predict_endpoint_structure():
    """Test prediction endpoint structure"""
    # Sample request
    payload = {
        "features": {
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "priceUsd_lag_1": 45000.0,
            "priceUsd_rolling_mean_6": 44800.0,
            "price_pct_change_1h": 0.002,
            "volatility_6h": 0.015
        }
    }
    
    response = client.post("/predict", json=payload)
    
    # Might be 503 if model not loaded
    assert response.status_code in [200, 400, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert "timestamp" in data


def test_predict_missing_features():
    """Test prediction with missing features"""
    payload = {
        "features": {
            "hour_sin": 0.5
            # Missing other required features
        }
    }
    
    response = client.post("/predict", json=payload)
    # Should be 400 (bad request) or 503 (model not loaded)
    assert response.status_code in [400, 503]


def test_predict_invalid_data():
    """Test prediction with invalid data"""
    payload = {
        "features": "invalid"  # Should be dict
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])