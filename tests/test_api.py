"""
File Name: test_api.py
Purpose: Programmatically hits our local API to make sure it doesn't crash when sent good/bad data.
Why it exists: To ensure future code changes don't accidentally break the API.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """
    What it does: Visits /health and checks for a 200 OK status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint_valid():
    """
    What it does: Sends a valid mock transaction to ensuring it returns a proper prediction.
    """
    payload = {
      "transaction_id": "T12345",
      "customer_id": "C999",
      "transaction_amount": 1500.50,
      "transaction_time": "2025-04-16T12:00:00",
      "merchant_category": "electronics",
      "is_international": 1,
      "time_since_last_txn": 300,
      "user_txn_count": 2,
      "user_mean_spend": 100.0,
      "dev_from_mean": 15.0,
      "transaction_velocity_24h": 5
    }
    
    response = client.post("/predict", json=payload)
    
    # We expect a success
    assert response.status_code == 200
    
    # We expect the payload return structure to match PredictionResponse
    data = response.json()
    assert "fraud_prediction" in data
    assert "fraud_probability" in data
    assert data["transaction_id"] == "T12345"

def test_predict_endpoint_invalid_data():
    """
    What it does: Intentionally sends broken data (missing amount) to make sure it is REJECTED cleanly with 422.
    """
    payload = {
      "transaction_id": "T12345"
      # missing everything else!
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # 422 Unprocessable Entity (Pydantic caught it)
