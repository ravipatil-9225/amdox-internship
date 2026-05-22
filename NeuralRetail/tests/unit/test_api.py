from fastapi.testclient import TestClient
from src.api.main import app
from src.api.security import create_access_token

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}

def test_demand_unauthorized():
    response = client.post("/api/v1/predict/demand", json={"sku_id": "123", "horizon_days": 7, "store_id": "456"})
    assert response.status_code == 401

def test_demand_authorized():
    token = create_access_token({"sub": "admin"})
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/v1/predict/demand", 
                           json={"sku_id": "123", "horizon_days": 7, "store_id": "456"},
                           headers=headers)
    assert response.status_code == 200
    assert response.json()["sku_id"] == "123"
