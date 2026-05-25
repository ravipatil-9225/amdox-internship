from fastapi.testclient import TestClient
from src.api.main import app
from src.api.security import create_access_token

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "2.0.0"}

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

def test_inventory_authorized():
    token = create_access_token({"sub": "admin"})
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/v1/inventory/reorder", 
                           json={"sku_id": "SKU-1001", "store_id": "STORE-001"},
                           headers=headers)
    assert response.status_code == 200
    assert response.json()["sku_id"] == "SKU-1001"
    assert "recommended_reorder_qty" in response.json()

def test_churn_authorized():
    token = create_access_token({"sub": "admin"})
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/v1/predict/churn", 
                           json={"customer_id": "CUST-10000"},
                           headers=headers)
    assert response.status_code == 200
    assert response.json()["customer_id"] == "CUST-10000"
    assert "churn_probability" in response.json()
