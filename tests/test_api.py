from fastapi.testclient import TestClient

from src.api.app import app
import src.pipelines.train as train_module

client = TestClient(app)


def test_health():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "/train" in data.get("endpoints", [])


def test_predict_minimal():
    payload = {"values": [1, 2, 3, 4, 5, 6, 7], "model_name": "XGBoost"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "XGBoost"
    assert isinstance(data["predictions"], list)


def test_forecast_minimal():
    payload = {"values": [1, 2, 3, 4, 5, 6, 7], "model_name": "XGBoost"}
    resp = client.post("/forecast?horizon=3", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 3


def test_train_endpoint(monkeypatch):
    # patch the long-running run() to a quick stub
    def fake_run(model_name=None):
        return {"scores": {"XGBoost": {"rmse": 1.0}}, "best_model": "XGBoost"}

    # the FastAPI app imports run directly, so patch both places
    monkeypatch.setattr(train_module, "run", fake_run)
    monkeypatch.setattr("src.api.app.run", fake_run)

    resp = client.post("/train", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["best_model"] == "XGBoost"
    assert "metrics" in data
