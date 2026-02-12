import json
from fastapi.testclient import TestClient

from app.main import app


class DummyModel:
    def predict(self, arr, sfreq=None, max_length=128, num_beams=1):
        # return single prediction and confidence
        return ["hello world"], [0.95]


def test_infer_endpoint():
    client = TestClient(app)
    # inject dummy model into app state
    app.state.model = DummyModel()
    payload = {"signals": [[0.0] * 100] * 4, "sfreq": 500}
    resp = client.post("/infer", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "hello world"
    assert 0.0 <= data["confidence"] <= 1.0

