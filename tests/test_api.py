from fastapi.testclient import TestClient
from app.main import app


def test_ask_endpoint_schema_and_sources():
    client = TestClient(app)
    payload = {"question": "Как мне подключиться к VPN?"}
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert "sources" in data and isinstance(data["sources"], list) and len(data["sources"]) >= 1
    assert "usage" in data and all(k in data["usage"] for k in ("prompt_tokens","completion_tokens","total_tokens"))