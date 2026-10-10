"""API-contract tests (model mocked; actual checkpoint isn't committed)."""
from io import BytesIO
###
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as service


class FakePredictor:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path

    def predict(self, image):
        assert image.mode == "RGB"
        return {
            "predicted_class": "defective",
            "confidence_score": 0.987,
            "probabilities": {"defective": 0.987, "normal": 0.013},
            "inference_time_ms": 1.25,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "test-secret-with-more-than-24-characters")
    monkeypatch.setattr(service, "Predictor", FakePredictor)
    with TestClient(service.app) as c:
        yield c


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (20, 20), (32, 128, 12)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"


def test_requires_api_key(client):
    response = client.post(
        "/predict", files={"file": ("x.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 401


def test_wrong_api_key(client):
    response = client.post(
        "/predict", headers={"X-API-Key": "wrong"},
        files={"file": ("x.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 401


def test_prediction_contract(client):
    response = client.post(
        "/predict", headers={"X-API-Key": "test-secret-with-more-than-24-characters"},
        files={"file": ("x.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["predicted_class"] == "defective"
    assert data["confidence_score"] == 0.987
    assert set(data["probabilities"]) == {"normal", "defective"}
    assert data["inference_time_ms"] >= 0


def test_invalid_image(client):
    response = client.post(
        "/predict", headers={"X-API-Key": "test-secret-with-more-than-24-characters"},
        files={"file": ("broken.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400


def test_wrong_media_type(client):
    response = client.post(
        "/predict", headers={"X-API-Key": "test-secret-with-more-than-24-characters"},
        files={"file": ("notes.txt", b"abc", "text/plain")},
    )
    assert response.status_code == 415


def test_oversized_upload(client):
    response = client.post(
        "/predict", headers={"X-API-Key": "test-secret-with-more-than-24-characters"},
        files={"file": ("big.jpg", b"x" * (10 * 1024 * 1024 + 1), "image/jpeg")},
    )
    assert response.status_code == 413
