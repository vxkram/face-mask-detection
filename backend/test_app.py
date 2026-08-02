import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


def make_test_image():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), color=(120, 140, 200)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_index_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Face Mask Detector" in response.data


def test_predict_without_file_returns_400(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert "No file selected" in response.get_json()["error"]


def test_predict_rejects_non_image(client):
    data = {"file": (io.BytesIO(b"not an image"), "notes.txt", "text/plain")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Only image files" in response.get_json()["error"]


def test_predict_returns_503_when_model_missing(client, monkeypatch):
    def raise_not_found():
        raise app_module.ModelNotFoundError("Model not found. Run `python train.py` first.")

    monkeypatch.setattr(app_module, "get_model", raise_not_found)

    data = {"file": (make_test_image(), "photo.jpg", "image/jpeg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 503
    assert "Model not found" in response.get_json()["error"]


def test_predict_happy_path(client, monkeypatch):
    class FakeModel:
        def predict(self, batch, verbose=0):
            return np.array([[0.1, 0.8, 0.1]])

    monkeypatch.setattr(
        app_module, "get_model", lambda: (FakeModel(), ["partial_mask", "with_mask", "without_mask"])
    )

    data = {"file": (make_test_image(), "photo.jpg", "image/jpeg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200

    body = response.get_json()
    assert body["label"] == "with_mask"
    assert body["confidence"] == pytest.approx(0.8)
    assert set(body["probabilities"]) == {"partial_mask", "with_mask", "without_mask"}
