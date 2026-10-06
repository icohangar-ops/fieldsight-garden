"""API tests for FieldSight Garden. They run in mock mode and do not call Tinker."""
from __future__ import annotations

import io
import os
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.diagnose import CLASSES, DISCLAIMER, presentation
from app.main import create_app
from app.settings import Settings
from app.stats import CREATE_TABLE_SQL, INSERT_SQL, MemoryStore, open_store


def jpeg_bytes(color: tuple[int, int, int] = (40, 120, 60), size: int = 32, noise: bool = False) -> bytes:
    if noise:
        raw = os.urandom(size * size * 3)
        im = Image.frombytes("RGB", (size, size), raw)
    else:
        im = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    settings = Settings.from_env()
    settings = Settings(
        mock=True,
        load_model=False,
        model_path=settings.model_path,
        database_url=None,
        max_bytes=settings.max_bytes,
        rate_limit=100,
        rate_window_s=60,
        global_rate_limit=100,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health_mock(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mock"] is True
    assert body["ready"] is True
    assert body["failed"] is False
    assert body["storage"] == "memory"
    assert "TINKER_API_KEY" not in response.text
    assert body["model_path"].startswith("tinker://")


def test_page_and_chart(client: TestClient):
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    html = page.text
    assert 'capture="environment"' in html
    assert "speechSynthesis" in html
    assert "Done, go outside" in html
    assert "79.7%" in html
    assert "$0.098" in html
    assert "PlantDoc" in html
    assert "Creative Commons Attribution 4.0" in html
    assert "extension" in html.lower()
    assert "/chart.png" in html
    chart = client.get("/chart.png")
    assert chart.status_code == 200
    assert chart.headers["content-type"].startswith("image/png")
    assert chart.content.startswith(b"\x89PNG")


def test_diagnose_mock_shape(client: TestClient):
    before = client.get("/api/stats").json()["diagnoses_today"]
    image = jpeg_bytes()
    first = client.post("/api/diagnose", files={"image": ("leaf.jpg", image, "image/jpeg")})
    second = client.post("/api/diagnose", files={"image": ("leaf.jpg", image, "image/jpeg")})
    assert first.status_code == 200
    body = first.json()
    assert body["ok"] is True
    assert body["mock"] is True
    assert body["label"] in CLASSES
    assert body["label"] == second.json()["label"]
    plant, condition = presentation(body["label"])
    assert body["plant"] == plant
    assert body["condition"] == condition
    assert len(body["steps"]) == 3
    assert all(isinstance(step, str) and step for step in body["steps"])
    assert body["confidence"] is None
    assert body["disclaimer"] == DISCLAIMER
    assert body["sources"]
    assert all(source["url"].startswith("https://") for source in body["sources"])
    assert body["diagnoses_today"] == before + 1
    stats = client.get("/api/stats").json()
    assert stats["diagnoses_today"] == before + 2
    assert stats["storage"] == "memory"
    rows = client.app.state.store.rows
    assert rows[-1].label == body["label"]
    assert rows[-1].latency_s == body["latency_s"]
    assert rows[-1].created_at is not None
    assert not hasattr(rows[-1], "image")
    assert set(rows[-1].__dict__) == {"label", "latency_s", "created_at"}


def test_every_class_has_a_display_name():
    for label in CLASSES:
        plant, condition = presentation(label)
        assert plant
        assert condition


def test_unreadable_and_missing_are_rejected(client: TestClient):
    before = client.get("/api/stats").json()["diagnoses_today"]
    bad = client.post("/api/diagnose", files={"image": ("notes.txt", b"not a photo", "text/plain")})
    assert bad.status_code == 400
    empty = client.post("/api/diagnose", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400
    missing = client.post("/api/diagnose")
    assert missing.status_code == 400
    assert client.get("/api/stats").json()["diagnoses_today"] == before


def test_oversize_upload():
    settings = Settings(
        mock=True,
        load_model=False,
        model_path="tinker://example",
        max_bytes=400,
        rate_limit=10,
        global_rate_limit=10,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        blob = jpeg_bytes(size=96, noise=True)
        assert len(blob) > 400
        response = client.post("/api/diagnose", files={"image": ("big.jpg", blob, "image/jpeg")})
        assert response.status_code == 413
        assert client.get("/api/stats").json()["diagnoses_today"] == 0


def test_rate_limit_does_not_count_the_rejected_call():
    settings = Settings(
        mock=True,
        load_model=False,
        model_path="tinker://example",
        rate_limit=2,
        rate_window_s=60,
        global_rate_limit=100,
    )
    app = create_app(settings)
    image = jpeg_bytes()
    with TestClient(app) as client:
        assert client.post("/api/diagnose", files={"image": ("a.jpg", image, "image/jpeg")}).status_code == 200
        assert client.post("/api/diagnose", files={"image": ("a.jpg", image, "image/jpeg")}).status_code == 200
        blocked = client.post("/api/diagnose", files={"image": ("a.jpg", image, "image/jpeg")})
        assert blocked.status_code == 429
        assert blocked.headers["retry-after"] == "60"
        assert client.get("/api/stats").json()["diagnoses_today"] == 2


def test_global_rate_limit():
    settings = Settings(
        mock=True,
        load_model=False,
        model_path="tinker://example",
        rate_limit=100,
        global_rate_limit=1,
    )
    app = create_app(settings)
    image = jpeg_bytes(color=(10, 20, 30))
    with TestClient(app) as client:
        assert client.post("/api/diagnose", files={"image": ("a.jpg", image, "image/jpeg")}).status_code == 200
        blocked = client.post("/api/diagnose", files={"image": ("a.jpg", image, "image/jpeg")})
        assert blocked.status_code == 429
        assert "busy" in blocked.json()["detail"].lower()


def test_model_not_ready_returns_503_without_tinker():
    settings = Settings(mock=False, load_model=False, model_path="tinker://example", database_url=None)
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["ready"] is False
        assert health["mock"] is False
        response = client.post(
            "/api/diagnose",
            files={"image": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 503
        assert "tinker" not in sys.modules
        assert "transformers" not in sys.modules
        assert "serving.fieldsight_infer" not in sys.modules


def test_mock_client_does_not_import_tinker(client: TestClient):
    client.get("/health")
    client.post("/api/diagnose", files={"image": ("leaf.jpg", jpeg_bytes(color=(1, 2, 3)), "image/jpeg")})
    assert "tinker" not in sys.modules
    assert "transformers" not in sys.modules


def test_memory_store_and_sql_keep_images_out():
    store = MemoryStore()
    store.record("tomato early blight", 2.5)
    assert store.count_today() == 1
    blob = CREATE_TABLE_SQL + INSERT_SQL
    assert "image" not in blob.lower()
    assert "label" in INSERT_SQL
    assert "latency_s" in INSERT_SQL


def test_database_failure_falls_back_to_memory(monkeypatch):
    def boom(url: str):
        raise OSError("down")

    store = open_store("postgres://fieldsight:secret@127.0.0.1:1/fieldsight", connector=boom)
    assert store.kind == "memory"
    assert store.fallback is True
    store.record("apple scab", 1.2)
    assert store.count_today() == 1


def test_open_store_without_url_is_memory():
    store = open_store(None)
    assert store.kind == "memory"
    assert store.fallback is False
