"""Tests for FastAPI server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mom.api.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModels:
    def test_list_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        assert "experts" in resp.json()


class TestHandle:
    def test_handle_math(self, client):
        resp = client.post("/handle", json={"input": "Calculate 2+2"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "metadata" in data

    def test_handle_missing_input(self, client):
        resp = client.post("/handle", json={})
        assert resp.status_code == 422


class TestBenchmark:
    def test_benchmark_default(self, client):
        resp = client.post("/benchmark", json={"mode": "A"})
        assert resp.status_code == 200
        data = resp.json()
        assert "mode" in data
        assert "accuracy" in data
