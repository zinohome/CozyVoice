"""Tests for the /health endpoint in CozyVoice main.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient with lifespan bypassed."""
    with patch("cozyvoice.main.load_config") as mock_cfg, \
         patch("cozyvoice.main.BrainClient") as mock_brain_cls, \
         patch("cozyvoice.providers.stt.openai_whisper.AsyncOpenAI", MagicMock()):

        mock_cfg.return_value = {}

        mock_brain = MagicMock()
        mock_brain.startup = AsyncMock()
        mock_brain.shutdown = AsyncMock()
        mock_brain_cls.return_value = mock_brain

        from cozyvoice.main import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_health_includes_version(client):
    from cozyvoice import __version__
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["version"] == __version__


def test_health_is_get_only(client):
    resp = client.post("/health")
    assert resp.status_code == 405
