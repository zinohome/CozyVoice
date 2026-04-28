"""Tests for the /health/ready endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _base_patches():
    """Context managers that suppress infrastructure initialisation."""
    mock_cfg = patch("cozyvoice.main.load_config", return_value={})
    mock_brain_cls = patch("cozyvoice.main.BrainClient")
    return mock_cfg, mock_brain_cls


def _make_brain_mock(brain_status):
    """Return a BrainClient mock whose _client behaves per brain_status.

    brain_status:
        200         → GET /health returns 200
        int != 200  → GET /health returns that HTTP status
        Exception() → GET /health raises that exception
        None        → _client is None (not initialised)
    """
    brain = MagicMock()
    brain.startup = AsyncMock()
    brain.shutdown = AsyncMock()

    if brain_status is None:
        brain._client = None
    else:
        http_client = MagicMock()
        if isinstance(brain_status, Exception):
            http_client.get = AsyncMock(side_effect=brain_status)
        else:
            resp = MagicMock()
            resp.status_code = brain_status
            http_client.get = AsyncMock(return_value=resp)
        brain._client = http_client

    return brain


@pytest.fixture
def client_all_ok():
    mock_cfg, mock_brain_cls = _base_patches()
    with mock_cfg, mock_brain_cls as brain_cls:
        brain_cls.return_value = _make_brain_mock(200)
        from cozyvoice.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            # STT/TTS are built from empty config → MockSTT/MockTTS instances
            yield c


@pytest.fixture
def client_brain_down():
    mock_cfg, mock_brain_cls = _base_patches()
    with mock_cfg, mock_brain_cls as brain_cls:
        brain_cls.return_value = _make_brain_mock(ConnectionError("refused"))
        from cozyvoice.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def client_brain_http_503():
    mock_cfg, mock_brain_cls = _base_patches()
    with mock_cfg, mock_brain_cls as brain_cls:
        brain_cls.return_value = _make_brain_mock(503)
        from cozyvoice.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def client_brain_not_configured():
    mock_cfg, mock_brain_cls = _base_patches()
    with mock_cfg, mock_brain_cls as brain_cls:
        brain_cls.return_value = _make_brain_mock(None)
        from cozyvoice.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def client_stt_missing():
    mock_cfg, mock_brain_cls = _base_patches()
    with mock_cfg, mock_brain_cls as brain_cls:
        brain_cls.return_value = _make_brain_mock(200)
        from cozyvoice.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            app.state.stt = None  # force missing
            yield c


@pytest.fixture
def client_tts_missing():
    mock_cfg, mock_brain_cls = _base_patches()
    with mock_cfg, mock_brain_cls as brain_cls:
        brain_cls.return_value = _make_brain_mock(200)
        from cozyvoice.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            app.state.tts = None  # force missing
            yield c


# ── happy path ─────────────────────────────────────────────────────────────────

def test_ready_all_ok_returns_200(client_all_ok):
    r = client_all_ok.get("/health/ready")
    assert r.status_code == 200


def test_ready_all_ok_status_ready(client_all_ok):
    r = client_all_ok.get("/health/ready")
    assert r.json()["status"] == "ready"


def test_ready_all_ok_checks_present(client_all_ok):
    checks = client_all_ok.get("/health/ready").json()["checks"]
    assert checks["brain"] == "ok"
    assert checks["stt"] == "ok"
    assert checks["tts"] == "ok"


# ── brain failures ─────────────────────────────────────────────────────────────

def test_ready_brain_exception_returns_503(client_brain_down):
    r = client_brain_down.get("/health/ready")
    assert r.status_code == 503


def test_ready_brain_exception_status_degraded(client_brain_down):
    r = client_brain_down.get("/health/ready")
    assert r.json()["status"] == "degraded"


def test_ready_brain_exception_check_down(client_brain_down):
    checks = client_brain_down.get("/health/ready").json()["checks"]
    assert checks["brain"].startswith("down:")


def test_ready_brain_http_503_returns_503(client_brain_http_503):
    r = client_brain_http_503.get("/health/ready")
    assert r.status_code == 503


def test_ready_brain_http_503_check(client_brain_http_503):
    checks = client_brain_http_503.get("/health/ready").json()["checks"]
    assert checks["brain"] == "http_503"


def test_ready_brain_not_configured_returns_503(client_brain_not_configured):
    r = client_brain_not_configured.get("/health/ready")
    assert r.status_code == 503


def test_ready_brain_not_configured_check(client_brain_not_configured):
    checks = client_brain_not_configured.get("/health/ready").json()["checks"]
    assert checks["brain"] == "not_configured"


# ── provider missing ───────────────────────────────────────────────────────────

def test_ready_stt_missing_returns_503(client_stt_missing):
    r = client_stt_missing.get("/health/ready")
    assert r.status_code == 503


def test_ready_stt_missing_check(client_stt_missing):
    checks = client_stt_missing.get("/health/ready").json()["checks"]
    assert checks["stt"] == "not_configured"


def test_ready_tts_missing_returns_503(client_tts_missing):
    r = client_tts_missing.get("/health/ready")
    assert r.status_code == 503


def test_ready_tts_missing_check(client_tts_missing):
    checks = client_tts_missing.get("/health/ready").json()["checks"]
    assert checks["tts"] == "not_configured"


# ── misc ───────────────────────────────────────────────────────────────────────

def test_ready_endpoint_is_get_only(client_all_ok):
    r = client_all_ok.post("/health/ready")
    assert r.status_code == 405
