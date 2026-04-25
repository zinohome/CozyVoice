"""Tests for _build_stt and _build_tts factory functions in main.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cozyvoice.providers.stt.mock import MockSTT
from cozyvoice.providers.tts.mock import MockTTS
from cozyvoice.providers.tts.fallback import FallbackTTS


def _import_builders():
    from cozyvoice.main import _build_stt, _build_tts
    return _build_stt, _build_tts


def test_build_stt_mock_is_default():
    _build_stt, _ = _import_builders()
    stt = _build_stt({})
    assert isinstance(stt, MockSTT)


def test_build_stt_mock_when_provider_is_mock():
    _build_stt, _ = _import_builders()
    stt = _build_stt({"stt": {"provider": "mock"}})
    assert isinstance(stt, MockSTT)


def test_build_stt_openai_whisper():
    _build_stt, _ = _import_builders()
    with patch("cozyvoice.providers.stt.openai_whisper.AsyncOpenAI"):
        stt = _build_stt({
            "stt": {
                "provider": "openai_whisper",
                "openai": {
                    "api_key": "test-key",
                    "model": "whisper-1",
                }
            }
        })
    from cozyvoice.providers.stt.openai_whisper import OpenAIWhisperSTT
    assert isinstance(stt, OpenAIWhisperSTT)


def test_build_tts_no_chain_returns_mock():
    _, _build_tts = _import_builders()
    tts = _build_tts({})
    assert isinstance(tts, MockTTS)


def test_build_tts_provider_mock_returns_mock():
    _, _build_tts = _import_builders()
    tts = _build_tts({"tts": {"provider": "mock"}})
    assert isinstance(tts, MockTTS)


def test_build_tts_with_edge_in_chain():
    _, _build_tts = _import_builders()
    cfg = {
        "tts": {
            "fallback_chain": [
                {"provider": "edge", "voice_id": "zh-CN-XiaoxiaoNeural", "timeout_ms": 3000}
            ]
        }
    }
    tts = _build_tts(cfg)
    assert isinstance(tts, FallbackTTS)
    assert len(tts._providers) == 1


def test_build_tts_skips_tencent_without_creds():
    _, _build_tts = _import_builders()
    cfg = {
        "tts": {
            "fallback_chain": [
                {"provider": "tencent", "tencent": {"secret_id": "", "secret_key": ""}}
            ]
        }
    }
    # Tencent entry with empty creds is skipped -> no valid providers -> MockTTS
    tts = _build_tts(cfg)
    assert isinstance(tts, MockTTS)


def test_build_tts_chain_empty_providers_returns_mock():
    _, _build_tts = _import_builders()
    cfg = {
        "tts": {
            "fallback_chain": [
                {"provider": "openai", "openai": {"api_key": ""}},
                {"provider": "tencent", "tencent": {"secret_id": "", "secret_key": ""}},
            ]
        }
    }
    tts = _build_tts(cfg)
    assert isinstance(tts, MockTTS)


def test_build_tts_with_openai_in_chain():
    _, _build_tts = _import_builders()
    with patch("cozyvoice.providers.tts.openai_tts.AsyncOpenAI"):
        cfg = {
            "tts": {
                "fallback_chain": [
                    {"provider": "openai", "openai": {"api_key": "sk-test"}, "timeout_ms": 5000}
                ]
            }
        }
        tts = _build_tts(cfg)
    assert isinstance(tts, FallbackTTS)
    assert len(tts._providers) == 1
