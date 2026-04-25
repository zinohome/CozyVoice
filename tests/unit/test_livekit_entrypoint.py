"""LiveKit entrypoint unit tests (mock LiveKit primitives)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.livekit_entrypoint import _parse_participant_metadata


def test_parse_valid_metadata() -> None:
    raw = json.dumps({"brain_jwt": "tok", "session_id": "s1", "personality_id": "p1"})
    result = _parse_participant_metadata(raw)
    assert result["brain_jwt"] == "tok"
    assert result["session_id"] == "s1"


def test_parse_empty_metadata() -> None:
    assert _parse_participant_metadata(None) == {}
    assert _parse_participant_metadata("") == {}


def test_parse_invalid_json() -> None:
    result = _parse_participant_metadata("not-json{{{")
    assert result == {}


def test_parse_non_dict_json() -> None:
    result = _parse_participant_metadata('"just a string"')
    assert result == {}
