"""Config loader (settings.py) unit tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from cozyvoice.config.settings import load_config


def test_load_config_env_substitution(tmp_path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("voice:\n  key: ${TEST_KEY_123:-default_val}\n")
    with patch.dict(os.environ, {"TEST_KEY_123": "real_val"}, clear=False):
        result = load_config(cfg_file)
    assert result["voice"]["key"] == "real_val"


def test_load_config_default_value(tmp_path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("voice:\n  key: ${NONEXISTENT_VAR_XYZ:-fallback}\n")
    result = load_config(cfg_file)
    assert result["voice"]["key"] == "fallback"


def test_load_config_no_default_empty_string(tmp_path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("voice:\n  key: ${NONEXISTENT_VAR_ABC}\n")
    result = load_config(cfg_file)
    # env var absent and no default: substituted to empty string, which YAML parses as null/None
    assert result["voice"]["key"] is None


def test_load_config_multiple_vars(tmp_path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("voice:\n  a: ${VAR_A:-alpha}\n  b: ${VAR_B:-beta}\n")
    with patch.dict(os.environ, {"VAR_A": "AAA"}, clear=False):
        result = load_config(cfg_file)
    assert result["voice"]["a"] == "AAA"
    assert result["voice"]["b"] == "beta"
