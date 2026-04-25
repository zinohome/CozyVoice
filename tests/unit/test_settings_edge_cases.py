"""Edge-case tests for cozyvoice.config.settings.load_config."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cozyvoice.config.settings import load_config


def test_load_nonexistent_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_load_empty_yaml(tmp_path):
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("")
    result = load_config(cfg_file)
    assert result == {}


def test_load_whitespace_only_yaml(tmp_path):
    cfg_file = tmp_path / "spaces.yaml"
    cfg_file.write_text("   \n\n  \n")
    result = load_config(cfg_file)
    assert result == {}


def test_nested_env_vars(tmp_path):
    cfg_file = tmp_path / "nested.yaml"
    cfg_file.write_text(
        "level1:\n"
        "  level2:\n"
        "    level3: ${DEEP_VAR:-deep_default}\n"
    )
    with patch.dict(os.environ, {"DEEP_VAR": "deep_value"}, clear=False):
        result = load_config(cfg_file)
    assert result["level1"]["level2"]["level3"] == "deep_value"


def test_nested_env_var_uses_default_when_absent(tmp_path):
    cfg_file = tmp_path / "nested_default.yaml"
    cfg_file.write_text(
        "outer:\n"
        "  inner: ${THIS_VAR_DOES_NOT_EXIST_XYZ:-nested_default}\n"
    )
    result = load_config(cfg_file)
    assert result["outer"]["inner"] == "nested_default"


def test_env_var_in_list(tmp_path):
    cfg_file = tmp_path / "list.yaml"
    cfg_file.write_text(
        "items:\n"
        "  - ${LIST_VAR_A:-item_a}\n"
        "  - ${LIST_VAR_B:-item_b}\n"
    )
    with patch.dict(os.environ, {"LIST_VAR_A": "actual_a"}, clear=False):
        result = load_config(cfg_file)
    assert result["items"][0] == "actual_a"
    assert result["items"][1] == "item_b"


def test_multiple_vars_on_same_line(tmp_path):
    cfg_file = tmp_path / "multi.yaml"
    cfg_file.write_text(
        'voice:\n  url: "http://${HOST:-localhost}:${PORT:-8000}"\n'
    )
    with patch.dict(os.environ, {"HOST": "myhost", "PORT": "9999"}, clear=False):
        result = load_config(cfg_file)
    assert result["voice"]["url"] == "http://myhost:9999"
