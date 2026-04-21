"""配置加载（yaml + env 插值）。"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    text = path.read_text()

    def _sub(m: "re.Match") -> str:
        key, default = m.group(1), m.group(2) or ""
        return os.environ.get(key, default)

    text = re.sub(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}", _sub, text)
    return yaml.safe_load(text) or {}
