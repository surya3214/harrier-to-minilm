"""Config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(cfg: dict[str, Any], key: str, root: Path | None = None) -> Path:
    root = root or project_root()
    rel = cfg["paths"][key]
    p = Path(rel)
    return p if p.is_absolute() else root / p
