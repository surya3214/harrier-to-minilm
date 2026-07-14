"""Dataset registry stubs for later language/task expansion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harrier_distill.config import load_yaml, project_root


def load_language_registry(cfg: dict[str, Any]) -> dict[str, Any]:
    path = cfg.get("languages", {}).get("registry", "configs/languages/registry.yaml")
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    return load_yaml(p)


def load_task_registry(cfg: dict[str, Any]) -> dict[str, Any]:
    path = cfg.get("tasks", {}).get("registry", "configs/tasks/registry.yaml")
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    return load_yaml(p)


def active_languages(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("languages", {}).get("active", ["en", "ko"]))


def active_tasks(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("tasks", {}).get("active", ["sts", "retrieval", "bitext"]))


def enabled_future_tasks(cfg: dict[str, Any]) -> list[str]:
    """Tasks marked for later expansion (clustering/classification/reranking)."""
    return list(cfg.get("tasks", {}).get("later", []))
