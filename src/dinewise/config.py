"""Configuration loading and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and attach the repository root."""
    cfg_path = Path(path) if path is not None else ROOT / "config" / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg: dict[str, Any] = yaml.safe_load(handle) or {}
    cfg["_root"] = str(ROOT)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def data_path(cfg: dict[str, Any], key: str, *parts: str) -> Path:
    """Return an absolute path under one of cfg['paths']."""
    base = cfg["paths"][key]
    return ROOT / base / Path(*parts)


def resolve(value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("raw_dir", "processed_dir", "docs_dir", "index_dir", "audit_dir"):
        data_path(cfg, key).mkdir(parents=True, exist_ok=True)
