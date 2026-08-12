"""Core algorithm layer and package-owned system configuration."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the local package configuration, or an explicit YAML override."""
    config_path = (
        Path(path)
        if path is not None
        else Path(str(files(__name__) / "system_config.yaml"))
    )
    if not config_path.is_file():
        example_path = config_path.with_name("system_config.example.yaml")
        raise FileNotFoundError(
            f"system config not found: {config_path}. "
            f"Copy {example_path} to {config_path} and adjust it for this machine."
        )
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"system config must be a mapping: {config_path}")
    return config
