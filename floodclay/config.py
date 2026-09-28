from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment configuration.

    JSON is used deliberately so the project has no YAML dependency. Relative
    model/output paths are resolved by the command that consumes the config;
    the dataset root is left untouched because it may live on another drive.
    """

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a copy with dotted-key overrides (mainly useful in tests)."""

    result = deepcopy(config)
    for dotted_key, value in overrides.items():
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result

