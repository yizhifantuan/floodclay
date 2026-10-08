from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]


def load_config(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    clay_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """读取配置；所有相对数据、权重和输出路径都以项目根目录为基准。"""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)

    if data_root is not None:
        config["data"]["root"] = str(data_root)
    if clay_checkpoint is not None:
        config["model"]["clay_checkpoint"] = str(clay_checkpoint)

    for section, key in (
        ("data", "root"),
        ("model", "clay_checkpoint"),
        ("training", "output_dir"),
    ):
        config[section][key] = str((PROJECT / config[section][key]).resolve())
    return config
