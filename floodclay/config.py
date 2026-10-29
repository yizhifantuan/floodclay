from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


# __file__获取当前文件的根目录
# .parents[1]：获取上上级目录 D:\Graduate\Flood monitoring\Flood monitoring\floodclay\config.py
PROJECT = Path(__file__).parents[1]
# 拼接路劲
DEFAULT_CONFIG = PROJECT / "configs" / "default.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    # 读取 configs/default.json，返回可直接使用的配置字典。
    with Path(path).open(encoding="utf-8") as handle:
        # 将 JSON 内容转换为 Python 字典。
        config = json.load(handle)
    return _prepare_config(config)


def load_evaluation_config(payload: dict[str, Any]) -> dict[str, Any]:
    """评估沿用检查点的训练配置，避免 default.json 后续修改影响模型和数据。"""
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("检查点缺少训练配置，无法恢复评估设置")
    return _prepare_config(deepcopy(config))

# 检查和整理配置
def _prepare_config(config: dict[str, Any]) -> dict[str, Any]:
    if config["data"]["label_size"] != config["model"]["output_size"]:
        raise ValueError("data.label_size must equal model.output_size for pixel-wise supervision")
    # 依次遍历需要处理的三个路径
    for section, key in (
        ("data", "root"),
        ("model", "clay_checkpoint"),
        ("training", "output_dir"),
    ):
        config[section][key] = str((PROJECT / config[section][key]).resolve())
    # 读取 AlphaEarth 配置
    auxiliary = config["alphaearth"]
    auxiliary["root"] = str((PROJECT / auxiliary["root"]).resolve())
    return config
