"""训练检查点保存及加载；仅允许显式关闭的 AlphaEarth 分支被跳过。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    *,
    split_manifest: dict[str, object],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
            "split_manifest": split_manifest,
            **({"scheduler": scheduler.state_dict()} if scheduler is not None else {}),
        },
        path,
    )


# 加载权重
def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = read_checkpoint(path, map_location)
    state = payload["model"]
    if hasattr(model, "alphaearth_fusion") and model.alphaearth_fusion is None:
        # 显式关闭 AlphaEarth 后可做推理消融；其余参数仍严格检查。
        state = {name: value for name, value in state.items() if not name.startswith("alphaearth_fusion.")}
    model.load_state_dict(state, strict=True)
    return payload


def read_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """先读取配置和划分，完成校验后再构造模型；避免重复加载大权重。"""
    return torch.load(path, map_location=map_location, weights_only=False)
