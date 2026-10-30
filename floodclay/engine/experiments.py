"""创建独立实验目录，并定位默认使用的实验权重。"""

from datetime import datetime
from pathlib import Path


def create_experiment_dir(output_root: str | Path) -> Path:
    """在整次训练开始时调用一次，所有轮次及后续测试复用返回的目录。"""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    suffix = 0
    while True:
        path = root / (name if suffix == 0 else f"{name}_{suffix:06d}")
        try:
            # 独占创建，时间相同或并发启动也不会复用已有实验目录。
            path.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return path


def latest_best_checkpoint(output_root: str | Path) -> Path:
    root = Path(output_root)
    # 默认读取 runs/result/run_*，同时兼容此前 runs 下的权重。
    roots = [root, root / "default"]
    if root.name == "result":
        roots.extend((root.parent, root.parent / "default"))
    else:
        roots.append(root / "result")
    candidates = [
        path for parent in roots
        for path in parent.glob("run_*/best.pt") if path.is_file()
    ]
    if candidates:
        return max(candidates, key=lambda path: path.parent.name)
    # 兼容此前 output_dir/best.pt 以及 runs/default/best.pt 的权重。
    for legacy in (parent / "best.pt" for parent in roots):
        if legacy.is_file():
            return legacy
    raise FileNotFoundError(f"未找到实验的 best.pt，请先训练或指定检查点路径：{root}")
