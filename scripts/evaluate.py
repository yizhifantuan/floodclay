from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config, load_evaluation_config  # noqa: E402
from floodclay.data import FloodPlanetDataset, scan_floodplanet, restore_checkpoint_splits  # noqa: E402
from floodclay.engine import load_checkpoint, read_checkpoint, move_to_device  # noqa: E402
from floodclay.engine.experiments import latest_best_checkpoint  # noqa: E402
from floodclay.metrics import BinarySegmentationMetrics  # noqa: E402
from floodclay.models import build_model  # noqa: E402

# 定义三模态的三种情况
SCENARIOS = {
    "complete": (1, 1, 1),
    "missing_s1": (0, 1, 1),
    "missing_s2": (1, 0, 1),
    "missing_ps": (1, 1, 0),
    "only_s1": (1, 0, 0),
    "only_s2": (0, 1, 0),
    "only_ps": (0, 0, 1),
}

# 不进行反向传播
@torch.no_grad()
def evaluate_loader(
    # 已经加载训练权重的模型
    model: torch.nn.Module,
    # 待评估数据的数据加载器
    loader: DataLoader,
    device: torch.device,
    # 指定模态组合，或使用实际可用情况
    pattern: tuple[int, int, int] | None,
    # 是否启用混合精度
    amp: bool,
) -> dict[str, float]:
    # 创建指标计算器
    metrics = BinarySegmentationMetrics()
    model.eval()
    # 同一评估场景只创建一次模态掩码，再按每批的样本数广播。
    scenario = torch.tensor([pattern], device=device, dtype=torch.bool) if pattern is not None else None
    # 按照批次进行获取
    for batch in loader:
        batch = move_to_device(batch, device)
        # 默认使用实际模态状态
        selected = None
        # 判断是否指定了模态组合
        if scenario is not None:
            # 扩展批次，-1为不变（1，3）->（4，3）
            selected = scenario.expand(batch["availability"].shape[0], -1)
        # 为下面的模型预测设置混合精度环境
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            # 执行学生分支预测
            output = model.predict(batch, availability=selected)
        # 累计当前批次的指标统计
        metrics.update(output["logits"], batch["target"], batch["valid"])
    # 返回整个数据集的指标
    return metrics.compute()


def main(checkpoint: str | Path | None = None) -> None:
    # 运行设置：None 默认选择输出根目录下最新一个已有 best.pt 的实验。
    disable_alphaearth = False  # True 表示在评估时关闭辅助分支，做消融对照。
    split_manifest = None  # 旧检查点可在这里指定同次训练的 split_manifest.json。
    output = None
    if checkpoint is None:
        default_config = load_config(PROJECT / "configs" / "default.json")
        checkpoint = latest_best_checkpoint(default_config["training"]["output_dir"])
    checkpoint = Path(checkpoint)
    payload = read_checkpoint(checkpoint)
    config = load_evaluation_config(payload)
    if disable_alphaearth:
        config.setdefault("alphaearth", {})["enabled"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 扫描 FloodPlanet 数据集所在的文件夹，找到数据集中的样本，并将样本信息保存到 records 变量中
    records = scan_floodplanet(config["data"]["root"])
    # 只恢复训练时固定的样本；新增文件或当前目录内容不参与重新划分。
    splits = restore_checkpoint_splits(payload, records, checkpoint, split_manifest)
    # 定义数据集所要用到的公共参数
    common: dict[str, Any] = {
        "sensor_config": config["sensors"],
        "label_size": config["data"]["label_size"],
        "augment": False,
        "use_geo_metadata": config["data"]["use_geo_metadata"],
        "alphaearth_config": config.get("alphaearth"),
    }
    #complete_set完整数据集；natural_set是缺失数据集
    complete_records = [record for record in splits["test"] if record.is_complete]
    # complete_records 已筛选完整模态，数据集无需重复筛选。
    complete_set = FloodPlanetDataset(complete_records, **common) if complete_records else None
    natural_set = FloodPlanetDataset(splits["test"], require_complete=False, **common)
    # 定义数据加载器所用的公共参数
    loader_args = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
        "shuffle": False,
    }
    # 两种数据集的数据加载器
    complete_loader = DataLoader(complete_set, **loader_args) if complete_set is not None else None
    natural_loader = DataLoader(natural_set, **loader_args)
    # 定义模型
    model = build_model(config).to(device)
    # 从已经保存的模型检查点（checkpoint）文件中加载模型参数，并将加载得到的信息保存到 payload 变量中
    load_checkpoint(checkpoint, model, device, payload=payload)
    # 获取配置文件里的amp（混合精度训练）
    amp = config["training"]["amp"]
    # 遍历 SCENARIOS 中定义的所有模态组合，让模型在每一种模态组合下进行测试，并把测试结果保存到 results 字典中
    results = {
        # evaluate_loader 是评估函数
        name: (evaluate_loader(model, complete_loader, device, pattern, amp)
               if complete_loader is not None else None)
        for name, pattern in SCENARIOS.items()
    }
    # 使用真实模态缺失的测试集评估模型，并将评估结果保存到 results["natural_missingness"] 中。
    results["natural_missingness"] = evaluate_loader(model, natural_loader, device, None, amp)
    # 把模型的训练轮数、测试集样本数量以及测试结果整理到一个字典 report 中，方便后续打印或者保存成 JSON 文件
    report = {
        "checkpoint_epoch": payload["epoch"],
        "complete_test_samples": len(complete_records),
        "natural_test_samples": len(natural_set),
        "split_source": "checkpoint" if payload.get("split_manifest") is not None else "training_manifest",
        "test_events": sorted({record.event for record in splits["test"]}),
        "metrics": results,
    }
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_json)
    # 未设置 output 时，把测试结果保存到模型检查点所在的文件夹。
    output = Path(output) if output is not None else checkpoint.with_name("evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_json, encoding="utf-8")
    print(f"Evaluation saved to {output}")


if __name__ == "__main__":
    main()
