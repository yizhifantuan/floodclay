# 用来监测数据集数据数据集是否完整、不同遥感影像是否正确配对、
# 训练集和测试集如何划分，以及洪涝分割标签是否符合要求
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
# Path(__file__)讲文件路径转换为path对象
# .resolve()获得文件的绝对路径，并规范化路径
# .parents[1]表示向上寻找第二级父目录。
PROJECT = Path(__file__).resolve().parents[1]
# 把项目根目录加入 Python 的模块搜索路径
sys.path.insert(0, str(PROJECT))

from floodclay.config import load_config  # noqa: E402
from floodclay.data.index import scan_floodplanet, split_by_event, summarize  # noqa: E402

# 读取一张洪涝标签影像，检查它的形状、数据类型、像素值范围，以及标签是否符合预期
def mask_signature(path: Path) -> tuple[object, ...]:
    with rasterio.open(path) as src:
        array = src.read()
    # 获取影像中所有不同的像素值
    values = np.unique(array)
    # 检查标签是否合法
    # 检查影像是不是只有一个波段
    # values.tolist()把NumPy 数组转换成 Python 列表
    # issubset判断实际像素值是否全部属于允许的集合
    #
    valid_binary = array.shape[0] == 1 and set(values.tolist()).issubset({0, 1, 2})
    # 标签维度
    # 像素数据类型
    # 最小像素值
    # 最大像素值
    # 标签通过检查
    return tuple(array.shape), str(array.dtype), int(values.min()), int(values.max()), valid_binary


def main() -> None:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Audit FloodPlanet pairing and label integrity")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "default.json"))
    parser.add_argument("--output", default=str(PROJECT / "runs" / "dataset_audit.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["data"]["root"])
    records = scan_floodplanet(root)
    splits = split_by_event(
        records,
        config["data"]["val_fraction"],
        config["data"]["test_fraction"],
        config["data"]["split_seed"],
    )

    report: dict[str, object] = {
        "root": str(root),
        "all": summarize(records),
        "splits": {name: summarize(items) for name, items in splits.items()},
    }
    mask_report: dict[str, object] = {}
    for modality in ("S1", "S2"):
        directory = root / modality / "masks"
        signatures: Counter[tuple[object, ...]] = Counter()
        invalid: list[str] = []
        for path in sorted(directory.glob("*.tif")):
            signature = mask_signature(path)
            signatures[signature] += 1
            if not signature[-1]:
                invalid.append(path.name)
        mask_report[modality] = {
            "patterns": {str(key): value for key, value in signatures.items()},
            "invalid_files": invalid,
        }
    report["masks"] = mask_report

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAudit saved to {output}")


if __name__ == "__main__":
    main()

