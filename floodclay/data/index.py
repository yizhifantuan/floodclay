from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# 定义名为 SampleRecord 的类
# @dataclass：这是一个主要用来保存数据的类，请自动帮我生成初始化函数。
@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    event: str
    ps: str
    s1: str | None
    s2: str | None
    label: str

    # 检验模态是否存在
    @property
    # 定义一个availability的属性
    def availability(self) -> tuple[bool, bool, bool]:
        """Availability in the fixed order (S1, S2, PS)."""
        # 返回tuple[bool, bool, bool]
        return self.s1 is not None, self.s2 is not None, True

    # 判断样本是否完整
    @property
    # 定义is_complete属性，判断三个模态是否都存在
    def is_complete(self) -> bool:
        return all(self.availability)

    # 将样本记录转换为字典
    def to_dict(self) -> dict[str, object]:
        # 把当前对象转换为字典
        result = asdict(self)
        # 字典里记录模态的可用性
        result["availability"] = list(self.availability)
        # 字典里记录三模态是否完整
        result["complete"] = self.is_complete
        return result

# 扫描一个文件夹中的 .tif
def _files_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")
    # 寻找当前文件夹里所有以 .tif 结尾的文件；path.stem获取不带扩展的文件名
    return {path.stem: path for path in directory.glob("*.tif")}

# 扫描整个 FloodPlanet 数据集
def scan_floodplanet(root: str | Path) -> list[SampleRecord]:
    # 把 root 转换成 Path 对象。
    root = Path(root).expanduser()
    # 四个数据的路径；分别建立字典
    ps = _files_by_stem(root / "PS" / "images")
    s1 = _files_by_stem(root / "S1" / "images")
    s2 = _files_by_stem(root / "S2" / "images")
    labels = _files_by_stem(root / "S2" / "masks")
    # 检查标签是否缺失
    missing_labels = sorted(set(ps) - set(labels))
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(f"PS samples without labels ({len(missing_labels)}): {preview}")
    # 建立样本记录，创建一个空列表，用来保存全部的样本；列表中存放SampleRecord对象
    records: list[SampleRecord] = []
    # 循环处理每一个ps样本
    for sample_id in sorted(ps):
        records.append(
            SampleRecord(
                sample_id=sample_id,
                event=sample_id.split("_", maxsplit=1)[0],
                ps=str(ps[sample_id]),
                s1=str(s1[sample_id]) if sample_id in s1 else None,
                s2=str(s2[sample_id]) if sample_id in s2 else None,
                label=str(labels[sample_id]),
            )
        )
    return records

# 按洪水事件划分数据集
def split_by_event(
    records: Iterable[SampleRecord],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[SampleRecord]]:

    # 把输入转换成列表
    records = list(records)
    # 取出所有不重复的事件名称
    events = sorted({record.event for record in records})
    if len(events) < 3:
        raise ValueError("At least three event groups are required for train/val/test")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to < 1")
    # 创建一个独立随机数生成器，种子为 42
    rng = random.Random(seed)
    # 随机打乱事件顺序
    rng.shuffle(events)
    # 计算测试集应该包含多少个事件
    n_test = max(1, round(len(events) * test_fraction))
    # 计算验证集应该包含多少个事件
    n_val = max(1, round(len(events) * val_fraction))
    # 检查测试集和训练集是否占用了全部的数据
    if n_test + n_val >= len(events):
        n_test = n_val = 1
    # 按数量进行分配
    test_events = set(events[:n_test])
    val_events = set(events[n_test : n_test + n_val])
    train_events = set(events[n_test + n_val :])

    return {
        "train": [r for r in records if r.event in train_events],
        "val": [r for r in records if r.event in val_events],
        "test": [r for r in records if r.event in test_events],
    }

# 统计数据集信息
def summarize(records: Iterable[SampleRecord]) -> dict[str, object]:
    # 输入一组样本，输出一个统计字典。
    records = list(records)
    return {
        "samples": len(records),
        "complete": sum(r.is_complete for r in records),
        "s1_missing": sum(r.s1 is None for r in records),
        "s2_missing": sum(r.s2 is None for r in records),
        "events": sorted({r.event for r in records}),
    }

