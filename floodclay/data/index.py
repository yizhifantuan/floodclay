from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# 定义名为 SampleRecord 的类
# @dataclass：这是一个主要用来保存数据的类，请自动帮我生成初始化函数。
@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    event: str
    ps: str | None
    s1: str | None
    s2: str | None
    label: str | None
    timestamp: str

    # 检验模态是否存在
    # @property把一个方法变成类似属性的访问方式
    @property
    # 定义一个availability的属性
    def availability(self) -> tuple[bool, bool, bool]:
        # 返回tuple[bool, bool, bool]
        return self.s1 is not None, self.s2 is not None, self.ps is not None

    @property
    def reference_modality(self) -> str:
        """固定优先级选择实际存在的影像，不要求任何一路必须存在。"""
        # getattr() 用于获取对象的属性;next()第一个匹配了就不继续往下
        return next(name for name in ("ps","s1","s2") if getattr(self, name) is not None)

    @property
    def reference_path(self) -> str:
        return getattr(self, self.reference_modality)

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
        result["reference_modality"] = self.reference_modality
        return result

# 扫描一个文件夹中的 .tif
def _files_by_stem(directory: Path) -> dict[str, Path]:
    # 寻找当前文件夹里所有以 .tif 结尾的文件；path.stem获取不带扩展的文件名
    return {path.stem: path for path in directory.glob("*.tif")}

# 扫描整个 FloodPlanet 数据集
def scan_floodplanet(root: str | Path, require_labels: bool = True) -> list[SampleRecord]:
    """按事件目录配对影像，读取参考影像的 STAC 时间（PS → S1 → S2）。"""
    root = Path(root)
    records: list[SampleRecord] = []
    seen_ids: set[str] = set()
    for event_dir in sorted((root / "FloodPlanet").iterdir()):
        if not event_dir.is_dir():
            continue
        files = {name: _files_by_stem(event_dir / name) for name in ("PS", "S1", "S2", "labels")}
        sample_ids = files["PS"].keys() | files["S1"].keys() | files["S2"].keys()
        if require_labels and (missing := sample_ids - files["labels"].keys()):
            raise ValueError(f"Image samples without labels in {event_dir.name}: {sorted(missing)}")
        for sample_id in sorted(sample_ids):
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id across events: {sample_id}")
            seen_ids.add(sample_id)
            reference = next(name for name in ("PS", "S1", "S2") if sample_id in files[name])
            metadata_path = root / "stac_catalog" / event_dir.name / reference / sample_id / f"{sample_id}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            timestamp = datetime.fromisoformat(metadata["properties"]["datetime"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError(f"STAC datetime requires a timezone: {metadata_path}")
            paths = {name.lower() if name != "labels" else "label":
                     str(items[sample_id]) if sample_id in items else None
                     for name, items in files.items()}
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    event=event_dir.name,
                    timestamp=timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    **paths,
                )
            )
    if not records:
        raise ValueError(f"No FloodPlanet image samples found in {root / 'FloodPlanet'}")
    return records

# 按洪水事件划分数据集
def split_by_event(
    records: list[SampleRecord],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[SampleRecord]]:
    # 取出所有不重复的事件名称
    events = sorted({record.event for record in records})
    # 创建一个独立随机数生成器，种子为 42
    rng = random.Random(seed)
    # 随机打乱事件顺序
    rng.shuffle(events)
    # 计算测试集应该包含多少个事件
    n_test = max(1, round(len(events) * test_fraction))
    # 计算验证集应该包含多少个事件
    n_val = max(1, round(len(events) * val_fraction))
    # 按数量进行分配
    test_events = set(events[:n_test])
    val_events = set(events[n_test : n_test + n_val])
    train_events = set(events[n_test + n_val :])
    # 字典有三个键，每个键对应一组 SampleRecord 样本记录
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
        "ps_missing": sum(r.ps is None for r in records),
        "events": sorted({r.event for r in records}),
    }
