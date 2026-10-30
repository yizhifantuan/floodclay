from __future__ import annotations

import json
from collections.abc import Sequence
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
    train_events: Sequence[str],
    val_events: Sequence[str],
    test_events: Sequence[str],
) -> dict[str, list[SampleRecord]]:
    """严格按配置的事件列表分组，不随机划分或按比例分配。"""
    groups = {"train": train_events, "val": val_events, "test": test_events}
    assigned: set[str] = set()
    for name, events in groups.items():
        if isinstance(events, str) or not events or len(set(events)) != len(events):
            raise ValueError(f"{name}_events 必须为非空且不重复的事件列表")
        overlap = assigned.intersection(events)
        if overlap:
            raise ValueError(f"训练、验证和测试事件不能重叠：{sorted(overlap)}")
        assigned.update(events)
    groups = {name: set(events) for name, events in groups.items()}
    unknown = {record.event for record in records} - assigned
    if unknown:
        raise ValueError(f"数据包含未配置划分的事件：{sorted(unknown)}")
    splits = {
        name: [record for record in records if record.event in events]
        for name, events in groups.items()
    }
    for name, items in splits.items():
        if not items:
            raise ValueError(f"{name} 子集没有样本，请检查事件列表和数据目录")
    return splits

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
