"""恢复检查点绑定的数据划分，禁止在评估时重新随机划分。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import warnings

from .index import SampleRecord


def restore_checkpoint_splits(
    payload: dict[str, Any],
    records: Sequence[SampleRecord],
    checkpoint_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, list[SampleRecord]]:
    """用保存的 ID 恢复划分；允许数据根目录迁移，拒绝样本缺失和模态状态变化。

    新增文件不会加入既有划分。旧检查点只能使用训练时留下的清单，
    不根据当前目录内容或保存的随机种子重新生成测试集。
    """
    manifest = payload.get("split_manifest")
    if manifest is None:
        path = Path(manifest_path) if manifest_path else Path(checkpoint_path).with_name("split_manifest.json")
        if not path.is_file():
            raise ValueError("检查点缺少数据划分；需要同次训练的 split_manifest.json（--split-manifest）")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        warnings.warn("旧检查点未内嵌划分；使用训练清单：" + str(path), UserWarning, stacklevel=2)
        # 兼容旧版小样本流程检查的清单格式，保留其选定的测试样本。
        if "selected" in manifest:
            selected = manifest["selected"]
            manifest = {
                "train": selected["train"], "val": selected["val"],
                "test": selected.get("natural_test", selected["test"]),
            }
    elif manifest_path is not None:
        raise ValueError("检查点已内嵌数据划分，不能使用 --split-manifest 替换")

    current = {record.sample_id: record for record in records}
    if len(current) != len(records):
        raise ValueError("当前数据包含重复 sample_id")
    restored: dict[str, list[SampleRecord]] = {}
    seen_ids: set[str] = set()
    seen_events: set[str] = set()
    for name in ("train", "val", "test"):
        saved = manifest.get(name)
        if not isinstance(saved, list) or not saved:
            raise ValueError(f"划分清单缺少非空 {name} 列表")
        items = []
        for row in saved:
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"划分清单样本重复或跨集合泄漏：{sample_id}")
            seen_ids.add(sample_id)
            if sample_id not in current:
                raise ValueError(f"训练清单中的样本已缺失：{sample_id}")
            record = current[sample_id]
            saved_availability = row.get("availability")
            availability = (tuple(saved_availability) if saved_availability is not None else
                            tuple(row.get(name) is not None for name in ("s1", "s2", "ps")))
            if record.event != row["event"] or record.availability != availability:
                raise ValueError(f"样本事件或模态可用性与训练清单不一致：{sample_id}")
            if "timestamp" in row and record.timestamp != row["timestamp"]:
                raise ValueError(f"样本时间与训练清单不一致：{sample_id}")
            items.append(record)
        events = {record.event for record in items}
        if events & seen_events:
            raise ValueError(f"划分清单的洪水事件跨集合重叠：{sorted(events & seen_events)}")
        seen_events.update(events)
        restored[name] = items
    return restored
