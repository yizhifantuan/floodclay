from pathlib import Path
import sys

# 保证直接运行本文件时也能找到 floodclay 包。
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.stdout.reconfigure(encoding="utf-8")

from floodclay.config import load_config
from floodclay.data import FloodPlanetDataset, scan_floodplanet

# 1. 读取配置并扫描数据文件。
config = load_config(PROJECT / "configs" / "default.json")
records = scan_floodplanet(config["data"]["root"])
print(f"找到 {len(records)} 个样本")

# 2. 根据 S1、S2 是否存在，找出缺失模态的样本编号。
only_s1_missing = [
    record.sample_id
    for record in records
    if record.s1 is None and record.s2 is not None
]
only_s2_missing = [
    record.sample_id
    for record in records
    if record.s1 is not None and record.s2 is None
]
both_missing = [
    record.sample_id
    for record in records
    if record.s1 is None and record.s2 is None
]

print(f"\n只有 S1 缺失（{len(only_s1_missing)} 个）：")
print(only_s1_missing)
print(f"\n只有 S2 缺失（{len(only_s2_missing)} 个）：")
print(only_s2_missing)
print(f"\nS1 和 S2 都缺失（{len(both_missing)} 个）：")
print(both_missing)

# 3. 影像保持原始尺寸，标签在这个测试中设置为 64×64。
dataset = FloodPlanetDataset(
    records,
    config["sensors"],
    label_size=64,
    augment=False,
    validate_geography=False,
)
target_id = "FLO_10_16"

index = next(
    (i for i, record in enumerate(records) if record.sample_id == target_id),
    None
)
# 4. 读取第一个样本并查看结果。
sample = dataset[index]
print(f"样本编号：{sample['id']}")
print(f"模态是否存在 [S1, S2, PS]：{sample['availability'].tolist()}")
print(f"S1 形状：{list(sample['images']['s1'].shape)}")
print(f"S2 形状：{list(sample['images']['s2'].shape)}")
print(f"PS 形状：{list(sample['images']['ps'].shape)}")
print(f"标签形状：{list(sample['target'].shape)}")
print(f"标签取值：{sample['target'].unique().tolist()}")
