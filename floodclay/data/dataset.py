from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import rasterio
import torch
from rasterio.warp import transform as transform_coordinates
from torch.utils.data import Dataset

from .index import SampleRecord
from .alphaearth import AlphaEarthReader

# 定义三个模态
MODALITIES = ("s1", "s2", "ps")

# 按原始尺寸读取遥感影像；特征尺寸在编码器之后统一。
# 这个函数执行完毕，返回值的类型应该是 `torch.Tensor`
def _read_image(path: str) -> torch.Tensor:
    with rasterio.open(path) as src:
        # 读取遥感数据
        array = src.read().astype("float32", copy=False)
    # 把 numpy数组（只能在CPU计算）转换为 pytorch张量（可以迁移到GPU）
    return torch.from_numpy(array)

# 读取洪涝标签
# 返回一个元组，元组里面固定有2个元素
def _read_label(path: str, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    with rasterio.open(path) as src:
        if (src.height, src.width) != (size, size):
            raise ValueError(
                f"Native label shape {(src.height, src.width)} does not match model output "
                f"{(size, size)}: {path}. Labels were not resized."
            )
        array = src.read(1)
    # FloodPlanet labels: 1 = background, 2 = inundation, 0 = no-data.
    # target 洪水和非洪水标签
    # valid 哪些像素参与loss计算
    # 生成有效像素的 mask
    valid = torch.from_numpy(array > 0)
    # 只有像素为 2 的地方是 true
    # unsqueeze(0) 增加一个维度
    target = torch.from_numpy((array == 2).astype("float32", copy=False)).unsqueeze(0)
    return target, valid.unsqueeze(0)

# 对每个波段进行归一化操作
def _normalise(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    # 检查通道数是否一致
    if image.shape[0] != len(mean):
        raise ValueError(f"Expected {len(mean)} bands, got {image.shape[0]}")
    # 进行广播
    # torch.as_tensor() 把数组转换为torch张量
    # (mean, dtype=image.dtype)数据类型保持一致
    # view(-1, 1, 1) 改变数据维度
    means = torch.as_tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    stds = torch.as_tensor(std, dtype=image.dtype).view(-1, 1, 1)
    # Replace NaN/Inf with the band mean, which maps to zero after normalisation.
    # torch.isfinite(image)判断每个像素是不是正常有限数
    # 正常值保留；非有限值替换成波段均值，自动广播到影像尺寸。
    image = torch.where(torch.isfinite(image), image, means)
    # 返回标准化后的结果
    return (image - means) / stds.clamp_min(1e-6)

# 计算影像中心位置
def _centroid(path: str) -> tuple[float, float]:
    with rasterio.open(path) as src:
        bounds = src.bounds
        lon = (bounds.left + bounds.right) / 2.0
        lat = (bounds.bottom + bounds.top) / 2.0
        if src.crs is None:
            raise ValueError(f"Geographic metadata requires a CRS: {path}")
        x, y = transform_coordinates(src.crs, "EPSG:4326", [lon], [lat])
        lon, lat = x[0], y[0]
    return lat, lon

# 把经纬度坐标改成Clay的位置编码
def _encode_latlon(lat: float, lon: float) -> torch.Tensor:
    """Four-value cyclic location encoding expected by Clay."""
    # 把度转换为弧度
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    return torch.tensor(
        [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
        dtype=torch.float32,
    )


def _encode_time(timestamp: str) -> torch.Tensor:
    """STAC UTC 时间 → Clay 的 [sin(周), cos(周), sin(小时), cos(小时)]。"""
    acquired = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    week = acquired.isocalendar().week * (2.0 * np.pi / 52)
    hour = acquired.hour * (2.0 * np.pi / 24)
    return torch.tensor([np.sin(week), np.cos(week), np.sin(hour), np.cos(hour)], dtype=torch.float32)

# 计算两张影像的空间重叠率
# 用于判断S1、S2和 PS数据是否真的覆盖同一个位置
# reference 基准框；candidate 候选框
def _coverage(reference: rasterio.coords.BoundingBox, candidate: rasterio.coords.BoundingBox) -> float:
    # 计算两个矩形的相交区域
    width = max(0.0, min(reference.right, candidate.right) - max(reference.left, candidate.left))
    height = max(0.0, min(reference.top, candidate.top) - max(reference.bottom, candidate.bottom))
    # 相交的面积
    intersection = width * height
    # 计算reference的面积
    ref_area = (reference.right - reference.left) * (reference.top - reference.bottom)
    # 计算candidate的面积
    candidate_area = (candidate.right - candidate.left) * (candidate.top - candidate.bottom)
    # 返回空间重叠率
    return intersection / max(min(ref_area, candidate_area), 1e-12)

# 检查数据是否地理对齐了，默认要求98%的空间覆盖
def validate_geographic_alignment(
    records: Sequence[SampleRecord], minimum_coverage: float = 0.98
) -> None:
    """Fail early when nominally paired rasters do not cover the same area."""
    # 逐个样本检查
    for record in records:
        # 使用实际存在的参考影像，不要求 PS 存在。
        reference_path = record.reference_path
        with rasterio.open(reference_path) as src:
            reference_bounds, reference_crs = src.bounds, src.crs
        # 依次进行比较
        candidates = [record.s1, record.s2, record.ps, record.label]
        # 逐个检查
        for path in candidates:
            if path is None:
                continue
            if path == reference_path:
                # 参考影像的元数据已读取，无需再次打开同一个文件。
                bounds, crs = reference_bounds, reference_crs
            else:
                with rasterio.open(path) as src:
                    bounds, crs = src.bounds, src.crs
            # 判断当前文件是不是标签
            is_label = path == record.label
            # 如果当前文件不是标签文件，并且已经有一个参考坐标系，同时当前文件的坐标系和参考坐标系不一致
            if not is_label and reference_crs is not None and crs != reference_crs:
                raise ValueError(f"CRS mismatch for {record.sample_id}: {path}")
            coverage = _coverage(reference_bounds, bounds)
            if coverage < minimum_coverage:
                raise ValueError(
                    f"Geographic coverage mismatch for {record.sample_id}: "
                    f"{coverage:.3f} < {minimum_coverage:.3f} ({path})"
                )

# 定义 PyTorch 数据集
# 数据集中的每个样本都会按照dict[str, Any]返回
class FloodPlanetDataset(Dataset[dict[str, Any]]):
    # 数据集初始化 __init__()；定义初始化需要传入的参数
    def __init__(
        self,
        # 接收 index.py 生成的样本记录
        records: Sequence[SampleRecord],
        # 接收传感器配置
        sensor_config: dict[str, Any],
        label_size: int = 1024,
        #  是否执行数据增强
        augment: bool = False,
        # 是否只保留三模态完整样本
        require_complete: bool = False,
        # 是否提供地理位置编码
        use_geo_metadata: bool = True,
        validate_geography: bool = True,
        alphaearth_config: dict[str, Any] | None = None,
    ) -> None:# 函数没有返回值
        # 根据模态完整性筛选样本，并把结果保存到当前对象的 self.records 中
        # is_complete 三个模态是否都存在
        self.records = [record for record in records if not require_complete or record.is_complete]
        self.sensor_config = sensor_config
        self.label_size = label_size
        self.augment = augment
        self.use_geo_metadata = use_geo_metadata
        # 是否启用alphaearth
        self.alphaearth_reader = (
            AlphaEarthReader(alphaearth_config)
            if alphaearth_config and alphaearth_config.get("enabled", False) else None
        )
        # 缺失模态没有原图，使用该模态首个已有样本的尺寸创建零占位。
        # 如果整个子集都缺少该模态，使用可用参考影像尺寸；availability 屏蔽占位。
        self.missing_shapes = {}
        for modality in MODALITIES:
            # 只有实际缺失的模态才需要读取占位尺寸。
            if all(getattr(record, modality) is not None for record in self.records):
                continue
            # 找到对应模态的影像路径
            # next() 获取生成器提供的第一个值
            path = next(
                (getattr(r, modality) for r in self.records if getattr(r, modality) is not None),
                self.records[0].reference_path,
            )
            with rasterio.open(path) as src:
                self.missing_shapes[modality] = (src.height, src.width)
        if validate_geography:
            validate_geographic_alignment(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        # 创建空字典，用来保存读取后的影像张量
        images: dict[str, torch.Tensor] = {}
        # 把 index.py 中的模态可用状态转换成 PyTorch 张量
        availability = torch.tensor(record.availability, dtype=torch.bool)

        # MODALITIES定义了三个模态
        for modality in MODALITIES:
            # 从配置文件中取得当前模态的配置
            spec = self.sensor_config[modality]
            path = getattr(record, modality)
            # 如果模态缺失，就创建一个全零张量作为占位符，使用mean的均值数量创建通道数
            if path is None:
                images[modality] = torch.zeros(
                    len(spec["mean"]), *self.missing_shapes[modality], dtype=torch.float32
                )
            else:
                # 读取图像
                image = _read_image(path)
                # 对数据进行标准化
                images[modality] = _normalise(image, spec["mean"], spec["std"])

        if record.label is not None:
            target, valid = _read_label(record.label, self.label_size)
        else:
            # 无标签预测仍提供固定形状占位；全 False 防止被当作监督真值。
            target = torch.zeros(1, self.label_size, self.label_size, dtype=torch.float32)
            valid = torch.zeros_like(target, dtype=torch.bool)
        auxiliary = {}
        if self.alphaearth_reader is not None:
            auxiliary["alphaearth"], auxiliary["alphaearth_valid"] = self.alphaearth_reader.read(record)

        # 对影像、辅助特征和标签同步进行随机水平、垂直翻转
        if self.augment:
            for dimension in (-1, -2):
                if random.random() < 0.5:
                    images = {key: value.flip(dimension) for key, value in images.items()}
                    auxiliary = {key: value.flip(dimension) for key, value in auxiliary.items()}
                    target, valid = target.flip(dimension), valid.flip(dimension)
        # 根据开关提供地理位置编码
        if self.use_geo_metadata:
            # 计算图像的中心位置
            lat, lon = _centroid(record.reference_path)
            # 把经纬度转换成位置编码
            latlon = _encode_latlon(lat, lon)
        else:
            # 关闭地理元数据时使用零占位
            latlon = torch.zeros(4, dtype=torch.float32)

        time = _encode_time(record.timestamp)

        return {
            "id": record.sample_id,
            "images": images,
            "availability": availability,
            "target": target,
            "valid": valid,
            "time": time,
            "latlon": latlon,
            **auxiliary,
        }
