from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from torch.utils.data import Dataset

from .index import SampleRecord

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
        array = src.read(
            1,
            out_shape=(size, size),
            resampling=Resampling.nearest,
        )
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
    # 正常保留；不正常替换成 mean；image条件成立选用的值；means.expand_as(image)条件不成立选用的值
    image = torch.where(torch.isfinite(image), image, means.expand_as(image))
    # 返回标准化后的结果
    return (image - means) / stds.clamp_min(1e-6)

# 计算影像中心位置
def _centroid(path: str) -> tuple[float, float]:
    with rasterio.open(path) as src:
        bounds = src.bounds
        lon = (bounds.left + bounds.right) / 2.0
        lat = (bounds.bottom + bounds.top) / 2.0
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
        # 以ps为基准
        with rasterio.open(record.ps) as src:
            # 记录PS的空间范围和crs
            reference_bounds, reference_crs = src.bounds, src.crs
        # 依次进行比较
        candidates = [record.s1, record.s2, record.label]
        # 逐个检查
        for path in candidates:
            if path is None:
                continue
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
class FloodPlanetDataset(Dataset[dict[str, Any]]):
    # 数据集初始化 __init__()
    def __init__(
        self,
        # 接收 index.py 生成的样本记录
        records: Sequence[SampleRecord],
        # 接收传感器配置
        sensor_config: dict[str, Any],
        label_size: int = 256,
        #  是否执行数据增强
        augment: bool = False,
        # 是否只保留三模态完整样本
        require_complete: bool = False,
        # 只保留 S1、S2、PS 全部存在的样本
        use_geo_metadata: bool = True,
        validate_geography: bool = True,
    ) -> None:
        self.records = [record for record in records if record.is_complete or not require_complete]
        self.sensor_config = sensor_config
        self.label_size = label_size
        self.augment = augment
        self.use_geo_metadata = use_geo_metadata
        if not self.records:
            raise ValueError("Dataset is empty after applying the completeness filter")
        # 缺失模态没有原图，使用该模态首个已有样本的尺寸创建零占位。
        # 如果整个子集都缺少该模态，使用 PS 尺寸；占位特征由 availability 屏蔽。
        self.missing_shapes = {}
        for modality in MODALITIES:
            path = next(
                (getattr(r, modality) for r in self.records if getattr(r, modality) is not None),
                self.records[0].ps,
            )
            with rasterio.open(path) as src:
                self.missing_shapes[modality] = (src.height, src.width)
        if validate_geography:
            validate_geographic_alignment(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        paths = {"s1": record.s1, "s2": record.s2, "ps": record.ps}
        # 创建空字典，用来保存读取后的影像张量
        images: dict[str, torch.Tensor] = {}
        # 把 index.py 中的模态可用状态转换成 PyTorch 张量
        availability = torch.tensor(record.availability, dtype=torch.bool)

        # MODALITIES定义了三个模态
        for modality in MODALITIES:
            # 从配置文件中取得当前模态的配置
            spec = self.sensor_config[modality]
            path = paths[modality]
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

        target, valid = _read_label(record.label, self.label_size)

        # 对图像进行旋转强化
        if self.augment:
            for dimension in (-1, -2):
                if bool(torch.rand(()) < 0.5):
                    images = {key: value.flip(dimension) for key, value in images.items()}
                    target, valid = target.flip(dimension), valid.flip(dimension)
        # 是否所有模态都存在
        if self.use_geo_metadata:
            # 计算图像的中心位置
            lat, lon = _centroid(record.ps)
            # 把经纬度转换成位置编码
            latlon = _encode_latlon(lat, lon)
        else:
            # 如果不存在的话改为0
            latlon = torch.zeros(4, dtype=torch.float32)


        # 当不知道遥感影像的拍摄时间时，用 4 个 0 作为时间元数据传给 CLAY，而不是随便编一个日期
        time = torch.zeros(4, dtype=torch.float32)

        return {
            "id": record.sample_id,
            "images": images,
            "availability": availability,
            "target": target,
            "valid": valid,
            "time": time,
            "latlon": latlon,
        }
