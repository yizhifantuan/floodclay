# 按样本名读取 AlphaEarth，并重投影到所选参考影像范围内的固定网格。

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
import torch
from .index import SampleRecord

class AlphaEarthReader:
    def __init__(self, config: dict[str, Any]) -> None:
        # 获取数据根目录，并转换为 Path对象
        root = Path(config["root"])
        self.size = int(config.get("input_size"))
        # 这里创建一个空字典，用来保存：
        # 样本 ID → AlphaEarth 文件路径
        self.paths: dict[str, Path] = {}
        # 递归搜索全部的 TIFF 文件
        for path in sorted(root.rglob("*.tif")):
            # stem文件名去掉扩展名
            if path.stem in self.paths:
                raise ValueError(f"AlphaEarth 样本名重复：{self.paths[path.stem]} 与 {path}")
            # 将地址加入字典中
            self.paths[path.stem] = path

    # 读取文件并返回两个Pytoch张量
    def read(self, record: SampleRecord) -> tuple[torch.Tensor, torch.Tensor]:
        # 与影像检查、位置编码和预测导出使用相同的参考。
        with rasterio.open(record.reference_path) as reference:
            # 获取参考影像的坐标参考系统，并保存到变量中
            target_crs = reference.crs
            # reference.transform 表示参考影像的仿射变换矩阵
            # Affine.scale建立一个缩放矩阵，确保影像缩放后仍然对应原来的地理范围
            target_transform = reference.transform * Affine.scale(
                reference.width / self.size,
                        reference.height / self.size,
            )
        with rasterio.open(self.paths[record.sample_id]) as src:
            # 将图像读取并保存
            values = src.read()
            # 获取有效像素位置；当前数据的掩码已覆盖全部 NoData 像素
            valid = src.dataset_mask() > 0
            # 将无效位置的全部 64 个通道统一设为 NoData
            values[:, ~valid] = -9999.0
            # 创建重投影后的目标数组，初始全部设为 NoData
            aligned = np.full((64, self.size, self.size), -9999.0, dtype=np.float32)
            # 输入已解量化；逐维平均后重新归一化，避免直接插值 int8 编码。
            reproject(
                source=values, destination=aligned,
                src_transform=src.transform, src_crs=src.crs, src_nodata=-9999.0,
                dst_transform=target_transform, dst_crs=target_crs, dst_nodata=-9999.0,
                resampling=Resampling.average, num_threads=1,
            )
        valid = np.isfinite(aligned).all(axis=0) & (aligned != -9999.0).all(axis=0)
        aligned[:, ~valid] = 0
        norm = np.linalg.norm(aligned, axis=0)
        valid &= norm > 1e-6
        aligned /= np.maximum(norm, 1e-6)[None]
        aligned[:, ~valid] = 0
        return torch.from_numpy(aligned), torch.from_numpy(valid[None])
