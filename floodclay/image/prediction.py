"""测试结果出图与 GeoTIFF 导出；所有图像输出逻辑集中在此文件。"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
import torch

from ..data.index import SampleRecord


WATER = np.array([36, 125, 216], dtype=np.float32) / 255
BACKGROUND = np.array([240, 244, 249], dtype=np.float32) / 255
FALSE_POSITIVE = np.array([245, 158, 35], dtype=np.float32) / 255
FALSE_NEGATIVE = np.array([224, 68, 91], dtype=np.float32) / 255
NO_DATA = np.array([182, 190, 200], dtype=np.float32) / 255

SCENARIO_NAMES = {
    "complete": "完整模态", "missing_s1": "缺 S1", "missing_s2": "缺 S2",
    "missing_ps": "缺 PS", "only_s1": "仅 S1", "only_s2": "仅 S2",
    "only_ps": "仅 PS", "natural_missingness": "实际可用模态",
}


def experiment_image_dir(checkpoint: str | Path) -> Path:
    """runs/result 下的实验对应 runs/image 下同名目录，兼容旧权重路径。"""
    experiment = Path(checkpoint).resolve().parent
    result_root = next((parent for parent in experiment.parents if parent.name == "result"), None)
    if result_root is not None:
        return result_root.parent / "image" / experiment.relative_to(result_root)
    if experiment.parent.name == "default":
        return experiment.parent.parent / "image" / "default" / experiment.name
    return experiment.parent / "image" / experiment.name


def write_geotiff(
    reference: str | Path, output: str | Path, array: np.ndarray, dtype: str,
    *, nodata: float | int | None = None,
) -> None:
    """保留输出网格，按参考影像范围调整地理变换；0 不是默认 NoData。"""
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("GeoTIFF prediction must be a nonempty 2D array")
    height, width = array.shape
    with rasterio.open(reference) as src:
        profile = src.profile.copy()
        transform = src.transform * Affine.scale(src.width / width, src.height / height)
    profile.update(
        count=1, height=height, width=width, transform=transform,
        dtype=dtype, nodata=nodata,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(array.astype(dtype, copy=False), 1)


def _read_rgb(record: SampleRecord, shape: tuple[int, int], valid: np.ndarray) -> tuple[np.ndarray, str]:
    # PS 缺失时优先展示 S2 RGB，最后退回 S1 VV 灰度；不展示零占位影像。
    name = next(name for name in ("ps", "s2", "s1") if getattr(record, name) is not None)
    with rasterio.open(getattr(record, name)) as src:
        indexes = [1] if name == "s1" else [3, 2, 1]
        values = src.read(
            indexes, out_shape=(len(indexes), *shape),
            masked=True, resampling=Resampling.bilinear,
        ).astype(np.float32).filled(np.nan).transpose(1, 2, 0)
    if len(indexes) == 1:
        values = np.repeat(values, 3, axis=2)
    rgb = np.zeros_like(values)
    for channel in range(3):
        layer = values[..., channel]
        usable = valid & np.isfinite(layer)
        if usable.any():
            lo, hi = np.percentile(layer[usable], [2, 98])
            rgb[..., channel] = np.clip((layer - lo) / max(float(hi - lo), 1e-6), 0, 1)
    rgb = np.nan_to_num(rgb)
    rgb[~valid | ~np.isfinite(values).all(axis=2)] = NO_DATA
    return rgb, name.upper()


def _comparison_layers(
    rgb: np.ndarray, truth: np.ndarray, prediction: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def mask_image(mask: np.ndarray) -> np.ndarray:
        image = np.broadcast_to(BACKGROUND, (*mask.shape, 3)).copy()
        image[mask & valid] = WATER
        image[~valid] = NO_DATA
        return image

    error = rgb * 0.56 + 0.13
    tp = truth & prediction & valid
    error[tp] = error[tp] * 0.22 + WATER * 0.78
    error[~truth & prediction & valid] = FALSE_POSITIVE
    error[truth & ~prediction & valid] = FALSE_NEGATIVE
    error[~valid] = NO_DATA
    return mask_image(truth), mask_image(prediction), error


@lru_cache(maxsize=1)
def _plot_font() -> tuple[Any, bool]:
    from matplotlib.font_manager import FontProperties, fontManager

    fonts = {font.name: font.fname for font in fontManager.ttflist}
    for name in (
        "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS",
    ):
        if name in fonts:
            return FontProperties(fname=fonts[name]), True
    # Linux 服务器未安装中文字体时使用英文，避免缺字方框。
    return FontProperties(family="DejaVu Sans"), False


def _save_comparison(
    path: Path, record: SampleRecord, probability: np.ndarray,
    truth: np.ndarray, valid: np.ndarray, threshold: float,
    scenario: str, available: Sequence[bool],
) -> None:
    # Agg 直接渲染到文件，服务器无需桌面，也不积累 pyplot Figure。
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rgb, sensor = _read_rgb(record, probability.shape, valid)
    ground_truth, prediction, error = _comparison_layers(rgb, truth, probability >= threshold, valid)
    font, chinese = _plot_font()
    visible = " / ".join(name for name, present in zip(("S1", "S2", "PS"), available) if present)
    scenario_name = SCENARIO_NAMES.get(scenario, scenario) if chinese else scenario
    titles = (
        [f"01   {sensor} 参考影像", "02   真实标签", "03   预测结果", "04   误差对比"]
        if chinese else [f"01   {sensor} reference", "02   Ground truth", "03   Prediction", "04   Error overlay"]
    )
    captions = (
        ["VV 灰度 · 2%–98% 拉伸" if sensor == "S1" else "RGB · 2%–98% 拉伸",
         "由原始标签读取", f"模型真实预测 · 阈值 {threshold:.2f}", "橙色为误检，红色为漏检"]
        if chinese else ["VV grayscale / 2–98% stretch" if sensor == "S1" else "RGB / 2–98% stretch",
                         "From original labels", f"Model output / threshold {threshold:.2f}", "Orange: false positive / red: false negative"]
    )
    fig = Figure(figsize=(18, 7.3), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    try:
        fig.text(0.035, 0.935, "洪涝分割结果对比" if chinese else "Flood segmentation results",
                 fontproperties=font, fontsize=25, color="#172b45", weight="bold")
        subtitle = (
            f"样本  {record.sample_id}     |     场景  {scenario_name}     |     模型输入  {visible}"
            if chinese else f"Sample  {record.sample_id}     |     Scenario  {scenario_name}     |     Model inputs  {visible}"
        )
        fig.text(0.035, 0.88, subtitle, fontproperties=font, fontsize=11.5, color="#62718a")
        fig.text(0.965, 0.94, "模型真实预测" if chinese else "Actual model prediction",
                 ha="right", va="center", fontproperties=font, fontsize=11.5, color="#176346",
                 bbox={"boxstyle": "round,pad=.5", "facecolor": "#e8f5ee", "edgecolor": "none"})
        for left, image, title, caption in zip(
            (0.035, 0.273, 0.511, 0.749), (rgb, ground_truth, prediction, error), titles, captions,
        ):
            ax = fig.add_axes([left, 0.245, 0.216, 0.533])
            ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#dce3ed")
                spine.set_linewidth(0.8)
            fig.text(left, 0.805, title, fontproperties=font, fontsize=13.5, color="#172b45", weight="bold")
            fig.text(left, 0.225, caption, fontproperties=font, fontsize=10, color="#6b778b")
        labels = (
            ["水体 / 正确检出", "非水体", "误检 FP", "漏检 FN", "无效区域"] if chinese
            else ["Water / true positive", "Non-water", "False positive", "False negative", "NoData"]
        )
        legend = [Patch(facecolor=color, label=label) for color, label in zip(
            (WATER, BACKGROUND, FALSE_POSITIVE, FALSE_NEGATIVE, NO_DATA), labels,
        )]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.126), ncol=5,
                   frameon=False, prop=font, handlelength=1.25, columnspacing=2)
        fig.add_artist(Line2D([0.035, 0.965], [0.105, 0.105], transform=fig.transFigure, color="#e8edf4", lw=1))
        note = (
            "参考影像仅用于展示，不代表本场景模型可见；灰色无效区域不参与误差比较。"
            if chinese else "Reference imagery is for display; model inputs are listed above. NoData is excluded from error comparisons."
        )
        fig.text(0.035, 0.062, note, fontproperties=font, fontsize=10.5, color="#6b778b")
        fig.text(0.035, 0.025, f"{record.sample_id}_probability.tif   |   {record.sample_id}_mask.tif",
                 fontproperties=font, fontsize=9.5, color="#6b778b")
        fig.savefig(path, dpi=160, facecolor="white")
    finally:
        fig.clear()


class PredictionImageWriter:
    """从评估现有批次直接导出，按 ID 配对样本，不重新运行模型。"""

    def __init__(
        self, records: Sequence[SampleRecord], output_dir: str | Path, *,
        scenario: str, threshold: float = 0.5, sample_ids: Collection[str] | None = None,
    ) -> None:
        self.records = {record.sample_id: record for record in records}
        if len(self.records) != len(records):
            raise ValueError("预测图的样本 ID 不能重复")
        if not 0 <= threshold <= 1:
            raise ValueError("预测阈值必须在 [0, 1] 内")
        self.sample_ids = set(self.records) if sample_ids is None else set(sample_ids)
        if not self.sample_ids <= self.records.keys():
            raise ValueError("出图样本不在测试清单中")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scenario = scenario
        self.threshold = threshold
        self.written: set[str] = set()

    @property
    def count(self) -> int:
        return len(self.written)

    @torch.no_grad()
    def write_batch(
        self, batch: dict[str, Any], logits: torch.Tensor,
        availability: torch.Tensor | None = None,
    ) -> None:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("出图 logits 必须为 [B,1,H,W]")
        if logits.shape != batch["target"].shape or logits.shape != batch["valid"].shape:
            raise ValueError("预测、标签和有效掩码尺寸必须一致")
        if len(batch["id"]) != logits.shape[0]:
            raise ValueError("样本 ID 数量与预测批次大小不一致")
        # 与指标统计使用相同精度的 sigmoid，再转到 CPU，保证阈值一致。
        probabilities = torch.sigmoid(logits.detach()).float().cpu().numpy()[:, 0]
        targets = batch["target"].detach().cpu().numpy()[:, 0] >= 0.5
        valid_masks = batch["valid"].detach().bool().cpu().numpy()[:, 0]
        selected = batch["availability"] if availability is None else availability
        selected = selected.detach().bool().cpu().tolist()
        for index, sample_id in enumerate(batch["id"]):
            record = self.records[sample_id]
            if sample_id not in self.sample_ids:
                continue
            if sample_id in self.written:
                raise ValueError(f"同一场景重复导出样本：{sample_id}")
            probability, valid = probabilities[index], valid_masks[index]
            if not np.isfinite(probability[valid]).all():
                raise ValueError(f"预测概率包含非有限值：{sample_id}")
            prediction = probability >= self.threshold
            # 无效像素用独立 NoData 值，避免误当成非水体。
            write_geotiff(
                record.reference_path, self.output_dir / f"{sample_id}_probability.tif",
                np.where(valid, probability, -9999.0), "float32", nodata=-9999.0,
            )
            write_geotiff(
                record.reference_path, self.output_dir / f"{sample_id}_mask.tif",
                np.where(valid, prediction, 255), "uint8", nodata=255,
            )
            _save_comparison(
                self.output_dir / f"{sample_id}.png", record, probability,
                targets[index], valid, self.threshold, self.scenario, selected[index],
            )
            self.written.add(sample_id)
