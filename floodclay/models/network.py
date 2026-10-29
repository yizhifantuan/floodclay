from __future__ import annotations

from typing import Any

import torch
from torch import nn

# 从项目中的其他文件导入已经写好的模块,负责把模块连接起来
from .clay_encoder import SharedClayEncoder
from .alignment import ScaleAligner
from .decoder import SegmentationDecoder
from .fusion import CrossModalAttentionFusion
from .generation import MissingModalityGenerator
from .masking import ModalityMaskSampler
from .alphaearth import AlphaEarthFusion

# 定义教师学生神经网络
class TeacherStudentFloodModel(nn.Module):
    # 三种模态
    modalities = ("s1", "s2", "ps")
    # 定义初始化函数
    def __init__(self, model_config: dict[str, Any], sensor_config: dict[str, Any],
                 alphaearth_config: dict[str, Any] | None = None) -> None:
        # 继承父类
        super().__init__()
        # 创建一个编码器对象，用clay_encoder里面的类
        # 调用的初始化
        self.encoder = SharedClayEncoder(model_config, sensor_config)
        # 读取特征通道数
        channels = model_config["feature_channels"]
        # 读取modalities的长度
        modalities = len(self.modalities)
        # 创建特征对齐模块。它接收编码器特征，并把特征调整到后续模块需要的通道数和空间大小
        self.aligner = ScaleAligner(
            self.encoder.output_dim,
            channels,
            self.modalities,
            model_config["feature_size"],
        )
        # 图中的 ShareF 单独投影，与 F1/F2/F3 对齐到相同通道和空间尺寸。
        self.shared_aligner = ScaleAligner(
            self.encoder.output_dim, channels, ("shared",), model_config["feature_size"],
        )
        # 创建随机“遮模态”的工具。训练学生时，它决定本次让学生看不到哪些模态
        self.mask_sampler = ModalityMaskSampler(
            max_drop=model_config["max_random_drop"],
            drop_probability=model_config["drop_probability"],
        )
        # 根据已有特征推测缺失模态的特征
        self.generator = MissingModalityGenerator(modalities, channels)
        # 创建两个跨模态注意力融合模块，教师和学生各用一个；头数由 fusion_heads 配置决定
        heads = model_config["fusion_heads"]
        self.teacher_fusion = CrossModalAttentionFusion(channels, heads)
        self.student_fusion = CrossModalAttentionFusion(channels, heads)
        # 创建两个解码器。解码器把融合后的特征转换成分割结果
        output_size = model_config["output_size"]
        self.teacher_decoder = SegmentationDecoder(channels, output_size)
        self.student_decoder = SegmentationDecoder(channels, output_size)
        # 在原有模块之后构造，关闭时保持原有参数结构及初始化顺序。
        auxiliary = alphaearth_config or {}
        self.alphaearth_enabled = bool(auxiliary.get("enabled", False))
        self.alphaearth_fusion = (
            AlphaEarthFusion(channels, auxiliary.get("hidden_channels", 64), auxiliary.get("residual_scale", 0.1))
            if self.alphaearth_enabled else None
        )

    def set_alphaearth_enabled(self, enabled: bool) -> None:
        """已构建辅助分支的模型可在运行中关闭/恢复，便于同一权重做对照。"""
        if enabled and self.alphaearth_fusion is None:
            raise ValueError("模型构建时未启用 AlphaEarth；请先在配置中启用并重新构建模型")
        self.alphaearth_enabled = bool(enabled)

    def _add_alphaearth(self, fused: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        if not self.alphaearth_enabled:
            return fused
        if "alphaearth" not in batch or "alphaearth_valid" not in batch:
            raise ValueError("AlphaEarth 已启用，但 batch 缺少向量或有效掩码；请检查数据集配置")
        return self.alphaearth_fusion(fused, batch["alphaearth"], batch["alphaearth_valid"])
    # 把原始输入变成对齐特征
    def encode(
        self, batch: dict[str, Any], availability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.encoder(
            batch["images"], batch["time"], batch["latlon"], availability=availability,
        )
        return self.aligner(base), self.shared_aligner(base)[:, 0]
    # 教师分支
    def _teacher_branch(self, aligned: torch.Tensor, encoder_shared: torch.Tensor,
                        batch: dict[str, Any], availability: torch.Tensor) -> dict[str, torch.Tensor]:
        # 直接使用编码器输出并对齐后的 ShareF，不再重新计算共享表示。
        shared = encoder_shared
        # 融合三个模态。融合模块返回融合特征 fused 和注意力结果 attention
        # forward 已确认所有模态完整，直接复用实际可用状态。
        fused, attention = self.teacher_fusion(aligned, availability)
        # 把共享特征加到融合特征上
        fused = fused + shared
        fused = self._add_alphaearth(fused, batch)
        # 然后解码成分割输出
        decoded = self.teacher_decoder(fused)
        return {**decoded, "shared": shared, "encoder_shared": encoder_shared, "fused": fused, "attention": attention}
    # 学生的 ShareF 已由编码器按 availability 独立计算。
    def _student_branch(
        self, aligned: torch.Tensor, availability: torch.Tensor, encoder_shared: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        # 直接使用学生编码器输出并对齐后的 ShareF。
        shared = encoder_shared
        # 生成器利用现有模态和共享特征，补出缺失模态对应的特征
        completed, generated = self.generator(aligned, availability, shared)
        # 融合特征、加上共享特征，再生成学生的分割结果
        fused, attention = self.student_fusion(completed, availability)
        fused = fused + shared
        fused = self._add_alphaearth(fused, batch)
        decoded = self.student_decoder(fused)
        return {
            **decoded,
            "shared": shared,
            "encoder_shared": encoder_shared,
            "fused": fused,
            "attention": attention,
            "completed": completed,
            "generated": generated,
        }
    # 前向传播
    def forward(
        self,
        batch: dict[str, Any],
        student_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:

        # 读取数据中实际有哪些模态，并转成 True / False
        actual = batch["availability"].bool()
        if not bool(actual.all()):
            raise ValueError("教师训练要求 S1/S2/PS 完整样本")
        # 先确定遮挡。验证传入固定掩码时沿用它，训练未传掩码时随机生成。
        if student_mask is None:
            student_mask = self.mask_sampler(actual)
        else:
            student_mask = student_mask.to(device=actual.device, dtype=torch.bool)
        # 编码器内部已经交换模态信息，学生必须独立按掩码编码，不能复用教师特征。
        aligned, teacher_shared = self.encode(batch, actual)
        student_aligned, student_shared = self.encode(batch, student_mask)
        return {
            "teacher": self._teacher_branch(aligned, teacher_shared, batch, actual),
            "student": self._student_branch(student_aligned, student_mask, student_shared, batch),
            "aligned": aligned,
            "student_aligned": student_aligned,
            "student_mask": student_mask,
            "actual_availability": actual,
        }
    # 走学生分支做预测
    def predict(
        self,
        batch: dict[str, Any],
        availability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference forward using real or scenario-forced availability."""
        # 获取模态缺失编码
        actual = batch["availability"].bool()
        # 未指定组合时使用实际模态状态，否则直接使用调用方提供的掩码。
        selected = actual if availability is None else availability.to(
            device=actual.device, dtype=torch.bool,
        )
        # 都是调用当前模型里面已经定义好的功能
        aligned, encoder_shared = self.encode(batch, selected)
        # 根据 selected 指定的可用模态，把对应特征送进 student 模型完成预测
        output = self._student_branch(aligned, selected, encoder_shared, batch)
        output["availability"] = selected
        return output

# 这是一个方便创建模型的函数。它从总配置 config 中取出模型配置和传感器配置，然后实例化上面的类
def build_model(config: dict[str, Any]) -> TeacherStudentFloodModel:
    return TeacherStudentFloodModel(config["model"], config["sensors"], config.get("alphaearth"))
