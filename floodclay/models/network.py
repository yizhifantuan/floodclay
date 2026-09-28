from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .clay_encoder import SharedClayEncoder
# 从项目中的其他文件导入已经写好的模块,负责把模块连接起来
from .modules import (
    CrossModalAttentionFusion,
    MissingModalityGenerator,
    ModalityMaskSampler,
    ScaleAligner,
    SegmentationDecoder,
    SharedRepresentation,
)

# 定义教师学生神经网络
class TeacherStudentFloodModel(nn.Module):
    # 三种模态
    modalities = ("s1", "s2", "ps")
    # 定义初始化函数
    def __init__(self, model_config: dict[str, Any], sensor_config: dict[str, Any]) -> None:
        # 继承父类
        super().__init__()
        # 创建一个解码器对象，用clay_encoder里面的类
        self.encoder = SharedClayEncoder(model_config, sensor_config)
        # 读取特征通道数
        channels = int(model_config["feature_channels"])
        # 读取modalities的长度
        modalities = len(self.modalities)
        # 创建特征对齐模块。它接收编码器特征，并把特征调整到后续模块需要的通道数和空间大小
        self.aligner = ScaleAligner(
            self.encoder.output_dim,
            channels,
            self.modalities,
            int(model_config["feature_size"]),
        )
        # 创建随机“遮模态”的工具。训练学生时，它决定本次让学生看不到哪些模态
        self.mask_sampler = ModalityMaskSampler(
            max_drop=int(model_config.get("max_random_drop", 2)),
            drop_probability=float(model_config.get("drop_probability", 1.0)),
        )
        # 从可用模态中提取共享特征
        self.shared_representation = SharedRepresentation(modalities, channels)
        # 根据已有特征推测缺失模态的特征
        self.generator = MissingModalityGenerator(modalities, channels)
        # 创建两个跨模态注意力融合模块，教师和学生各用一个；注意力头是4
        heads = int(model_config.get("fusion_heads", 4))
        self.teacher_fusion = CrossModalAttentionFusion(modalities, channels, heads)
        self.student_fusion = CrossModalAttentionFusion(modalities, channels, heads)
        # 创建两个解码器。解码器把融合后的特征转换成分割结果
        output_size = int(model_config["output_size"])
        self.teacher_decoder = SegmentationDecoder(channels, output_size)
        self.student_decoder = SegmentationDecoder(channels, output_size)
    # 把原始输入变成对齐特征
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        base = self.encoder(batch["images"], batch["time"], batch["latlon"])
        return self.aligner(base)
    # 教师分支
    def _teacher_branch(self, aligned: torch.Tensor) -> dict[str, torch.Tensor]:
        # 创建一个全是 True 的表，表示每个样本的每个模态都可用
        complete = torch.ones(
            aligned.shape[0], aligned.shape[1], device=aligned.device, dtype=torch.bool
        )
        # 计算共享特征
        shared = self.shared_representation(aligned, complete)
        # 融合三个模态。融合模块返回融合特征 fused 和注意力结果 attention
        fused, attention = self.teacher_fusion(aligned, complete)
        # 把共享特征加到融合特征上
        fused = fused + shared
        # 然后解码成分割输出
        decoded = self.teacher_decoder(fused)
        return {**decoded, "shared": shared, "fused": fused, "attention": attention}
    # availability 是布尔表：True 表示该模态可用，False 表示不可用。学生只根据指定的可用情况计算共享特征
    def _student_branch(
        self, aligned: torch.Tensor, availability: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # 首先得到共享特征
        shared = self.shared_representation(aligned, availability)
        # 生成器利用现有模态和共享特征，补出缺失模态对应的特征
        completed, generated = self.generator(aligned, availability, shared)
        # 融合特征、加上共享特征，再生成学生的分割结果
        fused, attention = self.student_fusion(completed, availability)
        fused = fused + shared
        decoded = self.student_decoder(fused)
        return {
            **decoded,
            "shared": shared,
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
        # 只要有样本缺失就报错
        if not bool(actual.all()):
            raise ValueError(
                "Teacher training requires complete S1/S2/PS samples. "
                "Construct the training dataset with require_complete=True."
            )
        # 先编码输入
        aligned = self.encode(batch)
        # 如果如果没有手动指定学生能看哪些模态，则随机生成，
        if student_mask is None:
            student_mask = self.mask_sampler(actual)
        else:
            student_mask = student_mask.bool() & actual
        # 沿着模态这一维统计每个样本还剩几个 True。如果有样本一个模态都不剩，就报错
        if torch.any(student_mask.sum(1) == 0):
            raise ValueError("Student mask cannot remove all modalities")
        return {
            "teacher": self._teacher_branch(aligned),
            "student": self._student_branch(aligned, student_mask),
            "aligned": aligned,
            "student_mask": student_mask,
            "actual_availability": actual,
        }
    # 实际预测时走的流程
    def predict(
        self,
        batch: dict[str, Any],
        availability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference forward using real or scenario-forced availability."""
        # 获取模态缺失编码
        actual = batch["availability"].bool()
        # 确保可以故意把真实存在的数据关掉，但不能把真实不存在的数据强行打开
        selected = actual if availability is None else (availability.bool() & actual)
        # 检查这一批数据里，有没有某个样本最后一个可用模态都没有
        if torch.any(selected.sum(1) == 0):
            raise ValueError("At least one genuinely available modality must remain")
        # 都是调用当前模型里面已经定义好的功能
        aligned = self.encode(batch)
        # 根据 selected 指定的可用模态，把对应特征送进 student 模型完成预测
        output = self._student_branch(aligned, selected)
        output["availability"] = selected
        return output

# 这是一个方便创建模型的函数。它从总配置 config 中取出模型配置和传感器配置，然后实例化上面的类
def build_model(config: dict[str, Any]) -> TeacherStudentFloodModel:
    return TeacherStudentFloodModel(config["model"], config["sensors"])
