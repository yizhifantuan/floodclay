from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# 定义不同大小的 Clay 网络参数
# dim:每个patch的特征长度；depth:几层Transformer；heads:几个注意力头；dim_head：每个注意力头维度；mlp_ratio：Transformer 内部 MLP 扩展比例
MODEL_SIZES = {
    "tiny": {"dim": 192, "depth": 6, "heads": 4, "dim_head": 48, "mlp_ratio": 2},
    "small": {"dim": 384, "depth": 6, "heads": 6, "dim_head": 64, "mlp_ratio": 2},
    "base": {"dim": 768, "depth": 12, "heads": 12, "dim_head": 64, "mlp_ratio": 4},
    "large": {"dim": 1024, "depth": 24, "heads": 16, "dim_head": 64, "mlp_ratio": 4},
}

# 定义 Clay v1.5 编码器的适配器
# nn.Module是 PyTorch 所有神经网络模块通常都要继承的基类
class ClayPatchEncoder(nn.Module):
    # 创建这个编码器对象时，会执行 __init__()
    def __init__(
        self,
        # Clay 预训练权重文件的位置
        checkpoint: str | Path,
        # 模型的尺寸
        model_size: str = "large",
        # 每个图像小块的边长
        patch_size: int = 8,
        # 是否冻结 Clay 编码器
        freeze: bool = True,
    ) -> None:
        # 初始化父类
        super().__init__()
        from claymodel.model import Encoder
        # 根据 model_size 从配置字典中找到对应参数
        size = MODEL_SIZES[model_size]
        # 记录 Clay 输出特征的维度
        self.dim = size["dim"]
        # 每一个块的大小
        self.patch_size = patch_size
        # 是否冻结
        self.freeze = freeze
        # 创建真正负责提取特征的 Clay 编码器，并把它保存为
        self.encoder = Encoder(
            # 不遮挡任何 patch，使用完整图像进行特征提取
            mask_ratio=0.0,
            patch_size=patch_size,
            # 不打乱 patch 的顺序
            shuffle=False,
            dim=size["dim"],
            depth=size["depth"],
            heads=size["heads"],
            dim_head=size["dim_head"],
            mlp_ratio=size["mlp_ratio"],
        )
        # 读取预训练权重
        self._load_encoder_weights(Path(checkpoint))
        if freeze:
            # 关闭编码器所有参数的梯度计算
            # 训练时，优化器不会更新这些参数
            self.encoder.requires_grad_(False)
            self.encoder.eval()
    # 读取clay1.5的权重
    def _load_encoder_weights(self, checkpoint: Path) -> None:
        """只读取官方 Clay checkpoint 中的 model.encoder.* 权重。"""
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        prefix = "model.encoder."
        selected = {
            key.removeprefix(prefix): value
            for key, value in payload["state_dict"].items()
            if key.startswith(prefix)
        }
        self.encoder.load_state_dict(selected, strict=True)
    # 重写 train() 方法
    # 返回的是一个 ClayPatchEncoder 类型的对象
    def train(self, mode: bool = True) -> "ClayPatchEncoder":
        # 确保冻结的 Clay 编码器仍然保持推理模式
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self
    # 前向传播
    def forward(
        self,
        # 遥感影像张量
        pixels: torch.Tensor,
        # 影像采集时间信息
        time: torch.Tensor,
        # 影像对应的经纬度信息
        latlon: torch.Tensor,
        # 各遥感波段对应的中心波长
        waves: torch.Tensor,
        # 地面采样距离
        gsd: float,
    ) -> torch.Tensor:# 返回一个张量
        # 决定是否计算梯度
        context = torch.no_grad() if self.freeze else nullcontext()
        # 记录梯度
        with context:
            # 把数据送进 Clay 编码器。
            # Clay 编码器会返回四个结果，但当前代码只需要第一个
            encoded, _, _, _ = self.encoder(
                # Clay 的输入是一个字典，其中包含多种信息
                {
                    "pixels": pixels,
                    "time": time,
                    "latlon": latlon,
                    "waves": waves.to(pixels.device),
                    "gsd": torch.tensor(gsd, device=pixels.device, dtype=pixels.dtype),
                }
            )
        # 删除第一个特殊 token
        # 从编号 1 开始，删除编号 0 的特殊 token
        patch_tokens = encoded[:, 1:, :]
        # 计算 patch 网格高度
        grid_h = pixels.shape[-2] // self.patch_size
        # 计算 patch 网格宽度
        grid_w = pixels.shape[-1] // self.patch_size
        # 检查 token 数量是否正确
        if grid_h * grid_w != patch_tokens.shape[1]:
            raise RuntimeError(f"Clay token count does not match input grid: {patch_tokens.shape}")
        # 把 token 恢复成二维特征图
        return patch_tokens.transpose(1, 2).reshape(pixels.shape[0], self.dim, grid_h, grid_w)

# 共享编码器
class SharedClayEncoder(nn.Module):

    # 三个模态；圆括号表示一个元组 tuple；表示里面的东西不会修改
    modalities = ("s1", "s2", "ps")
    # 定义初始化函数
    # model_config：模型相关配置；sensor_config:不同传感器的配置
    def __init__(self, model_config: dict[str, Any], sensor_config: dict[str, Any]) -> None:
        # 定义父类
        super().__init__()
        # 创建一个 ClayPatchEncoder，保存到 backbone
        self.backbone = ClayPatchEncoder(
            checkpoint=model_config["clay_checkpoint"],
            model_size=model_config["clay_model_size"],
            patch_size=model_config["patch_size"],
            freeze=model_config["freeze_clay"],
        )
        # 读取 Clay 编码器输出的特征维度，并保存下来
        self.output_dim = self.backbone.dim
        # 创建一个空字典，用于保存每个模态的 GSD
        self.gsd: dict[str, float] = {}
        # 遍历三个模态
        for modality in self.modalities:
            # 从传感器配置里获取当前模态的波长，并转换成 PyTorch 张量
            waves = torch.tensor(sensor_config[modality]["waves"], dtype=torch.float32)
            # 把波长张量注册为模型的 buffer
            self.register_buffer(f"waves_{modality}", waves, persistent=False)
            # 读取当前模态的 GSD，并转成浮点数
            self.gsd[modality] = float(sensor_config[modality]["gsd"])
    # 进行前向传播
    def forward(
        self,
        # 都是从当前模型里面拿东西
        images: dict[str, torch.Tensor],
        time: torch.Tensor,
        latlon: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 创建features的空字典
        features = {}
        # 依次处理影像
        for modality in self.modalities:
            # 取出当前模态的影像
            pixels = images[modality]
            # 获取影像的长和宽
            height, width = pixels.shape[-2:]
            # 获取patch_size
            patch_size = self.backbone.patch_size
            # 补边计算需要补多少像素
            pad_h, pad_w = (-height) % patch_size, (-width) % patch_size
            # 判断是否需要补边
            if pad_h or pad_w:
                # F.pad() 是 PyTorch 的补边函数；只在影像右边和下边补像素
                pixels = F.pad(pixels, (0, pad_w, 0, pad_h), mode="replicate")
            # 调用 Clay 编码器；把当前模态的数据交给 Clay 编码器，并将输出保存在结果字典中
            features[modality] = self.backbone(
                pixels,
                time,
                latlon,
                getattr(self, f"waves_{modality}"),
                self.gsd[modality],
            )
        return features
