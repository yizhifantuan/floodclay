from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..masks import modality_mask, zero_invalid
from .mixture_adapter import MixtureAdapter
from .ps_detail import PSDetailFusion

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
        # 冻结主干时，允许最后一个 Transformer Block 学习。
        unfreeze_last_block: bool = False,
        # 指定开放末尾多少个 Block；未设置时兼容旧的布尔开关。
        unfreeze_last_blocks: int | None = None,
    ) -> None:
        # 初始化父类
        super().__init__()
        from claymodel.model import Encoder
        # 根据 model_size 从配置字典中找到对应参数
        size = MODEL_SIZES[model_size]
        if unfreeze_last_blocks is None:
            unfreeze_last_blocks = 1 if unfreeze_last_block else 0
        if (
            isinstance(unfreeze_last_blocks, bool)
            or not isinstance(unfreeze_last_blocks, int)
            or not 0 <= unfreeze_last_blocks <= size["depth"]
        ):
            raise ValueError(f"unfreeze_last_blocks must be an integer from 0 to {size['depth']}")
        # 记录 Clay 输出特征的维度
        self.dim = size["dim"]
        # 每一个块的大小
        self.patch_size = patch_size
        # 是否冻结
        self.freeze = freeze
        self.unfreeze_last_blocks = unfreeze_last_blocks
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
            # 先冻结主干，再按配置开放末尾 N 个 Block；N=0 时全部冻结。
            self.encoder.requires_grad_(False)
            if self.unfreeze_last_blocks:
                for block in self.encoder.transformer.layers[-self.unfreeze_last_blocks:]:
                    block.requires_grad_(True)
        self.train(self.training)
    # 读取clay1.5的权重
    def _load_encoder_weights(self, checkpoint: Path) -> None:
        # 读取官方 Clay checkpoint 中的 model.encoder.* 权重。
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        # 定义一个字符串变量 prefix，它的值是 "model.encoder."
        prefix = "model.encoder."
        # Python字典推导式，先写字典内容，再写循环，再写条件
        selected = {
            # 去掉前缀
            key.removeprefix(prefix): value
            # payload["state_dict"]遍历这个字典里面所有的键和值
            # .items() 用于获取字典中的所有键值对
            for key, value in payload["state_dict"].items()
            # startswith() 的作用是：检查字符串是否以指定内容开头
            if key.startswith(prefix)
        }
        # load_state_dict把字典中保存的模型参数加载到神经网络中
        # strict=True 要求 selected 中的参数名称与 self.encoder 所需要的参数名称完全匹配
        self.encoder.load_state_dict(selected, strict=True)
    # 重写 train() 方法
    # 返回的是一个 ClayPatchEncoder 类型的对象
    def train(self, mode: bool = True) -> "ClayPatchEncoder":
        # 冻结部分保持评估模式，末尾可训练 Block 跟随外部模式切换。
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
            if self.unfreeze_last_blocks:
                for block in self.encoder.transformer.layers[-self.unfreeze_last_blocks:]:
                    block.train(mode)
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
        # 单模态入口保持可用；三路交互由 SharedClayEncoder 逐层执行。
        # prepare_tokens 单独处理 Patch Embedding 的冻结；末尾 Block 可独立训练。
        tokens, grid = self.prepare_tokens(pixels, time, latlon, waves, gsd)
        encoded = self.encoder.transformer(tokens)
        return self.to_feature_map(encoded, grid)

    # 把原始遥感影像转换成 Transformer 能够理解的 Token（特征向量）
    # 再给这些 Token 添加位置、时间、经纬度信息，最后加上一个 CLS Token
    def prepare_tokens(
        # pixels 来自 images[modality]
        self, pixels: torch.Tensor, time: torch.Tensor, latlon: torch.Tensor,
        # waves是各波段的中心波长
        waves: torch.Tensor, gsd: float,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        # 官方 Patch Embedding + 位置/时间/经纬度编码 + CLS token。
        # 根据 Patch 的二维位置和影像空间分辨率，生成位置编码
        from claymodel.utils import posemb_sincos_2d_with_gsd
        # Patch Embedding 仍随 freeze 冻结，开放末尾 Block 不改变这里。
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            height, width = pixels.shape[-2:]
            # 计算高和宽分别需要补多少像素，才能被 patch_size 整除
            pad_h, pad_w = (-height) % self.patch_size, (-width) % self.patch_size
            # 张量填充函数
            if pad_h or pad_w:
                # replicate 表示复制边缘像素值进行填充
                pixels = F.pad(pixels, (0, pad_w, 0, pad_h), mode="replicate")
            # 计算影像划分成 Patch 后，有多少行、多少列
            grid = (pixels.shape[-2] // self.patch_size, pixels.shape[-1] // self.patch_size)
            # 将输入的遥感影像划分为 Patch
            # 并通过 Clay 的 Patch Embedding 模块，将每个 Patch 转换成特征向量。
            # [B, (H // 8) * (W // 8), 1024]
            patches, _ = self.encoder.to_patch_embed(pixels, waves.to(pixels.device))
            # 直接按高、宽构造官方位置编码，同时支持非正方形影像。
            # 获取每个patch的位置编码
            # position 的形状为 [N, self.dim - 8]
            position = posemb_sincos_2d_with_gsd(
                h=grid[0], w=grid[1], dim=self.dim - 8,
                # 官方位置编码在 CPU 上构造坐标，避免先放入 GPU 再拷回 CPU。
                gsd=torch.tensor(gsd, device="cpu", dtype=pixels.dtype),
            ).to(device=patches.device, dtype=patches.dtype)
            # 把时间信息 time 和地理位置信息 latlon 拼接在一起
            # 然后让拼接后的数据与 patches 使用相同的设备和数据类型
            # 沿着张量的最后一个维度进行拼接
            metadata = torch.cat((time, latlon), dim=-1).to(device=patches.device, dtype=patches.dtype)
            # 把每个图像 Patch 的位置编码 position 和整张遥感影像的时间、
            # 经纬度信息 metadata 拼接起来，形成新的位置特征
            patches = patches + torch.cat((
                # unsqueeze 在第 几 维增加一个维度
                position.unsqueeze(0).expand(patches.shape[0], -1, -1),
                metadata.unsqueeze(1).expand(-1, patches.shape[1], -1),
            ), dim=-1)
            # 取出 Clay 编码器中的 CLS 特殊标记
            # 将它转换为与 Patch 特征相同的数据类型，并扩展到当前批次中的每张影像
            # 扩展第一维
            cls = self.encoder.cls_token.to(patches.dtype).expand(patches.shape[0], -1, -1)
            # 拼接
            return torch.cat((cls, patches), dim=1), grid
    # 执行一个 Transformer Block
    def forward_block(self, tokens: torch.Tensor, index: int) -> torch.Tensor:
        # 执行一层官方 Transformer；主干冻结时仍允许梯度传回上游 Adapter
        # 取出指定 Block 的两个模块：attention 和 feed_forward
        attention, feed_forward = self.encoder.transformer.layers[index]
        # 先计算自注意力，再加上原始输入，形成残差连接
        tokens = tokens + attention(tokens)
        # 把上一步更新后的特征送入前馈网络，再做一次残差相加，返回当前 Block 的输出
        #  [B, N, D]
        return tokens + feed_forward(tokens)

    # to_feature_map：还原二维特征图
    # tokens：Transformer 的 token 序列
    # grid：Patch 网格的 (行数, 列数)
    def to_feature_map(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        # 去掉第一个 CLS token，只保留空间 Patch token
        patches = tokens[:, 1:, :]
        # Patch 特征序列还原成二维特征图
        # transpose交换维度
        # *解包
        return patches.transpose(1, 2).reshape(tokens.shape[0], self.dim, *grid)

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
            # 旧检查点未配置时，沿用原先的主干冻结策略。
            unfreeze_last_block=model_config.get("unfreeze_clay_last_block", False),
            unfreeze_last_blocks=model_config.get("unfreeze_clay_last_blocks"),
        )
        # 读取 Clay 编码器输出的特征维度，并保存下来
        self.output_dim = self.backbone.dim
        self.depth = len(self.backbone.encoder.transformer.layers)
        # 每个 Block 后都有一个独立的 MixtureAdapter；Clay-Large 共 24 个。
        self.adapter_layers = tuple(range(1, self.depth + 1))
        # Adapter 的瓶颈维度
        bottleneck_dim = int(model_config.get("adapter_bottleneck_dim"))
        # 设置 Adapter 使用的网格大小
        grid_size = int(model_config.get("adapter_grid_size"))
        # 创建所有 MixtureAdapter
        self.adapters = nn.ModuleDict({
            # nn.ModuleDict 的键必须是字符串，所以把层编号转换为字符串
            str(layer): MixtureAdapter(
                self.output_dim, bottleneck_dim, grid_size, self.modalities,
                gate_hidden_dim=int(model_config.get("adapter_gate_hidden_dim", 16)),
                residual_scale=float(model_config.get("adapter_residual_scale", 0.1)),
            )
            # 遍历层编号
            for layer in self.adapter_layers
        })
        # 用 1×1 卷积将瓶颈通道数恢复为 Clay 主干的特征维度
        self.shared_output = nn.Conv2d(bottleneck_dim, self.output_dim, 1, bias=False)
        # 用普通字典保存各模态的波长和 GSD
        self.waves = {}
        self.gsd = {}
        # 遍历三个模态
        for modality in self.modalities:
            # 从传感器配置里获取当前模态的波长，并转换成 PyTorch 张量
            self.waves[modality] = torch.tensor(
                sensor_config[modality]["waves"], dtype=torch.float32,
            )
            # 读取当前模态的 GSD，并转成浮点数
            self.gsd[modality] = float(sensor_config[modality]["gsd"])

        # 旧检查点没有新配置时，沿用原有结构及输入尺寸。
        self.ps_encoder_size = model_config.get("ps_encoder_size")
        if self.ps_encoder_size is not None:
            size = self.ps_encoder_size
            if (isinstance(size, bool) or not isinstance(size, int)
                    or size < self.backbone.patch_size
                    or size % self.backbone.patch_size):
                raise ValueError("ps_encoder_size 必须是 patch_size 的正整数倍")
        self.ps_detail = (
            PSDetailFusion(len(sensor_config["ps"]["waves"]), self.output_dim)
            if model_config.get("ps_detail_enabled", False) else None
        )
    # 进行前向传播
    def forward(
        self,
        # 都是从当前模型里面拿东西
        images: dict[str, torch.Tensor],
        time: torch.Tensor,
        latlon: torch.Tensor,
        availability: torch.Tensor | None = None,
        # 返回 s1/s2/ps（F1/F2/F3）和 ShareF。三路共享 Clay 权重。
    ) -> dict[str, torch.Tensor]:
        # 取出 S1 影像，作为获取批次大小和设备的参考
        reference = images[self.modalities[0]]
        # 将 availability 移动到与 reference 相同的设备，并转换为布尔类型
        availability = availability.to(
            device=reference.device,
            dtype=torch.bool
        )
        # 创建两个字典
        # tokens 保存每种模态的 token 序列
        # grids：保存每种模态的 Patch 网格尺寸
        tokens, grids = {}, {}
        # 遍历 self.modalities 中的所有模态，同时获取每个模态的编号 index 和名称
        # enumerate() 会给列表或元组中的每个元素分配一个编号，编号默认从 0 开始
        for index, modality in enumerate(self.modalities):
            # 在 Patch Embedding 之前遮挡，防止隐藏影像进入任何跨模态交互。
            # 取出当前模态在所有样本中的可用状态，将形状从 [B] 改成 [B, 1, 1, 1]
            mask = modality_mask(availability[:, index], images[modality])
            # 对当前模态： 可用：保留原影像；不可用：整张影像置零
            pixels = zero_invalid(images[modality], mask)
            gsd = self.gsd[modality]
            if modality == "ps" and self.ps_encoder_size is not None:
                height, width = pixels.shape[-2:]
                target = self.ps_encoder_size
                if (height, width) != (target, target):
                    # 双线性插值
                    pixels = F.interpolate(
                        pixels, size=(target, target), mode="bilinear",
                        align_corners=False, antialias=True,
                    )
                # 地理范围不变；例如原图 1024→512 时，GSD 从 3 变为 6 米。
                gsd = gsd * height / target
            # 把当前模态的遥感影像、时间、经纬度、波长和空间分辨率传入 prepare_tokens() 函数，将影像转换成 Token，并保存 Token 和 Patch 网格大小
            # 完成 影像补齐 → Patch Embedding → 加入位置/时间/经纬度信息 → 添加 CLS
            tokens[modality], grids[modality] = self.backbone.prepare_tokens(
                pixels, time, latlon, self.waves[modality], gsd,
            )

        shared = None
        # 层级同步：三路都执行当前 Block 后，才进入同一个 Mixture Adapter。
        # 当前 large 模型有 24 层，所以 index 从 0 到 23
        for index in range(self.depth):
            for modality in self.modalities:
                # 让每个模态分别通过当前 Block
                tokens[modality] = self.backbone.forward_block(tokens[modality], index)
            # 把当前 Transformer Block 的索引 index 加 1，然后转换成字符串，保存到变量 layer 中
            layer = str(index + 1)
            # 将每种模态的 token 转成二维特征图
            maps = {name: self.backbone.to_feature_map(value, grids[name]) for name, value in tokens.items()}
            # 每个插入点独立处理三路特征，不再传入上一层共享表示。
            enhanced, shared = self.adapters[layer](maps, availability)
            # 让 Adapter 对 Clay 提取的特征进行补充，而不是直接替换原来的特征
            for modality in self.modalities:
                # 将增强后的 patch 特征转回 token；Adapter 内已完成残差相加。
                # [B, 1024, 42, 42]->[B, 1024, 1764]->[B, 1764, 1024]
                # flatten(2)从第 2 维开始，将后面的维度展平
                # transpose(1, 2) 交换第 1 维和第 2 维
                patches = enhanced[modality].flatten(2).transpose(1, 2)
                # Adapter 只回注空间 token；CLS token 沿官方主干继续传播。
                current = tokens[modality]
                # 保留 CLS token，与增强后的空间 token 重新拼接
                # current[B, N + 1, 1024] cls # [B,     1, 1024]
                tokens[modality] = torch.cat((current[:, :1], patches), dim=1)
        # 创建一个空字典，用来保存最终输出的特征图。把 S1、S2、PS 和 shared 的特征放进去
        features = {}
        # 遍历三种模态
        # enumerate() 会同时获取索引和模态名称
        for index, modality in enumerate(self.modalities):
            # 对 Token 进行归一化；调用 Clay Transformer 的最终归一化层
            normalized = self.backbone.encoder.transformer.norm(tokens[modality])
            # 把 Token 转换成特征图
            feature = self.backbone.to_feature_map(normalized, grids[modality])
            # 取出当前模态的可用状态，再调整形状，让它能够作用于整张特征图
            mask = modality_mask(availability[:, index], feature)
            # 根据模态是否存在，决定是否保留特征
            features[modality] = zero_invalid(feature, mask)
        # 最后一个插入点的专家混合结果生成 ShareF，兼容解码器/蒸馏接口。
        # 把最后一个 Adapter 的共享特征恢复到 Clay 的通道数，再存入输出字典
        if self.ps_detail is not None:
            features["ps"] = self.ps_detail(
                images["ps"], features["ps"], availability[:, 2],
            )
        features["shared"] = self.shared_output(shared)
        return features
