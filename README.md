# FloodPlanet 三模态缺失鲁棒洪水分割

本工程把 S1、S2、PlanetScope（PS）完整串成一个可训练、可评估、可推理的模块化流程：完整三模态教师分支 + 随机缺失学生分支 + 共享 Clay 编码器 + 缺失模态特征生成 + 跨模态注意力融合 + 多层知识蒸馏。

默认数据路径已经写入 `configs/default.json`：

```text
D:/桌面/课题/数据/FloodPlanet/FloodPlanet1
```

## 已核实的数据情况

| 项目 | 数量 |
|---|---:|
| PS 影像 / 标签 | 366 |
| S1 影像 | 362 |
| S2 影像 | 298 |
| 三模态完整样本 | 294 |
| 缺 S1 | 4 |
| 缺 S2 | 68 |

本地数据还有一个重要异常：`S1/masks` 中 9 个 `NPL_*.tif` 是四波段 PS 影像，不是分割标签；`S2/masks` 的 366 份标签均为合法的 1/2 二值掩码。因此数据集类固定使用 `S2/masks` 作为统一真值，映射为 `1→背景、2→洪水`，同时把 `0` 作为无效像素。

PS 发布影像是 4 波段、逐瓦片归一化到 8-bit 的版本，不能直接套 Clay 内置的 8 波段 `planetscope-sr` 统计。本工程采用 `[blue, green, red, nir]` 波长，并使用当前 366 幅影像抽样估计的均值/标准差；S1、S2 使用 Clay 官方元数据统计。

## 模块对应关系

```text
S1 / S2 / PS
    │  波长、GSD、归一化参数不同；编码器权重只有一份
    ▼
SharedClayEncoder                         (步骤 1、2)
    ▼
ScaleAligner：1×1 卷积 + 双线性采样      (步骤 3、4)
    ├──────────── 完整三模态教师 ────────────┐
    │                                        │
    └─ 随机/真实缺失掩码 + 缺失状态编码       │
                    ▼                        │
       SharedRepresentation                  │
                    ▼                        │
       MissingModalityGenerator              │  (步骤 5)
                    ▼                        ▼
       Student Cross-modal Attention   Teacher Cross-modal Attention (步骤 6)
                    ▼                        ▼
       Student Decoder                 Teacher Decoder
                    └─────────┬──────────────┘
                              ▼
       分割监督 + 共享特征蒸馏 + 边界蒸馏 + 预测蒸馏
                  + 被遮蔽模态特征重建             (步骤 7)
```

关键实现约束：

- 教师训练集只取 294 个三模态完整样本，保证教师始终看见真实完整数据。
- 学生对完整样本随机删除 1–2 个模态，且至少保留一个模态。
- 真正缺失的影像用零张量占位，但 `availability` 明确标记，零值绝不当作有效影像参与共享表示。
- 缺失生成器只读取已有模态的聚合特征、共享特征和目标模态 token，不会读取被遮蔽模态特征。
- 生成特征对训练时被遮蔽的真实 Clay 特征做 L1 重建；真实缺失样本没有伪造的重建真值。
- 跨模态注意力在每个特征位置上计算三模态权重，生成模态也可以贡献，但其“缺失/生成”状态通过可学习编码显式告知融合层。
- 数据按事件前缀分组切分，避免同一洪水事件的相邻瓦片同时出现在训练集和测试集。
- 数据集初始化时检查 CRS 与地理范围覆盖率；分辨率和像素尺寸可以不同，进入 Clay 后再统一尺度。

## 文件结构

```text
floodclay/
  data/index.py          文件配对、真实缺失状态、事件级切分
  data/dataset.py        GeoTIFF 读取、归一化、同步增强、范围校验
  models/clay_encoder.py 官方 Clay 空间 token 适配器
  models/modules.py      掩码、尺度对齐、共享表示、生成、注意力、解码器
  models/network.py      教师—学生完整组装
  losses.py              分割、边界、三层蒸馏和特征重建损失
  metrics.py             IoU、F1、Precision、Recall、Accuracy
scripts/
  audit_dataset.py       训练前数据审计
  download_clay.py       下载 Clay v1.5 权重
  train.py               训练
  evaluate.py            7 种固定缺失情形 + 数据真实缺失评估
  predict.py             单瓦片推理并写回带地理信息的 GeoTIFF
tests/                    掩码、模型、损失、真实数据读取测试
```

## 1. 安装

建议使用 Python 3.11/3.12 和带 CUDA 的 PyTorch 环境：

```powershell
conda activate floodclay
cd "D:\Graduate\Flood monitoring\Flood monitoring"
python -m pip install -e .
python -m pip install git+https://github.com/Clay-foundation/model.git
python -c "from claymodel.model import Encoder; print(Encoder)"
```

运行测试或在 IDE 中编辑时，使用安装依赖的同一个 Python 解释器。

官方 Clay v1.5 输入包含归一化影像、波长、GSD、时间和经纬度元数据；编码器输出 patch token。本工程直接实例化官方 `Encoder` 并只加载 checkpoint 中的 `model.encoder.*` 权重，避免为了分割任务同时实例化 Clay 预训练时使用的重建器和视觉教师。

## 2. 下载 Clay 权重

```powershell
python scripts/download_clay.py
```

默认保存为 `checkpoints/clay-v1.5.ckpt`。如果已经有权重，可训练时传入：

```powershell
python scripts/train.py --clay-checkpoint D:\path\to\clay-v1.5.ckpt
```

## 3. 数据审计

影像输入保留各自 GeoTIFF 的原始尺寸，不再使用 `data.image_size`。
共享编码器按模态分别提取特征；不能被 patch 大小整除的影像仅在右侧、
底部复制边缘补齐。随后 `ScaleAligner` 将特征统一到 `model.feature_size`。
标签仍按 `data.label_size` 最近邻采样，与模型的 `model.output_size` 保持一致。
例如 S1/S2 的 334×334 补到 336×336 后产生 42×42 特征，PS 的
1024×1024 产生 128×128 特征（patch=8），再统一到默认的 16×16。
原尺寸 Clay 编码会显著增加显存占用，默认 batch size 为 1；若使用更大批次，
同一模态在该批次内必须具有相同尺寸，否则默认 DataLoader 无法堆叠。

```powershell
python scripts/audit_dataset.py
```

它会报告配对数量、真实缺失、事件级切分及非法掩码，结果保存到 `runs/dataset_audit.json`。

## 4. 正式训练

```powershell
python scripts/train.py
```

默认冻结 Clay、batch size 为 1，适配 6 GB 显存。输出包括：

- `runs/default/best.pt`：验证 IoU 最优权重；
- `runs/default/last.pt`：最后一轮权重；
- `runs/default/history.jsonl`：逐轮损失与指标；
- `runs/default/split_manifest.json`：可复现实验划分。

若仍显存不足，优先把 `feature_channels` 从 256 调到 128、`feature_size` 从 16 调到 8；不要改变 `patch_size=8`，否则无法严格加载 Clay v1.5 权重。

## 5. 缺失情形评估

```powershell
python scripts/evaluate.py --checkpoint runs/default/best.pt
```

程序在完整测试样本上分别评估：完整、缺 S1、缺 S2、缺 PS、仅 S1、仅 S2、仅 PS；还会在事件级测试集中按数据真实缺失状态评估一次。输出写入 checkpoint 同目录的 `evaluation.json`。

## 6. 对真实缺失样本推理

例如 `BOL_1041` 缺少 S2：

```powershell
python scripts/predict.py --checkpoint runs/default/best.pt --sample-id BOL_1041
```

输出：

```text
predictions/BOL_1041_probability.tif
predictions/BOL_1041_mask.tif
```

两个文件都以 PS 影像为地理参考，恢复到 1024×1024，并保留投影与仿射变换。

## 实验解释注意事项

1. 当前发布文件名不含可靠采集时间，因此 `time` 元数据置零；经纬度从 PS GeoTIFF 中心点编码。若后续获得 STAC 日期，应在 `dataset.py` 中补入 Clay 的周期时间编码。
2. PS 是公开发布的 8-bit 逐瓦片归一化版本，和论文实验使用的原始商业影像量纲不同。结果应明确注明这一点。
3. 模态生成发生在特征空间，不生成伪遥感影像；这更适合分割目标，也避免把视觉上合理但光谱不可信的像素当作观测。
4. `lite` 后端的结果不能作为 Clay 复现实验结果，只用于代码和消融检查。

## 参考接口

- Clay 官方模型与安装：<https://github.com/Clay-foundation/model>
- Clay v1.5 基础用法：<https://clay-foundation.github.io/model/getting-started/basic_use.html>
- FloodPlanet 论文与数据说明：<https://spj.science.org/doi/10.34133/remotesensing.0575>
