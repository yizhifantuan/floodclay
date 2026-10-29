# 完整流程测试

在 IDE 中选择 `D:/anaconda3/envs/floodclay/python.exe` 解释器，直接运行相应文件的 `main()`，无需填写运行参数。

| 文件 | 用途 | 设置位置 |
| --- | --- | --- |
| `scripts/train.py` | 训练并逐轮验证 | `configs/default.json` |
| `scripts/evaluate.py` | 评估已训练模型 | `main()` 开头；默认选择输出根目录下最新一个已有 `best.pt` 的实验，恢复其训练配置与固定数据划分 |
| `scripts/predict.py` | 导出单个样本的预测图 | `main()` 开头的 `sample_id`、`checkpoint`、`threshold` 和 `output_dir` |
| `scripts/train_and_evaluate.py` | 依次调用训练和评估函数 | `configs/default.json` |
| `test/test_pipeline.py` | 小样本完整流程检查 | `main()` 开头的轮数、样本数、设备与精度设置；PS 设置读取正式配置 |
| `test/test_ps_detail.py` | PS 双路输入与训练回归检查 | 使用随机小型 Clay 主干，直接运行，无需真实数据或 Clay 权重 |

默认模型分割/边界输出与原始标签均为 1024×1024，标签直接按原始尺寸读取，不提供标签缩放开关；标签尺寸与配置不一致时直接报错。输出文件保留模型网格，并按参考范围调整地理变换。

三路样本在各事件目录内按文件名并集合并，PS 也可以真实缺失。地理参考按 PS → S1 → S2 选择存在的影像，预测不要求标签。教师训练要求三路文件齐全。

训练配置默认从 `configs/default.json` 读取；代码中调用 `load_config()` 即可获得配置字典。相对数据、权重和输出路径仍以项目根目录为基准。

正式训练的 `training.output_dir` 默认是 `runs`。每次启动一次完整实验时只新建一个 `runs/run_年月日_时分秒_微秒/`，全部训练轮次、逐轮验证和最后的测试共用该目录。目录与原来的 `runs/default/` 并列，旧目录及其内部文件不会改动。目录在训练开始时创建，方便及时保存权重；每轮不会另建目录。目录中保存 `config.json`、`split_manifest.json`、`history.jsonl`、`last.pt` 和 `best.pt`；运行 `scripts/train_and_evaluate.py` 后，`evaluation.json` 也写入同一目录。再次启动完整实验才会另建一个目录，目录名冲突时自动追加序号。实验内部逐轮更新 `last.pt`，出现更优验证 IoU 时更新 `best.pt`。训练返回本次 `best.pt` 的路径，组合入口直接评估此路径。单独评估或预测默认选择最新一个已有 `best.pt` 的实验，同时兼容旧的 `runs/default/run_*/best.pt` 和 `runs/default/best.pt`。如需评估特定实验，将其权重路径传给 `scripts.evaluate.main(checkpoint)`；预测可修改 `main()` 中的 `checkpoint`。

AlphaEarth 辅助分支默认关闭，在配置文件中修改 `alphaearth.enabled` 和 `alphaearth.root`。评估恢复检查点中的训练配置；消融可在评估 `main()` 开头设置 `disable_alphaearth=True`。

当前 `patch_size=8`，直接严格加载 Clay 原始 8×8 Patch 权重，不对权重重采样。PS 在模型内部缩放后，位置编码使用有效 GSD：原始 GSD × 原图边长 / Clay 输入边长；例如 1024→512 时为 3×2=6 米。S1/S2 使用原始 GSD，不做 Patch 尺寸补偿。此前按 Patch=16 训练的完整分割检查点不能直接用于当前 Patch=8 模型。

主干默认设置为 `freeze_clay=true`、`unfreeze_clay_last_blocks=3`：冻结其余 Clay 参数，仅训练最后三个 Transformer Block（Large 为第 22、23、24 层，含 Attention、FeedForward 和 Block 内部 LayerNorm）。Patch Embedding、CLS token、前 21 层和 Transformer 末尾的独立 LayerNorm 保持冻结；Adapter 及其他任务模块照常训练。开放的 Block 跟随训练/评估模式切换，由现有 AdamW 自动收集，使用配置中的学习率及余弦调度。启动训练时打印 Clay 主干可训练参数量。将数量设为 `0` 可完全冻结主干，设为 `1` 可恢复仅开放最后一层，设为 `3` 开放最后三层；数量必须为不超过模型深度的非负整数。设置 `freeze_clay=false` 则训练整个主干。新数量配置优先于旧布尔开关；旧检查点只有 `unfreeze_clay_last_block=true/false` 时仍分别开放一层/零层，两个配置都没有时保持完全冻结。配置和更新后的主干权重均随检查点保存，评估旧模型时沿用检查点配置。开放更多层会增加梯度与优化器状态显存。

PS 默认采用双路输入：原图送入轻量 CNN，同时生成 512×512 输入送入 Clay。配置位于 `model.ps_encoder_size=512` 和 `model.ps_detail_enabled=true`。CNN 使用四级深度可分离卷积，通道为 16/32/64/64，原图 1024×1024 时输出 64×64 特征；1×1 卷积投影至 Clay 通道数，在最后一层 Adapter 和最终归一化后与 PS 特征残差相加，再进入现有特征对齐。Large 配置新增 73,653 个参数。
融合系数 scale 从 0 开始：第一步先更新 scale，随后梯度进入 CNN。教师和学生共用 CNN 参数，按各自模态掩码分别执行；PS 被隐藏时，原图在 CNN 前屏蔽，融合结果再次屏蔽，全批次缺 PS 时跳过 CNN。CNN 不直接改写编码器的 ShareF。
数据集和流程测试始终保留原尺寸 PS；当前方形 Clay 输入要求原 PS 为方形，避免单一 GSD 无法表达非等比例缩放。标签和分割输出仍为 1024×1024。融合特征随后会对齐到 16×16，原图 CNN 有助于补充局部表征，但不能保证完整保留精细边界。
新模型需重新训练；预测和评估均恢复检查点的训练配置。没有这两个新配置的旧检查点按原输入尺寸、不构建 CNN 的结构恢复。
正式 `scripts/train.py` 每轮在验证开始前打印训练峰值显存，并写入 `history.jsonl` 的 `train_gpu_memory`：`peak_allocated_gib` 为张量占用峰值，`peak_reserved_gib` 为 PyTorch 保留峰值（含缓存），单位 GiB；CPU 下为 null。

完整流程入口为 `test/test_pipeline.py` 的 `main()`，使用真实 FloodPlanet 数据和配置中的 Clay 预训练权重。
默认数据根目录为 `D:/桌面/课题/数据/FloodPlanet`，影像位于 `FloodPlanet/<事件名>/{PS,S1,S2}/`，标签位于 `FloodPlanet/<事件名>/labels/`，事件名直接取文件夹名，不再提取影像文件名前缀。
时间从 `stac_catalog/<事件名>/<参考传感器>/<样本ID>/<样本ID>.json` 的 `properties.datetime` 读取，统一为 UTC，按 ISO 周数和小时生成 `[sin(2π×周/52), cos(2π×周/52), sin(2π×小时/24), cos(2π×小时/24)]`，作为 `batch["time"]` 输入 Clay；单个样本形状为 `[4]`，批次为 `[B,4]`。时间和地理参考使用相同传感器优先级，当前数据各模态的目录日期相同。
当前 STAC 时间均为零点，同一事件内日期相同：这里只使用目录提供的日期，不能据此认定已获得每张影像的精确拍摄时刻。缺少时间目录时直接报错，不再使用全零时间占位。样本时间随训练划分清单保存，恢复评估时检查一致性。
此次切换会按事件文件夹名重新分组，并启用时间编码；请重新训练模型，旧清单不能直接替换为新的事件划分。
该数据的 PS 原始分辨率为 3 米，配置使用 `gsd=3.0`；S1/S2 分别为 2/10 波段，PS 为 4 波段。标签使用 1 表示非水体、2 表示水体，0 作为无效区域处理。
Clay 源码位于项目根目录的 `claymodel/`；总测试会检查实际加载路径，防止误用环境中的旧副本。

在 IDE 中打开 `test/test_pipeline.py`，直接点击运行。

默认训练 1 轮：训练集 2 个完整样本，验证集和测试集各 1 个完整样本。
实际可用模态评估另外从测试集取最多 1 个真实缺失样本；如果该测试集没有缺失样本，报告会标明数量为 0。
训练、验证和测试按洪水事件分组，三个子集的事件不重叠。

本完整流程测试和正式入口采用相同输入策略：S1/S2 使用原始影像尺寸（如 311×311）；数据集保留 PS 原图供 CNN 使用，模型内部按 `configs/default.json` 的 `model.ps_encoder_size` 生成 Clay 输入，默认 512×512。测试脚本不提前缩小 PS。标签、输出尺寸、特征通道、融合、解码器和全部损失沿用正式配置。
运行时打印验证参考样本的各模态输入尺寸，结束时打印整个流程的峰值显存和当前占用（MiB），并写入 `evaluation.json` 的 `gpu_memory`。
结束时还打印总运行时长（秒和分钟），并写入 `evaluation.json` 的 `runtime_seconds`。计时使用 `perf_counter()`，从 `main()` 开始，覆盖配置/数据准备、模型加载、训练、验证、权重保存与恢复、测试、预测导出和流程检查；结束前同步 CUDA，不包含模块导入和最后的报告写入/打印。
三路 Clay 共享真实预训练主干权重，默认开放最后三个 Block 学习；每个 Block 后插入一个独立的 MixtureAdapter，默认 large 模型共 24 个。
每个 Adapter 使用三路独立 LayerNorm、共享降维/升维、3×3 深度卷积本模态专家、r→r 通道 MLP 跨模态专家，以及 3r→16→2 位置级软门控。
共享 r→1 评分器在其他两路可用观测中归一化，汇聚互补特征；缺失模态不参与来源评分，只有一路时退回本模态专家。
三路影像应覆盖相同地理范围（数据集已有检查），瓶颈特征在 16×16 网格上交互，再插值回各自 patch 网格；CLS token 不进入 Adapter。
默认瓶颈维度 32，升维层零初始化，可训练残差尺度 gamma 初始为 0.1；不同插入点参数独立，点内除 LayerNorm 外全部共享。
Adapter 数量自动等于 Clay 主干的 Block 数量，插入位置固定覆盖所有 Block，无需配置 `adapter_layers`。
可通过 `adapter_bottleneck_dim`、`adapter_gate_hidden_dim`、`adapter_residual_scale` 调整容量和初始化。
最后一个插入点将各可用分支的低维专家混合结果做掩码平均，通过原有 `shared_output` 投影生成 ShareF，兼容现有解码器和蒸馏损失；不再递归传递上一 Adapter 的 ShareF。
MixtureAdapter 只接收整路模态可用性，已删除训练/推理未接入的 `valid_masks` 接口及相关空间掩码插值。共用屏蔽、掩码 Softmax 和平均操作位于 `floodclay/masks.py`。
来源评分是分割任务学习出的权重，不能解释为传感器可靠性概率。
教师用完整模态编码，学生先确定遮挡后独立编码，避免隐藏影像经 Adapter 泄露。
教师和学生同时训练，教师始终使用完整输入。学生默认 `max_random_drop=2`、`drop_probability=6/7`（JSON 中为 `0.8571428571428571`）：完整输入概率为 `1/7`；触发缺失时，等概率删除一路或两路，再随机选择删除对象，因此六种缺失组合各有 `1/7` 概率。七种组合包括完整输入、缺 S1、缺 S2、缺 PS、仅 S1、仅 S2、仅 PS。每个样本只采样一种组合，有限次抽样的数量可能有波动；至少保留一路真实可用模态。完整输入时没有隐藏模态，重建损失为零，学生仍接受分割、边界和蒸馏监督。验证仍按批次固定轮换缺 S1、缺 S2、缺 PS，用于早停和选择最优权重；训练后的独立测试逐一评估七种组合。此采样调整需要重新训练才能影响权重。
CUDA 上默认沿用配置启用 AMP；CPU 使用普通精度。

流程检查包括四路编码输出、每个 Block 后 Adapter 的执行和有限梯度、PS CNN 分支的有限非零梯度、直接使用 ShareF、参数更新、验证/测试期间无参数更新、最佳权重和优化器恢复、恢复前后预测一致性。
分别改变被遮挡的 S1/S2/PS 影像，检查学生预测、共享表示和 ShareF 完全不变。
测试时使用学生分支，评估完整模态、缺 S1、缺 S2、缺 PS、仅 S1、仅 S2、仅 PS 和实际可用模态，分别输出 IoU、F1、Precision、Recall、Accuracy。
最后保留模型输出网格，按所选参考影像的范围输出概率图和二值掩码 GeoTIFF，并检查写回后的数值、尺寸、坐标系和地理变换。

每次运行会创建独立的 `runs/pipeline_test/run_.../` 目录，保存：

- `config.json`、`split_manifest.json`：实际配置和样本/事件记录；
- `history.json`：每轮训练、验证的全部损失和评价指标；
- `last.pt`、`best.pt`：最后一轮和验证集最优权重，包含优化器状态；
- `evaluation.json`：八种情形的全部指标、流程检查结果和样本数量；
- 报告中的 `reference_input_sizes` 为参考样本各模态原图尺寸，`clay_input_sizes` 为实际送入 Clay 的尺寸；默认 `ps_image_size=512`，`ps_detail_enabled=true`，`input_size_mode=native_images_model_resizes_ps_for_clay`；
- 报告中的 `clay_source_files`：实际加载的项目内 Clay 源码路径；
- `predictions/`：洪水概率图和二值掩码。

增加少量数据或轮数：

```python
# 修改 test/test_pipeline.py 的 main() 开头：
epochs = 2
train_samples = 4
val_samples = 2
test_samples = 2
```

PS 的 Clay 输入尺寸和 CNN 开关统一在 `configs/default.json` 的 `model` 中配置，例如：

```json
"ps_encoder_size": 512,
"ps_detail_enabled": true
```

单独验证新增分支和训练入口：运行 `D:/anaconda3/envs/floodclay/python.exe -B test/test_ps_detail.py`。检查覆盖原尺寸 CNN 输入、Clay 缩放及 GSD、两步训练后 CNN 参数更新、非零融合时缺 PS 的 NaN 隔离、检查点恢复和旧配置兼容。小模型检查不代表真实 Large 模型显存峰值。

使用 CPU 和普通精度时，在同一设置区修改：

```python
device_setting = "cpu"
amp_override = False
```

数据路径和 Clay 权重路径分别在 `configs/default.json` 中修改 `data.root` 和 `model.clay_checkpoint`。

小样本测试的指标用于检查流程，不能代表正式检测精度；IoU 为 0 也可能正常。
真实 large 模型的参数规模保持不变，因此仍需要加载预训练权重，并为保存的检查点预留磁盘空间。
冻结的是 Clay 主干，Adapter 仍然参与训练；各 Transformer Block 直接执行，反向传播仍可经过冻结主干回传到 Adapter。
新双专家 Adapter 与旧 Adapter 的参数结构不同，需要重新训练完整模型权重；原始 Clay 预训练权重仍可严格加载。
升维层零初始化时，第一步分支残差损失只更新升维层；升维层更新后，梯度才传入更早的专家和路由参数。末个 Adapter 的 ShareF 路径也直接提供训练梯度。
成功时输出 `PASS`，失败会抛出异常并以非零退出码结束。
