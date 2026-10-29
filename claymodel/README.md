# 项目内的 Clay v1.5 源码

本目录是原安装环境中 `claymodel==1.5.0` 的 26 个 Python 源文件副本，未修改源码逻辑。
保留了完整包结构，包括当前洪水分割项目未使用的原始预训练、微调和示例模块。
复制时间：2026-10-12。C 盘的原安装包保持不变。

原始项目：<https://github.com/Clay-foundation/model>
安装记录中的提交：`f14e698f3c237cabf8d28dec669a362d66625381`。
来源安装记录见 `UPSTREAM_SOURCE.json`，版本及上游依赖见 `UPSTREAM_METADATA.txt`，许可证见 `LICENSE`。

## 代码位置

- `model.py`：官方 Encoder、Decoder 和 MAE 模型。
- `backbone.py`：Transformer、Attention 和 FeedForward。
- `factory.py`：根据波长生成权重的动态 Patch Embedding。
- `utils.py`：二维 GSD 位置编码、一维波长编码等工具。
- `finetune/`：上游分类、回归、分割等微调代码。

项目的训练、评估、预测及总测试入口将项目根目录置于模块搜索路径首位，
因此 `from claymodel...` 会使用本目录，而不是 C 盘的第三方库副本。
项目打包配置也包含 `claymodel*`，不再需要单独安装 Clay 包。
如果在已加载过旧 Clay 的交互式 Python/Jupyter 会话中使用，需先重启会话。

当前洪水分割流程所需的依赖已在根目录的 `requirements.txt` 和 `pyproject.toml` 中声明。
若单独运行上游的预训练、数据模块或示例，需要另外安装 `UPSTREAM_METADATA.txt` 中列出的相关依赖。
预训练权重仍位于项目的 `checkpoints/`，本次复制不改变网络结构或权重格式。

在项目根目录验证加载位置：

```powershell
python -c "import claymodel; print(claymodel.__file__)"
python test/test_pipeline.py
```

此后可以直接在本目录阅读、注释和修改 Clay 源码；修改模型结构时仍需注意预训练权重兼容性。
