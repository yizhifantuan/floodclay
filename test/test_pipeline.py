"""真实完整模型的小规模训练、验证、权重恢复、测试集评估和 GeoTIFF 预测。

在 IDE 中直接运行本文件的 main()。
默认：训练 2 个样本，验证/测试各 1 个，额外 1 个真实缺失样本，1 轮。
S1/S2 保留原始尺寸，PS 原图送入 CNN，Clay 输入尺寸从配置读取，默认 512×512。
结束时打印整个流程的 CUDA 显存峰值和当前占用，并写入 evaluation.json（单位 MiB）。
同时打印从 main() 开始到流程检查结束的总耗时，并写入 runtime_seconds（单位秒）。
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import importlib
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]

# 直接调用 conda 解释器时也能找到 GeoTIFF 的投影数据库。
proj_directory = Path(sys.prefix) / "Library/share/proj"
if (proj_directory / "proj.db").is_file():
    os.environ.setdefault("PROJ_DATA", str(proj_directory))

import numpy as np
import rasterio
from affine import Affine
import torch
from torch.utils.data import DataLoader, Dataset

from floodclay.config import load_config
from floodclay.data import FloodPlanetDataset, scan_floodplanet, split_by_event
from floodclay.engine import load_checkpoint, move_to_device, run_epoch, save_checkpoint, seed_everything
from floodclay.losses import MultiLevelDistillationLoss
from floodclay.models import build_model
from floodclay.models.mixture_adapter import MixtureAdapter
from scripts.evaluate import SCENARIOS, evaluate_loader
from scripts.predict import write_geotiff

METRIC_NAMES = ("iou", "f1", "precision", "recall", "accuracy")


class PipelineDataset(Dataset):
    """保留全部影像原尺寸，模型内部处理 PS 两路输入。"""

    def __init__(self, source: FloodPlanetDataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict:
        sample = self.source[index]
        if not bool(sample["valid"].any()):
            raise RuntimeError(f"样本 {sample['id']} 没有有效标签像素")
        return sample


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def collect_gpu_memory(device: torch.device) -> dict | None:
    """统计当前设备的 PyTorch 显存；reserved 包含缓存，不代表张量实际占用。"""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    mib = 1024 ** 2
    return {
        "device_name": torch.cuda.get_device_name(device),
        "allocated_mib": torch.cuda.memory_allocated(device) / mib,
        "reserved_mib": torch.cuda.memory_reserved(device) / mib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / mib,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / mib,
    }


def check_values(values: dict[str, float], stage: str, *, metrics: bool = False) -> None:
    if not values or not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"{stage} 结果为空或包含 NaN/Inf")
    if metrics and (set(values) != set(METRIC_NAMES) or any(not 0 <= value <= 1 for value in values.values())):
        raise RuntimeError(f"{stage} 指标不完整或超出 [0, 1]")


def state_versions(model: torch.nn.Module) -> dict[str, int]:
    return {name: value._version for name, value in model.state_dict().items()}


def make_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"],
    )


def check_local_clay_source() -> dict[str, str]:
    """项目必须使用项目内的 Clay 源码，而不是用户级 site-packages 副本。"""
    local_root = (PROJECT / "claymodel").resolve()
    sources = {}
    for name in ("claymodel", "claymodel.model", "claymodel.backbone", "claymodel.factory", "claymodel.utils"):
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(local_root):
            raise RuntimeError(f"{name} 未从项目内加载：{source}")
        sources[name] = str(source)
    return sources


def check_gradients(model: torch.nn.Module) -> None:
    names = ["aligner", "shared_aligner", "generator", "teacher_fusion",
             "student_fusion", "teacher_decoder", "student_decoder"]
    if model.alphaearth_enabled:
        names.append("alphaearth_fusion")
    for name in names:
        gradients = [parameter.grad for parameter in getattr(model, name).parameters() if parameter.grad is not None]
        if (not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
                or not any(bool(gradient.abs().sum() > 0) for gradient in gradients)):
            raise RuntimeError(f"{name} 没有有效梯度")
    for name, module in {**dict(model.encoder.adapters.items()), "shared_output": model.encoder.shared_output}.items():
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        if (not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
                or not any(bool(gradient.abs().sum() > 0) for gradient in gradients)):
            raise RuntimeError(f"Mixture Adapter {name} 没有有效梯度")
    if model.encoder.ps_detail is not None:
        gradients = [p.grad for p in model.encoder.ps_detail.parameters() if p.grad is not None]
        if (not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients)
                or not any(bool(g.abs().sum() > 0) for g in gradients)):
            raise RuntimeError("PS CNN 分支没有有效梯度；scale 零初始化时第一步仅更新 scale")
    if any(parameter.grad is not None for parameter in model.encoder.backbone.parameters()):
        raise RuntimeError("已冻结的 Clay 主干收到了梯度")


def check_mixture_adapter() -> None:
    """验证零初始化、缺失模态隔离、路由和分支梯度。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        names = ("s1", "s2", "ps")
        adapter = MixtureAdapter(12, 4, 5, names)
        features = {name: torch.randn(1, 12, 5, 5) for name in names}
        complete = torch.ones(1, 3, dtype=torch.bool)
        enhanced, shared = adapter(features, complete)
        for name in names:
            torch.testing.assert_close(enhanced[name], features[name], rtol=0, atol=0)
        assert shared.shape == (1, 4, 5, 5)
        # 只从分支输出反传，避免 ShareF 捷径掩盖分支梯度被切断。
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        loss = sum(enhanced[name].square().mean() for name in names)
        loss.backward()
        assert adapter.up.weight.grad.abs().sum() > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        enhanced, _ = adapter(features, complete)
        sum(enhanced[name].square().mean() for name in names).backward()
        for module in (adapter.down, adapter.local_expert, adapter.cross_expert,
                       adapter.source_scorer, adapter.router, adapter.norms):
            gradients = [p.grad for p in module.parameters()]
            assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
            assert any(bool(g.abs().sum() > 0) for g in gradients)
        assert adapter.gamma.grad.abs().sum() > 0

        with torch.no_grad():
            # 穷举所有非空模态组合；隐藏分支变成 NaN 也不能污染可用分支/ShareF。
            for bits in range(1, 8):
                available = torch.tensor([[bool(bits & (1 << i)) for i in range(3)]])
                original, original_shared = adapter(features, available)
                hidden = {name: features[name] if available[0, i] else torch.full_like(features[name], float("nan"))
                          for i, name in enumerate(names)}
                changed, changed_shared = adapter(hidden, available)
                torch.testing.assert_close(original_shared, changed_shared, rtol=0, atol=0)
                assert bool(torch.isfinite(changed_shared).all())
                for i, name in enumerate(names):
                    if available[0, i]:
                        torch.testing.assert_close(original[name], changed[name], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(original[name], features[name], rtol=0, atol=0)
            # 跨模态专家不能读取目标本模态：改变 S1，S1 的互补输入不变。
            complements = []
            hook = adapter.cross_expert.register_forward_pre_hook(
                lambda module, inputs: complements.append(inputs[0].clone()))
            try:
                adapter(features, complete)
                altered = {**features, "s1": torch.randn_like(features["s1"])}
                adapter(altered, complete)
            finally:
                hook.remove()
            torch.testing.assert_close(complements[0], complements[3], rtol=0, atol=0)
            assert not torch.equal(complements[1], complements[4])
            # 仅 S1 时，cross 的大偏置不应影响输出。
            only_s1 = torch.tensor([[True, False, False]])
            before, _ = adapter(features, only_s1)
            adapter.cross_expert[0].bias.add_(1000)
            after, _ = adapter(features, only_s1)
            torch.testing.assert_close(before["s1"], after["s1"], rtol=0, atol=0)
            adapter.cross_expert[0].bias.sub_(1000)
            # 原生 patch 网格可以不同，残差输出必须回到各自尺寸。
            different = {name: torch.randn(1, 12, 3 + i, 6 + i) for i, name in enumerate(names)}
            enhanced, shared = adapter(different, complete)
            assert all(enhanced[name].shape == different[name].shape for name in names)
            assert bool(torch.isfinite(shared).all())


@torch.no_grad()
def check_encoder_outputs(model: torch.nn.Module, batch: dict) -> None:
    model.eval()
    encoder = model.encoder
    expected_layers = tuple(range(1, encoder.depth + 1))
    if encoder.adapter_layers != expected_layers or tuple(encoder.adapters) != tuple(map(str, expected_layers)):
        raise RuntimeError("每个 Block 后必须有一个独立的 MixtureAdapter，包括最后一个 Block")
    calls = {name: 0 for name in encoder.adapters}

    def record_call(name):
        def hook(module, inputs, output):
            calls[name] += 1
        return hook

    hooks = [module.register_forward_hook(record_call(name)) for name, module in encoder.adapters.items()]
    try:
        outputs = encoder(batch["images"], batch["time"], batch["latlon"], batch["availability"])
    finally:
        for hook in hooks:
            hook.remove()
    if any(count != 1 for count in calls.values()):
        raise RuntimeError("编码器单次前向必须执行每个 Adapter 一次")
    if set(outputs) != {"s1", "s2", "ps", "shared"}:
        raise RuntimeError("编码器没有返回 F1/F2/F3/ShareF 四路输出")
    for name in encoder.modalities:
        image = batch["images"][name]
        input_shape = image.shape[-2:]
        if name == "ps" and encoder.ps_encoder_size is not None:
            input_shape = (encoder.ps_encoder_size, encoder.ps_encoder_size)
        grid = tuple((size + encoder.backbone.patch_size - 1) // encoder.backbone.patch_size for size in input_shape)
        expected = (image.shape[0], encoder.output_dim, *grid)
        if tuple(outputs[name].shape) != expected or not bool(torch.isfinite(outputs[name]).all()):
            raise RuntimeError(f"编码器输出 {name} 尺寸或数值有误")
    shared_grid = encoder.adapters[str(encoder.adapter_layers[-1])].grid_size
    expected_shared = (batch["availability"].shape[0], encoder.output_dim, shared_grid, shared_grid)
    if tuple(outputs["shared"].shape) != expected_shared or not bool(torch.isfinite(outputs["shared"]).all()):
        raise RuntimeError("ShareF 尺寸或数值有误")
    if hasattr(model, "shared_representation"):
        raise RuntimeError("编码器外的旧共享表示模块应已删除")
    branches = model(batch, student_mask=batch["availability"])
    for name in ("teacher", "student"):
        torch.testing.assert_close(branches[name]["shared"], branches[name]["encoder_shared"], rtol=0, atol=0)


@torch.no_grad()
def check_no_hidden_information(model: torch.nn.Module, batch: dict) -> None:
    model.eval()
    for index, name in enumerate(model.modalities):
        mask = batch["availability"].clone()
        mask[:, index] = False
        original = model.predict(batch, availability=mask)
        changed = dict(batch)
        changed["images"] = dict(batch["images"])
        changed["images"][name] = batch["images"][name] + 100.0
        perturbed = model.predict(changed, availability=mask)
        for key in ("logits", "shared", "encoder_shared"):
            torch.testing.assert_close(original[key], perturbed[key], rtol=0, atol=0)


def make_loaders(
    config: dict, *, train_samples: int, val_samples: int, test_samples: int,
    missing_samples: int,
) -> tuple[dict, dict, dict]:
    splits = split_by_event(
        scan_floodplanet(config["data"]["root"]), config["data"]["val_fraction"],
        config["data"]["test_fraction"], config["data"]["split_seed"],
    )
    events = {name: {record.event for record in records} for name, records in splits.items()}
    if any(events[a] & events[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("训练、验证和测试的洪水事件存在交叉")
    selected = {}
    for name, count in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        selected[name] = [record for record in splits[name] if record.is_complete][:count]
        if not selected[name]:
            raise ValueError(f"{name} 子集没有完整 S1/S2/PS 样本")
    missing = [record for record in splits["test"] if not record.is_complete][:missing_samples]
    selected["natural_test"] = selected["test"] + missing
    loaders = {}
    for name, records in selected.items():
        source = FloodPlanetDataset(
            records, config["sensors"], label_size=config["data"]["label_size"],
            # 完整子集已筛选；natural_test 则保留真实缺失样本。
            augment=name == "train",
            use_geo_metadata=config["data"]["use_geo_metadata"],
            alphaearth_config=config.get("alphaearth"),
        )
        loaders[name] = DataLoader(
            PipelineDataset(source), batch_size=1,
            shuffle=name == "train", num_workers=0,
        )
        print(f"{name}: {[record.sample_id for record in records]}", flush=True)
    manifest = {
        "split_events": {name: sorted(items) for name, items in events.items()},
        "selected": {name: [record.to_dict() for record in records] for name, records in selected.items()},
        "available_counts": {name: len(records) for name, records in splits.items()},
        "natural_missing_samples": len(missing),
    }
    return loaders, selected, manifest


@torch.no_grad()
def export_prediction(model: torch.nn.Module, loader: DataLoader, record, device: torch.device, root: Path) -> dict:
    model.eval()
    batch = move_to_device(next(iter(loader)), device)
    probability = torch.sigmoid(model.predict(batch)["logits"])[0, 0].cpu().numpy()
    shape = probability.shape
    with rasterio.open(record.reference_path) as reference:
        transform = reference.transform * Affine.scale(reference.width / shape[1], reference.height / shape[0])
        crs = reference.crs
    mask = (probability >= 0.5).astype("uint8")
    files = {
        "probability": root / "predictions" / f"{record.sample_id}_probability.tif",
        "mask": root / "predictions" / f"{record.sample_id}_mask.tif",
    }
    write_geotiff(record.reference_path, files["probability"], probability, "float32")
    write_geotiff(record.reference_path, files["mask"], mask, "uint8")
    for name, path in files.items():
        with rasterio.open(path) as saved:
            array = saved.read(1)
            if array.shape != shape or saved.transform != transform or saved.crs != crs:
                raise RuntimeError(f"{name} 预测影像的尺寸或地理信息有误")
            if not bool(np.isfinite(array).all()):
                raise RuntimeError(f"{name} 预测影像包含 NaN/Inf")
            expected = probability if name == "probability" else mask
            if not np.array_equal(array, expected):
                raise RuntimeError(f"{name} 预测影像写入后数值不一致")
    return {name: str(path) for name, path in files.items()}


def main() -> None:
    started_at = perf_counter()
    # 运行设置：修改这里后直接运行 main()，不需要填写运行参数。
    config_path = PROJECT / "configs" / "default.json"
    device_setting = "auto"  # auto、cpu 或 cuda。
    epochs = 1
    train_samples = 2
    val_samples = 1
    test_samples = 1
    missing_samples = 1
    amp_override = None  # None 沿用配置，False 关闭，True 显式启用（需要 CUDA）。
    output_dir = PROJECT / "runs" / "pipeline_test"

    if min(epochs, train_samples, val_samples, test_samples) < 1 or missing_samples < 0:
        raise ValueError("epochs 和训练/验证/测试样本数必须为正整数，missing_samples 必须非负")
    config = load_config(config_path)
    ps_image_size = config["model"].get("ps_encoder_size")
    if config["data"]["label_size"] != config["model"]["output_size"]:
        raise ValueError("label_size 与 output_size 必须一致")
    if not Path(config["model"]["clay_checkpoint"]).is_file():
        raise FileNotFoundError(f"Clay 权重不存在：{config['model']['clay_checkpoint']}")
    if device_setting not in ("auto", "cpu", "cuda"):
        raise ValueError("device_setting 必须是 auto、cpu 或 cuda")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if device_setting == "auto" else device_setting
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用 CUDA")
    if amp_override is True and device_name != "cuda":
        raise ValueError("AMP 需要 CUDA；CPU 请设置 amp_override=False")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    amp = bool((config["training"]["amp"] if amp_override is None else amp_override) and device.type == "cuda")
    config["model"]["freeze_clay"] = True
    config["training"].update(epochs=epochs, batch_size=1, amp=amp)
    config["data"]["num_workers"] = 0
    seed_everything(config["training"]["seed"])
    check_mixture_adapter()
    loaders, selected, manifest = make_loaders(
        config, train_samples=train_samples, val_samples=val_samples,
        test_samples=test_samples, missing_samples=missing_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run_", dir=output_dir.resolve()))
    config["training"]["output_dir"] = str(root)
    write_json(root / "config.json", config)
    write_json(root / "split_manifest.json", manifest)
    ps_clay_description = f"{ps_image_size}x{ps_image_size}" if ps_image_size else "原始尺寸"
    input_description = f"S1/S2=原始尺寸，PS Clay={ps_clay_description}，PS CNN=原始尺寸"
    print(f"device={device} Clay={config['model']['clay_model_size']} pretrained=True amp={amp} "
          f"epochs={epochs} input={input_description}\n输出目录：{root}", flush=True)
    print("加载真实预训练模型……", flush=True)
    model = build_model(config).to(device)
    clay_source_files = check_local_clay_source()
    print(f"Clay 源码：{clay_source_files['claymodel.model']}", flush=True)
    criterion = MultiLevelDistillationLoss(config["loss"]["weights"])
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp, init_scale=128.0)
    reference_batch = move_to_device(next(iter(loaders["val"])), device)
    reference_input_sizes = {name: list(image.shape[-2:]) for name, image in reference_batch["images"].items()}
    clay_input_sizes = {name: size.copy() for name, size in reference_input_sizes.items()}
    if ps_image_size is not None:
        clay_input_sizes["ps"] = [ps_image_size, ps_image_size]
    print("验证参考样本输入尺寸：" + ", ".join(
        f"{name.upper()}={height}x{width}" for name, (height, width) in reference_input_sizes.items()
    ), flush=True)
    print(f"PS Clay 输入：{clay_input_sizes['ps']}，"
          f"PS CNN 输入：{reference_input_sizes['ps']}，"
          f"CNN 已启用：{model.encoder.ps_detail is not None}", flush=True)
    check_encoder_outputs(model, reference_batch)
    history, best_iou, best_epoch, best_logits = [], -1.0, 0, None

    for epoch in range(1, epochs + 1):
        before = {name: getattr(model, name).segmentation_head.weight.detach().clone()
                  for name in ("teacher_decoder", "student_decoder")}
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_metrics = run_epoch(
            model, loaders["train"], criterion, device, scaler, optimizer=optimizer,
            gradient_clip=config["training"]["gradient_clip"],
        )
        check_values(train_loss, "训练损失")
        check_values(train_metrics, "训练指标", metrics=True)
        check_gradients(model)
        if any(torch.equal(weights, getattr(model, name).segmentation_head.weight) for name, weights in before.items()):
            raise RuntimeError("教师或学生分割头未更新，检查梯度或 AMP 缩放")
        versions = state_versions(model)
        val_loss, val_metrics = run_epoch(model, loaders["val"], criterion, device, scaler)
        check_values(val_loss, "验证损失")
        check_values(val_metrics, "验证指标", metrics=True)
        if versions != state_versions(model):
            raise RuntimeError("验证阶段改变了模型参数或缓冲区")
        scheduler.step()
        history.append({"epoch": epoch, "lr": learning_rate, "train_loss": train_loss,
                        "train_metrics": train_metrics, "val_loss": val_loss, "val_metrics": val_metrics})
        print(f"epoch={epoch} train_loss={train_loss['total']:.4f} val_loss={val_loss['total']:.4f} "
              f"val_iou={val_metrics['iou']:.4f}，保存权重……", flush=True)
        bound_manifest = {
            "train": manifest["selected"]["train"], "val": manifest["selected"]["val"],
            "test": manifest["selected"]["natural_test"],
        }
        save_checkpoint(root / "last.pt", model, optimizer, epoch, config, val_metrics, split_manifest=bound_manifest)
        if val_metrics["iou"] > best_iou:
            best_iou, best_epoch = val_metrics["iou"], epoch
            save_checkpoint(root / "best.pt", model, optimizer, epoch, config, val_metrics, split_manifest=bound_manifest)
            with torch.no_grad():
                best_logits = model.predict(reference_batch)["logits"].cpu()
        write_json(root / "history.json", history)

    # 释放训练模型后再重建，避免 large 编码器在显存里同时存在两份。
    del model, optimizer, scheduler, before
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("重建模型并加载验证集选出的 best.pt……", flush=True)
    model = build_model(config).to(device).eval()
    payload = load_checkpoint(root / "best.pt", model, "cpu")
    if payload["epoch"] != best_epoch or payload["metrics"]["iou"] != best_iou:
        raise RuntimeError("加载的模型与最佳验证轮次不一致")
    optimizer = make_optimizer(model, config)
    optimizer.load_state_dict(payload["optimizer"])
    if not optimizer.state:
        raise RuntimeError("优化器状态恢复失败")
    del payload, optimizer
    gc.collect()
    with torch.no_grad():
        restored_logits = model.predict(reference_batch)["logits"]
    torch.testing.assert_close(restored_logits.cpu(), best_logits, rtol=0, atol=0)
    check_no_hidden_information(model, reference_batch)
    print("PASS：四路编码输出、Adapter 梯度、教师/学生更新、权重恢复和遮挡信息无泄露。", flush=True)

    def reject_teacher(module, inputs):
        raise RuntimeError("推理阶段调用了教师分支")

    def check_output(module, inputs, output):
        expected = (1, 1, config["model"]["output_size"], config["model"]["output_size"])
        for key in ("logits", "boundary_logits"):
            value = output[key]
            if tuple(value.shape) != expected or not bool(torch.isfinite(value).all()) or value.requires_grad:
                raise RuntimeError(f"测试输出 {key} 尺寸、数值或梯度状态有误")

    teacher_hook = model.teacher_fusion.register_forward_pre_hook(reject_teacher)
    output_hook = model.student_decoder.register_forward_hook(check_output)
    versions = state_versions(model)
    results = {}
    print(f"\n{'scenario':<22} {'IoU':>9} {'F1':>9} {'Precision':>9} {'Recall':>9} {'Accuracy':>9}", flush=True)
    for name, pattern in {**SCENARIOS, "natural_missingness": None}.items():
        loader = loaders["natural_test"] if pattern is None else loaders["test"]
        results[name] = evaluate_loader(model, loader, device, pattern, amp)
        check_values(results[name], f"测试 {name}", metrics=True)
        print(f"{name:<22} " + " ".join(f"{results[name][metric]:9.4f}" for metric in METRIC_NAMES), flush=True)
    predictions = export_prediction(model, loaders["natural_test"], selected["natural_test"][0], device, root)
    if versions != state_versions(model):
        raise RuntimeError("测试或预测阶段改变了模型参数或缓冲区")
    teacher_hook.remove()
    output_hook.remove()
    # collect_gpu_memory 在 CUDA 上同步，确保计入尚未完成的 GPU 工作。
    gpu_memory = collect_gpu_memory(device)
    runtime_seconds = perf_counter() - started_at
    report = {
        "status": "PASS", "purpose": "小样本流程检查，指标不用于正式精度结论",
        "device": str(device), "amp": amp, "pretrained": True, "clay_model_size": config["model"]["clay_model_size"],
        "gpu_memory": gpu_memory,
        "runtime_seconds": runtime_seconds,
        "epochs": epochs, "ps_image_size": ps_image_size, "label_size": config["data"]["label_size"],
        "input_size_mode": "native_images_model_resizes_ps_for_clay",
        "reference_input_sizes": reference_input_sizes,
        "clay_input_sizes": clay_input_sizes,
        "ps_detail_enabled": model.encoder.ps_detail is not None,
        "encoder_architecture": "shared Clay + MixtureAdapter after every block with heterogeneous experts and spatial soft routing + direct ShareF",
        "clay_source_files": clay_source_files,
        "adapter_layers": list(model.encoder.adapter_layers),
        "checkpoint_epoch": best_epoch, "best_validation_iou": best_iou,
        "complete_test_samples": len(selected["test"]), "natural_test_samples": len(selected["natural_test"]),
        "natural_missing_samples": manifest["natural_missing_samples"],
        "natural_missingness_note": "使用实际可用模态；没有真实缺失样本时仅包含完整样本",
        "samples": {name: [record.sample_id for record in records] for name, records in selected.items()},
        "history": history, "metrics": results, "predictions": predictions,
        "checks": {name: True for name in (
            "event_disjoint_splits", "teacher_student_updated", "all_trainable_module_gradients",
            "validation_no_updates", "checkpoint_optimizer_restored", "restored_prediction_identical",
            "all_modality_scenarios", "student_only_inference", "test_no_updates", "geotiff_round_trip",
            "encoder_four_outputs", "mixture_adapter_gradients", "hidden_modality_no_leakage",
            "mixture_adapter_routing_and_masks",
            "adapter_after_every_block", "direct_encoder_shared_feature",
            "project_local_clay_source",
        )},
    }
    write_json(root / "evaluation.json", report)
    print(f"\nPASS：训练 → 验证 → 保存/恢复 → 独立测试集八种评估 → GeoTIFF 预测。\n"
          f"评价报告：{root / 'evaluation.json'}\n预测目录：{root / 'predictions'}", flush=True)
    print(f"总运行时长：{runtime_seconds:.2f} 秒（{runtime_seconds / 60:.2f} 分钟）", flush=True)
    if gpu_memory is None:
        print("显存统计：CPU 模式，未使用 CUDA 显存。", flush=True)
    else:
        print(
            f"\n显存统计（{gpu_memory['device_name']}，单位 MiB）：\n"
            f"整个流程峰值：张量占用 {gpu_memory['peak_allocated_mib']:.2f}，"
            f"PyTorch 保留（含缓存）{gpu_memory['peak_reserved_mib']:.2f}\n"
            f"结束时占用：张量占用 {gpu_memory['allocated_mib']:.2f}，"
            f"PyTorch 保留（含缓存）{gpu_memory['reserved_mib']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
