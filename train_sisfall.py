#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a SisFall SCN on Sigma-Delta event tensors.

Example:
    python train_sisfall.py --epochs 2

Outputs are written to:
    runs/<run>_sisfall_<YYYYMMDD>_<HHMMSS>[_<N>]/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from train.utils import (
    count_parameters,
    extract_dual_outputs,
    extract_single_logits,
    fit_model,
    format_dual_epoch_line,
    format_single_epoch_line,
    get_device,
    json_ready,
    make_autocast,
    make_grad_scaler,
    print_kv_block,
    save_classification_outputs,
    save_json,
    scn_model_kwargs,
    set_seed,
    squeeze_binary_logits,
)

import numpy as np
import torch

from loaders.sisfall_event_loader import (
    EventLoaderConfig,
    H1_STATIC,
    estimate_head2_pos_weight,
    load_or_build_sisfall_event_splits,
    make_event_loaders,
    make_single_head_event_loaders,
    summarize_split,
)
from loaders.utils import balanced_class_weights
from models.losses import dual_head_losses, weighted_cross_entropy_loss
from models.models import DualHeadSCN, SingleHeadSCN
from preprocess.utils import named_class_counts

DEFAULT_THETA_SISFALL = list(EventLoaderConfig().theta_sd)
DUAL_HEAD1_NAMES = {0: "Fall", 1: "Dynamic", 2: "Static"}
DUAL_HEAD2_NAMES = {0: "Stable Posture", 1: "Postural Transition"}
SINGLE_HEAD_NAMES = {0: "Fall", 1: "Dynamic", 2: "Stable Posture", 3: "Postural Transition"}


# CLI
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one SisFall single-head or dual-head SCN from cached Sigma-Delta event tensors."
    )

    # Data/cache
    p.add_argument("--dataset-root", default="data/SisFall_dataset", help="Root directory containing SisFall TXT files.")
    p.add_argument("--label-dir", default="data/labels_transitions", help="Folder with manual transition labels.")
    p.add_argument("--result-dir", default="data/sisfall_tagging_results", help="Folder for posture-tagging outputs.")
    p.add_argument("--cache-dir", default="data/sisfall_cache", help="Folder where event-loader .npy cache is stored.")
    p.add_argument("--cache-name", default="default", help="Cache name suffix used by sisfall_event_loader.")
    p.add_argument("--refresh-cache", action="store_true", help="Force rebuild of the SisFall event cache.")
    p.add_argument("--refresh-posture", action="store_true", help="Force rebuild of posture-refinement labels.")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"), help="File-level split fractions.")

    # Event encoding
    p.add_argument("--raw-win", type=int, default=410, help="Raw SisFall window length.")
    p.add_argument("--out-win", type=int, default=2048, help="Interpolated window length before Sigma-Delta encoding.")
    p.add_argument("--theta", type=float, nargs=6, default=DEFAULT_THETA_SISFALL, help="Six-channel Sigma-Delta theta vector.")
    p.add_argument("--dead-zone", type=float, default=0.5, help="Sigma-Delta dead-zone factor.")
    p.add_argument("--sd-init", default="x0", help="Sigma-Delta reconstruction initialization mode.")
    p.add_argument("--stride-fall", type=int, default=160, help="Sliding-window stride for fall recordings.")
    p.add_argument("--stride-dynamic", type=int, default=410, help="Sliding-window stride for dynamic ADL recordings.")
    p.add_argument("--stride-static", type=int, default=205, help="Sliding-window stride for static ADL recordings.")
    p.add_argument("--fall-policy", default="contain_global_max", choices=["contain_global_max", "all"], help="Fall window selection policy.")
    p.add_argument("--acc", default="ADXL345", choices=["ADXL345", "MMA8451Q"], help="Accelerometer source used from each SisFall sample.")
    p.add_argument("--dtype", default="int32", help="Numeric dtype used when loading raw SisFall values.")

    # Run/execution
    p.add_argument("--head-mode", default="dual", choices=["dual", "single"], help="Training target layout.")
    p.add_argument("--out-dir", default="runs", help="Root folder for training outputs.")
    p.add_argument("--run", default="sisfall_scn", help="Run name inside out-dir.")
    p.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    p.add_argument("--workers", type=int, default=4, help="PyTorch DataLoader workers.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    p.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    p.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoaders.")
    p.add_argument("--quiet-loader", action="store_true", help="Reduce loader/cache logging.")

    # Optimizer/scheduler
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--wd", type=float, default=1e-4, help="Adam weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables it.")
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    p.add_argument("--plateau-patience", type=int, default=5, help="ReduceLROnPlateau patience.")
    p.add_argument("--lr-factor", type=float, default=0.5, help="Learning-rate reduction factor.")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate.")

    # Loss/sampling
    p.add_argument("--lambda-h1", type=float, default=1.0, help="Weight for dual Head 1 loss.")
    p.add_argument("--lambda-h2", type=float, default=0.5, help="Weight for dual Head 2 loss.")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Cross-entropy label smoothing.")
    p.add_argument("--class-weight", default="none", choices=["balanced", "none"], help="Cross-entropy class weighting.")
    p.add_argument("--use-ambiguous", action="store_true", help="Use ambiguous static windows in the dual Head 2 loss.")
    p.add_argument("--ambiguous-weight", type=float, default=0.0, help="Sample weight for ambiguous static windows.")
    p.add_argument("--manual-weight", type=float, default=1.0, help="Sample weight for manual posture labels.")
    p.add_argument("--autotag-weight", type=float, default=0.4, help="Sample weight for automatic posture labels.")
    p.add_argument("--single-ambiguous", default="drop", choices=["drop", "stable", "transition"], help="How single-head training handles ambiguous static windows.")

    # LIF/model
    p.add_argument("--tau", type=float, default=0.75, help="LIF membrane decay.")
    p.add_argument("--thresh", type=float, default=0.5, help="LIF firing threshold.")
    p.add_argument("--hard-reset", action="store_true", help="Use hard reset instead of soft reset.")
    p.add_argument("--n-ch", type=int, default=6, help="Number of IMU channels.")
    p.add_argument("--conv-ch", type=int, nargs=3, default=[32, 64, 64], help="Channels in the three conv blocks.")
    p.add_argument("--kernels", type=int, nargs=3, default=[32, 32, 8], help="Conv1D kernel sizes.")
    p.add_argument("--strides", type=int, nargs=3, default=[4, 2, 1], help="Conv1D strides.")
    p.add_argument("--pool-kernels", type=int, nargs=3, default=[2, 2, 2], help="MaxPool1D kernel sizes.")
    p.add_argument("--pool-strides", type=int, nargs=3, default=[2, 2, 2], help="MaxPool1D strides.")
    p.add_argument("--pool-paddings", type=int, nargs=3, default=[0, 0, 0], help="MaxPool1D paddings.")
    p.add_argument("--p-drop", type=float, default=0.35, help="Dropout in the event branch.")
    p.add_argument("--merge-polarities", action="store_true", help="Merge positive and negative polarities before the first conv.")
    p.add_argument("--offset-hidden", type=int, default=8, help="Hidden units in the offset LIF branch.")
    p.add_argument("--head-rate-scale", type=float, default=8.0, help="Rate scaling for spiking heads.")
    p.add_argument("--spiking-heads", action=argparse.BooleanOptionalAction, default=False, help="Use spiking readout heads instead of dense heads.")

    return p.parse_args()



def make_loader_cfg(cfg: argparse.Namespace) -> EventLoaderConfig:
    return EventLoaderConfig(
        seed=cfg.seed,
        split=tuple(float(x) for x in cfg.split),
        raw_win=cfg.raw_win,
        out_win=cfg.out_win,
        n_ch=cfg.n_ch,
        stride_fall=cfg.stride_fall,
        stride_dynamic=cfg.stride_dynamic,
        stride_static=cfg.stride_static,
        fall_policy=cfg.fall_policy,
        theta_sd=tuple(float(x) for x in cfg.theta),
        dead_zone=cfg.dead_zone,
        sd_init=cfg.sd_init,
        manual_weight=cfg.manual_weight,
        autotag_weight=cfg.autotag_weight,
        ambiguous_weight=cfg.ambiguous_weight,
        ignore_ambiguous=not cfg.use_ambiguous,
        acc=cfg.acc,
        dtype=cfg.dtype,
        batch_size=512,
        num_workers=0,
        pin_memory=False,
    )

def build_data(cfg: argparse.Namespace, dev: torch.device) -> dict[str, Any]:
    loader_cfg = make_loader_cfg(cfg)
    splits, metadata = load_or_build_sisfall_event_splits(
        dataset_root=cfg.dataset_root,
        label_dir=cfg.label_dir,
        cache_dir=cfg.cache_dir,
        result_dir=cfg.result_dir,
        run_name=cfg.cache_name,
        cfg=loader_cfg,
        refresh=cfg.refresh_cache,
        refresh_posture=cfg.refresh_posture,
        mmap_mode="r",
        verbose=not cfg.quiet_loader,
    )

    pin = bool((not cfg.no_pin_memory) and dev.type == "cuda")
    out: dict[str, Any] = {
        "splits": splits,
        "metadata": metadata,
        "loader_cfg": loader_cfg,
        "pos_weight_h2": estimate_head2_pos_weight(splits["train"]),
    }

    if cfg.head_mode == "dual":
        out["loaders"] = make_event_loaders(splits,cfg=loader_cfg,batch_size=cfg.batch_size,num_workers=cfg.workers,pin_memory=pin,)
        out["datasets"] = None
    else:
        out["loaders"], out["datasets"] = make_single_head_event_loaders(splits,batch_size=cfg.batch_size,num_workers=cfg.workers,pin_memory=pin,ambiguous_policy=cfg.single_ambiguous,ambiguous_weight=cfg.ambiguous_weight,)

    return out


# Models
def build_model(cfg: argparse.Namespace, dev: torch.device) -> torch.nn.Module:
    kwargs = scn_model_kwargs(cfg)
    if cfg.head_mode == "dual":
        return DualHeadSCN(n_classes_head1=3, **kwargs).to(dev)
    return SingleHeadSCN(n_classes=4, **kwargs).to(dev)


# Training/evaluation
def unpack_dual_batch(batch: dict[str, torch.Tensor], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(dev, non_blocking=True).float()
    offset = batch["offset"].to(dev, non_blocking=True).float()
    y_h1 = batch["y_h1"].to(dev, non_blocking=True).long()
    y_h2 = batch["y_h2"].to(dev, non_blocking=True).float()
    src_h2 = batch["src_h2"].to(dev, non_blocking=True)
    w_h2 = batch["w_h2"].to(dev, non_blocking=True).float()
    return x, offset, y_h1, y_h2, src_h2, w_h2


def unpack_single_batch(batch: dict[str, torch.Tensor], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(dev, non_blocking=True).float()
    offset = batch["offset"].to(dev, non_blocking=True).float()
    y = batch["y"].to(dev, non_blocking=True).long()
    sample_weight = batch["sample_weight"].to(dev, non_blocking=True).float()
    return x, offset, y, sample_weight

def dual_head_metrics(outputs: dict[str, torch.Tensor], y_h1: torch.Tensor, y_h2: torch.Tensor, w_h2: torch.Tensor) -> dict[str, Any]:
    pred_h1 = outputs["head1"].argmax(dim=1)
    logits_h2 = squeeze_binary_logits(outputs["head2"])
    valid_h2 = (y_h1 == H1_STATIC) & (~torch.isclose(y_h2, torch.tensor(0.5, device=y_h2.device, dtype=y_h2.dtype))) & (w_h2 > 0)

    n = int(y_h1.shape[0])
    n_h2 = int(valid_h2.sum().item())
    h2_correct = 0
    if n_h2 > 0:
        pred_h2 = (torch.sigmoid(logits_h2) >= 0.5).float()
        h2_correct = int((pred_h2[valid_h2] == y_h2[valid_h2]).sum().item())

    return {
        "n": n,
        "h1_correct": int((pred_h1 == y_h1).sum().item()),
        "h2_correct": h2_correct,
        "n_h2": n_h2,
    }


def run_epoch_dual(
    model: torch.nn.Module,
    loader: Any,
    dev: torch.device,
    cfg: argparse.Namespace,
    pos_weight_h2: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    class_weight: torch.Tensor | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    amp_enabled = bool(cfg.amp and dev.type == "cuda")

    loss_sum = loss_h1_sum = loss_h2_sum = 0.0
    total = h1_correct = h2_correct = h2_total = 0
    y_true_h1: list[torch.Tensor] = []
    y_pred_h1: list[torch.Tensor] = []
    y_true_h2: list[torch.Tensor] = []
    y_pred_h2: list[torch.Tensor] = []

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            x, offset, y_h1, y_h2, src_h2, w_h2 = unpack_dual_batch(batch, dev)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with make_autocast(amp_enabled):
                outputs = extract_dual_outputs(model(x, offset=offset))
                losses = dual_head_losses(
                    outputs=outputs,
                    y_head1=y_h1,
                    y_head2=y_h2,
                    lambda_head1=cfg.lambda_h1,
                    lambda_head2=cfg.lambda_h2,
                    head1_kwargs={"class_weight": class_weight, "label_smoothing": cfg.label_smoothing},
                    head2_kwargs={
                        "y_head1": y_h1,
                        "static_class_idx": H1_STATIC,
                        "mask_only_static": True,
                        "ignore_ambiguous": not cfg.use_ambiguous,
                        "sample_weight": w_h2,
                        "label_source": src_h2,
                        "gold_weight": cfg.manual_weight,
                        "silver_weight": cfg.autotag_weight,
                        "pos_weight": pos_weight_h2,
                    },
                )
                loss = losses["loss"]

            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            bs = int(y_h1.shape[0])
            m = dual_head_metrics(outputs, y_h1, y_h2, w_h2)
            total += bs
            h1_correct += m["h1_correct"]
            h2_correct += m["h2_correct"]
            h2_total += m["n_h2"]
            loss_sum += float(loss.detach().item()) * bs
            loss_h1_sum += float(losses["loss_head1"].detach().item()) * bs
            loss_h2_sum += float(losses["loss_head2"].detach().item()) * bs

            if collect_predictions:
                pred_h1 = outputs["head1"].argmax(dim=1)
                y_true_h1.append(y_h1.detach().cpu())
                y_pred_h1.append(pred_h1.detach().cpu())

                logits_h2 = squeeze_binary_logits(outputs["head2"])
                valid_h2 = (y_h1 == H1_STATIC) & (~torch.isclose(y_h2, torch.tensor(0.5, device=y_h2.device, dtype=y_h2.dtype))) & (w_h2 > 0)
                if bool(valid_h2.any()):
                    pred_h2 = (torch.sigmoid(logits_h2[valid_h2]) >= 0.5).long()
                    y_true_h2.append(y_h2[valid_h2].long().detach().cpu())
                    y_pred_h2.append(pred_h2.detach().cpu())

    out: dict[str, Any] = {
        "loss": loss_sum / max(total, 1),
        "loss_h1": loss_h1_sum / max(total, 1),
        "loss_h2": loss_h2_sum / max(total, 1),
        "acc_h1": h1_correct / max(total, 1),
        "acc_h2": h2_correct / max(h2_total, 1) if h2_total > 0 else float("nan"),
        "n": int(total),
        "n_h2": int(h2_total),
    }
    if collect_predictions:
        out["y_true_h1"] = torch.cat(y_true_h1).numpy() if y_true_h1 else np.array([], dtype=np.int64)
        out["y_pred_h1"] = torch.cat(y_pred_h1).numpy() if y_pred_h1 else np.array([], dtype=np.int64)
        out["y_true_h2"] = torch.cat(y_true_h2).numpy() if y_true_h2 else np.array([], dtype=np.int64)
        out["y_pred_h2"] = torch.cat(y_pred_h2).numpy() if y_pred_h2 else np.array([], dtype=np.int64)
    return out


def run_epoch_single(
    model: torch.nn.Module,
    loader: Any,
    dev: torch.device,
    cfg: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    class_weight: torch.Tensor | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    amp_enabled = bool(cfg.amp and dev.type == "cuda")

    loss_sum = 0.0
    correct = 0
    total = 0
    y_true_chunks: list[torch.Tensor] = []
    y_pred_chunks: list[torch.Tensor] = []

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            x, offset, y, sample_weight = unpack_single_batch(batch, dev)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with make_autocast(amp_enabled):
                logits = extract_single_logits(model(x, offset=offset))
                loss = weighted_cross_entropy_loss(
                    logits,
                    y,
                    sample_weight=sample_weight,
                    class_weight=class_weight,
                    label_smoothing=cfg.label_smoothing,
                )

            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            bs = int(y.shape[0])
            pred = logits.argmax(dim=1)
            total += bs
            loss_sum += float(loss.detach().item()) * bs
            correct += int((pred == y).sum().item())

            if collect_predictions:
                y_true_chunks.append(y.detach().cpu())
                y_pred_chunks.append(pred.detach().cpu())

    out: dict[str, Any] = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "n": int(total),
    }
    if collect_predictions:
        out["y_true"] = torch.cat(y_true_chunks).numpy() if y_true_chunks else np.array([], dtype=np.int64)
        out["y_pred"] = torch.cat(y_pred_chunks).numpy() if y_pred_chunks else np.array([], dtype=np.int64)
    return out


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    dev: torch.device,
    cfg: argparse.Namespace,
    pos_weight_h2: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    class_weight: torch.Tensor | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    if cfg.head_mode == "dual":
        return run_epoch_dual(
            model,
            loader,
            dev,
            cfg,
            pos_weight_h2=pos_weight_h2,
            optimizer=optimizer,
            scaler=scaler,
            class_weight=class_weight,
            collect_predictions=collect_predictions,
        )
    return run_epoch_single(
        model,
        loader,
        dev,
        cfg,
        optimizer=optimizer,
        scaler=scaler,
        class_weight=class_weight,
        collect_predictions=collect_predictions,
    )


def estimate_class_weight(cfg: argparse.Namespace, data: dict[str, Any], dev: torch.device) -> torch.Tensor | None:
    if cfg.class_weight == "none":
        return None

    if cfg.head_mode == "dual":
        y = np.asarray(data["splits"]["train"]["y_h1"], dtype=np.int64)
        n_classes = 3
    else:
        y = data["datasets"]["train"].y.numpy()
        n_classes = 4

    weights = balanced_class_weights(y, n_classes=n_classes)
    return torch.as_tensor(weights, dtype=torch.float32, device=dev)


def save_test_outputs(cfg: argparse.Namespace, metrics_dir: Path, test_metrics: dict[str, Any]) -> dict[str, Any]:
    if cfg.head_mode == "dual":
        y_true_h1 = test_metrics.pop("y_true_h1")
        y_pred_h1 = test_metrics.pop("y_pred_h1")
        y_true_h2 = test_metrics.pop("y_true_h2")
        y_pred_h2 = test_metrics.pop("y_pred_h2")

        save_classification_outputs(
            metrics_dir,
            prefix="h1",
            y_true=y_true_h1,
            y_pred=y_pred_h1,
            labels=[0, 1, 2],
            names=[DUAL_HEAD1_NAMES[i] for i in range(3)],
        )
        if y_true_h2.size > 0:
            save_classification_outputs(
                metrics_dir,
                prefix="h2",
                y_true=y_true_h2,
                y_pred=y_pred_h2,
                labels=[0, 1],
                names=[DUAL_HEAD2_NAMES[i] for i in range(2)],
            )
        test_metrics["class_counts_h1_test"] = named_class_counts(y_true_h1, DUAL_HEAD1_NAMES)
        test_metrics["class_counts_h2_test"] = named_class_counts(y_true_h2, DUAL_HEAD2_NAMES) if y_true_h2.size > 0 else {}
        return test_metrics

    y_true = test_metrics.pop("y_true")
    y_pred = test_metrics.pop("y_pred")
    save_classification_outputs(
        metrics_dir,
        prefix="single",
        y_true=y_true,
        y_pred=y_pred,
        labels=[0, 1, 2, 3],
        names=[SINGLE_HEAD_NAMES[i] for i in range(4)],
    )
    test_metrics["class_counts_test"] = named_class_counts(y_true, SINGLE_HEAD_NAMES)
    return test_metrics


# Main
def main() -> None:
    cfg = get_args()
    set_seed(cfg.seed)
    dev = get_device(cfg.gpu)

    dataset_tag = "sisfall"
    requested_run = cfg.run
    run_stem = requested_run if dataset_tag in requested_run.lower() else f"{requested_run}_{dataset_tag}"
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    base_run_dir = Path(cfg.out_dir) / f"{run_stem}_{run_stamp}"
    run_dir = base_run_dir
    suffix = 0
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            run_dir = Path(f"{base_run_dir}_{suffix}")

    cfg.requested_run = requested_run
    cfg.run = run_dir.name
    ckpt_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    ckpt_path = ckpt_dir / "best_model.pt"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", cfg)

    print_kv_block(
        "run",
        [
            ("dataset", "SisFall"),
            ("device", dev),
            ("run_dir", run_dir),
            ("head_mode", cfg.head_mode),
            ("spiking_heads", cfg.spiking_heads),
            ("dataset_root", cfg.dataset_root),
            ("cache", Path(cfg.cache_dir) / f"sisfall_event_loader_{cfg.cache_name}"),
            ("theta", np.asarray(cfg.theta, dtype=np.float32)),
        ],
    )

    data = build_data(cfg, dev)
    loaders = data["loaders"]
    splits = data["splits"]
    pos_weight_h2 = float(data["pos_weight_h2"])

    print("\n[data]")
    for name in ("train", "val", "test"):
        summary = summarize_split(splits[name])
        if cfg.head_mode == "single":
            single_y = data["datasets"][name].y.numpy()
            print(f"  {name:<5}: {json.dumps(json_ready(summary), ensure_ascii=False)} | single={named_class_counts(single_y, SINGLE_HEAD_NAMES)}")
        else:
            print(f"  {name:<5}: {json.dumps(json_ready(summary), ensure_ascii=False)}")
    save_json(run_dir / "dataset_metadata.json", data["metadata"])

    class_weight = estimate_class_weight(cfg, data, dev)
    loss_items = [("class_weight", "none" if class_weight is None else class_weight.detach().cpu().numpy())]
    if cfg.head_mode == "dual":
        loss_items.append(("pos_weight_h2", f"{pos_weight_h2:.4f}"))
    print()
    print_kv_block("loss", loss_items)

    model = build_model(cfg, dev)
    n_params, n_trainable = count_parameters(model)
    print()
    print_kv_block(
        "model",
        [
            ("name", "DualHeadSCN" if cfg.head_mode == "dual" else "SingleHeadSCN"),
            ("params", f"{n_params:,}"),
            ("trainable", f"{n_trainable:,}"),
            ("conv_ch", tuple(cfg.conv_ch)),
            ("kernels", tuple(cfg.kernels)),
            ("strides", tuple(cfg.strides)),
            ("tau_thresh", f"{cfg.tau}/{cfg.thresh}"),
        ],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.plateau_patience,
        min_lr=cfg.min_lr,
    )
    scaler = make_grad_scaler(enabled=bool(cfg.amp and dev.type == "cuda"))

    def epoch_runner(loader, optimizer=None, scaler=None, collect_predictions: bool = False):
        return run_epoch(
            model,
            loader,
            dev,
            cfg,
            pos_weight_h2=pos_weight_h2,
            optimizer=optimizer,
            scaler=scaler,
            class_weight=class_weight,
            collect_predictions=collect_predictions,
        )

    def checkpoint_extra():
        return {"class_weight": None if class_weight is None else class_weight.detach().cpu()}

    fit_result = fit_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        cfg=cfg,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metrics_dir=metrics_dir,
        ckpt_path=ckpt_path,
        run_epoch_fn=epoch_runner,
        metric_name="loss",
        format_epoch_fn=format_dual_epoch_line if cfg.head_mode == "dual" else format_single_epoch_line,
        checkpoint_extra_fn=checkpoint_extra,
    )
    best_val = float(fit_result["best_val"])

    if not ckpt_path.is_file():
        raise RuntimeError("No checkpoint was saved. Check training/validation data and loss values.")

    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = run_epoch(
        model,
        loaders["test"],
        dev,
        cfg,
        pos_weight_h2=pos_weight_h2,
        optimizer=None,
        scaler=None,
        class_weight=class_weight,
        collect_predictions=True,
    )
    test_metrics.update(
        {
            "best_val_loss": float(best_val),
            "best_epoch": int(ckpt.get("epoch", -1)),
            "total_time_sec": float(time.time() - fit_result["started_at"]),
        }
    )
    test_metrics = save_test_outputs(cfg, metrics_dir, test_metrics)

    save_json(metrics_dir / "test_metrics.json", test_metrics)

    print("\n[test]")
    print(json.dumps(json_ready(test_metrics), indent=2, ensure_ascii=False))
    print(f"[done] saved_run={run_dir}")


if __name__ == "__main__":
    main()
