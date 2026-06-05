#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a single-head SCN on UCI HAR Sigma-Delta event tensors.

Example:
    python train_ucihar.py --epochs 2

Outputs are written to:
    runs/<run>_ucihar_<YYYYMMDD>_<HHMMSS>[_<N>]/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from train.utils import (
    count_parameters,
    extract_single_logits,
    fit_model,
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
)

import numpy as np
import torch
import torch.nn.functional as F

from loaders.ucihar_event_loader import (
    ACTIVITY_LABELS,
    UCIHAREventLoaderConfig,
    estimate_class_weights,
    load_or_build_ucihar_event_splits,
    make_event_loaders,
    named_class_counts,
)

DEFAULT_THETA_UCIHAR = [
    0.151215,
    0.297011,
    0.135961,
    0.091156,
    0.067695,
    0.066330,
]

# CLI
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one UCI HAR single-head SCN from cached Sigma-Delta event tensors."
    )

    # Data/cache
    p.add_argument("--data-root", default="data", help="Project data root containing the UCI HAR Dataset folder.")
    p.add_argument("--dataset-dir", default="UCI HAR Dataset", help="UCI HAR dataset folder name inside data-root.")
    p.add_argument("--cache-dir", default="data/ucihar_cache", help="Folder where event-loader .npy cache is stored.")
    p.add_argument("--cache-name", default="default", help="Cache name suffix used by ucihar_event_loader.")
    p.add_argument("--refresh-cache", action="store_true", help="Force rebuild of the UCI HAR event cache.")
    p.add_argument("--case", default="random", choices=["official", "random", "subject", "subject_large"], help="Split mode used by preprocess_ucihar.")
    p.add_argument("--target-domain", type=int, default=0, help="Held-out subject/domain for subject-based cases.")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"), help="Random split fractions.")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio used when splitting official train data.")

    # Event encoding
    p.add_argument("--ups", type=float, default=5.0, help="Upsampling factor for each UCI HAR window.")
    p.add_argument("--out-win", type=int, default=None, help="Explicit interpolated length. If omitted, derived from UPS.")
    p.add_argument("--dead-zone", type=float, default=0.5, help="Sigma-Delta dead-zone factor.")
    p.add_argument("--theta", type=float, nargs=6, default=DEFAULT_THETA_UCIHAR, help="Six-channel Sigma-Delta theta vector.")
    p.add_argument("--interp-method", default="linear", choices=["linear", "zoh", "cubic"], help="Interpolation method used before Sigma-Delta encoding.")
    p.add_argument("--sd-init", default="x0", help="Sigma-Delta reconstruction initialization mode.")

    # Run/execution
    p.add_argument("--out-dir", default="runs", help="Root folder for training outputs.")
    p.add_argument("--run", default="ucihar_scn", help="Run name inside out-dir.")
    p.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
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
    p.add_argument("--class-weight", default="balanced", choices=["balanced", "none"], help="Cross-entropy class weighting.")
    p.add_argument("--balanced-sampler", action="store_true", help="Use a WeightedRandomSampler on the training split.")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Cross-entropy label smoothing.")

    # LIF/model
    p.add_argument("--tau", type=float, default=0.75, help="LIF membrane decay.")
    p.add_argument("--thresh", type=float, default=0.5, help="LIF firing threshold.")
    p.add_argument("--hard-reset", action="store_true", help="Use hard reset instead of soft reset.")
    p.add_argument("--n-ch", type=int, default=6, help="Number of IMU channels.")
    p.add_argument("--n-classes", type=int, default=6, help="Number of UCI HAR classes.")
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
    p.add_argument("--spiking-heads", action="store_true", help="Use spiking readout heads instead of dense heads.")

    return p.parse_args()


# Model

def build_model(cfg: argparse.Namespace, dev: torch.device) -> torch.nn.Module:
    from models.models import SingleHeadSCN

    return SingleHeadSCN(n_classes=cfg.n_classes, **scn_model_kwargs(cfg)).to(dev)


# Data
def build_data(cfg: argparse.Namespace, dev: torch.device) -> dict[str, Any]:
    cache_cfg = UCIHAREventLoaderConfig(
        data_root=cfg.data_root,
        dataset_dir=cfg.dataset_dir,
        case=cfg.case,
        target_domain=cfg.target_domain,
        split=tuple(float(x) for x in cfg.split),
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        raw_win=128,
        ups=cfg.ups,
        out_win=cfg.out_win,
        interp_method=cfg.interp_method,
        theta_sd=tuple(float(x) for x in cfg.theta),
        dead_zone=cfg.dead_zone,
        sd_init=cfg.sd_init,
        n_ch=cfg.n_ch,
        cache_dir=cfg.cache_dir,
        run_name=cfg.cache_name,
        batch_size=256,
        num_workers=0,
        pin_memory=False,
    )

    splits, metadata = load_or_build_ucihar_event_splits(
        data_root=cfg.data_root,
        dataset_dir=cfg.dataset_dir,
        cache_dir=cfg.cache_dir,
        run_name=cfg.cache_name,
        cfg=cache_cfg,
        refresh=cfg.refresh_cache,
        mmap_mode="r",
        verbose=not cfg.quiet_loader,
    )

    pin = bool((not cfg.no_pin_memory) and dev.type == "cuda")
    loaders = make_event_loaders(
        splits,
        cfg=cache_cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=pin,
        balanced_sampler=cfg.balanced_sampler,
    )

    return {
        "splits": splits,
        "metadata": metadata,
        "loaders": loaders,
        "class_weights": estimate_class_weights(splits["train"], n_classes=cfg.n_classes),
    }


# Training/evaluation
def unpack_batch(batch: dict[str, torch.Tensor], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(dev, non_blocking=True).float()
    offset = batch["offset"].to(dev, non_blocking=True).float()
    y = batch["y"].to(dev, non_blocking=True).long()
    return x, offset, y


def run_epoch(
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
            x, offset, y = unpack_batch(batch, dev)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with make_autocast(amp_enabled):
                logits = extract_single_logits(model(x, offset=offset))
                loss = F.cross_entropy(
                    logits,
                    y,
                    weight=class_weight,
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

    out = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "n": int(total),
    }
    if collect_predictions:
        out["y_true"] = torch.cat(y_true_chunks).numpy() if y_true_chunks else np.array([], dtype=np.int64)
        out["y_pred"] = torch.cat(y_pred_chunks).numpy() if y_pred_chunks else np.array([], dtype=np.int64)
    return out



# Main
def main() -> None:
    cfg = get_args()
    set_seed(cfg.seed)
    dev = get_device(cfg.gpu)

    dataset_tag = "ucihar"
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
            ("dataset", "UCI HAR"),
            ("device", dev),
            ("run_dir", run_dir),
            ("data_root", cfg.data_root),
            ("cache", Path(cfg.cache_dir) / f"ucihar_event_loader_{cfg.cache_name}"),
            ("ups_dead_zone", f"{cfg.ups}/{cfg.dead_zone}"),
            ("theta", np.asarray(cfg.theta, dtype=np.float32)),
        ],
    )

    data = build_data(cfg, dev)
    loaders = data["loaders"]
    splits = data["splits"]

    print("\n[data]")
    for name in ("train", "val", "test"):
        y = np.asarray(splits[name]["y"], dtype=np.int64)
        x_shape = tuple(np.asarray(splits[name]["x_ev"]).shape)
        off_shape = tuple(np.asarray(splits[name]["offset"]).shape)
        print(f"  {name:<5}: x_ev={x_shape} | offset={off_shape} | y={y.shape} | counts={named_class_counts(y)}")

    save_json(run_dir / "dataset_metadata.json", data["metadata"])

    if cfg.class_weight == "balanced":
        class_weight = torch.as_tensor(data["class_weights"], dtype=torch.float32, device=dev)
    else:
        class_weight = None
    print()
    print_kv_block("loss", [("class_weight", "none" if class_weight is None else class_weight.detach().cpu().numpy())])

    model = build_model(cfg, dev)
    n_params, n_trainable = count_parameters(model)
    print()
    print_kv_block(
        "model",
        [
            ("name", "SingleHeadSCN"),
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
        format_epoch_fn=format_single_epoch_line,
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
        optimizer=None,
        scaler=None,
        class_weight=class_weight,
        collect_predictions=True,
    )
    y_true = test_metrics.pop("y_true")
    y_pred = test_metrics.pop("y_pred")

    test_metrics.update(
        {
            "best_val_loss": float(best_val),
            "best_epoch": int(ckpt.get("epoch", -1)),
            "total_time_sec": float(time.time() - fit_result["started_at"]),
            "class_counts_test": named_class_counts(y_true),
        }
    )

    save_json(metrics_dir / "test_metrics.json", test_metrics)
    labels = list(range(cfg.n_classes))
    names = [ACTIVITY_LABELS.get(i, str(i)) for i in labels]
    save_classification_outputs(metrics_dir, y_true=y_true, y_pred=y_pred, labels=labels, names=names)

    print("\n[test]")
    print(json.dumps(json_ready(test_metrics), indent=2, ensure_ascii=False))
    print(f"[done] saved_run={run_dir}")


if __name__ == "__main__":
    main()
