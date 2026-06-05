#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared training helpers for event-based HAR scripts."""

from __future__ import annotations

import os

# Set thread limits before importing torch/numpy-heavy modules.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from preprocess.utils import json_ready, save_json


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch RNGs."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def get_device(gpu: int) -> torch.device:
    """Return the selected CUDA device, or CPU when CUDA is unavailable."""
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{int(gpu)}")
        torch.cuda.set_device(dev)
        return dev
    return torch.device("cpu")


def format_seconds(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = int(max(0, round(float(seconds))))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_autocast(enabled: bool):
    """Create a CUDA autocast context when mixed precision is enabled."""
    if enabled and hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=True)
    if enabled and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def make_grad_scaler(enabled: bool):
    """Create a GradScaler compatible with current and older PyTorch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def unwrap_model_output(model_out: Any) -> Any:
    """Return the output tensor/dict from models that also return a representation."""
    return model_out[0] if isinstance(model_out, tuple) else model_out


def extract_single_logits(model_out: Any) -> torch.Tensor:
    """Extract single-head logits from common model output formats."""
    out = unwrap_model_output(model_out)
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for key in ("head1", "logits", "out"):
            if key in out:
                return out[key]
    raise TypeError(f"Unsupported model output type: {type(model_out)}")


def extract_dual_outputs(model_out: Any) -> dict[str, torch.Tensor]:
    """Extract dual-head outputs and validate required keys."""
    out = unwrap_model_output(model_out)
    if not isinstance(out, dict) or "head1" not in out or "head2" not in out:
        raise TypeError("Dual-head model output must contain 'head1' and 'head2'.")
    return out


def squeeze_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Squeeze binary logits shaped as (N, 1) into (N,)."""
    if logits.dim() == 2 and logits.shape[1] == 1:
        return logits.squeeze(1)
    return logits


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def scn_model_kwargs(cfg: Any) -> dict[str, Any]:
    """Build shared SCN keyword arguments from a training config."""
    return dict(
        n_channels=cfg.n_ch,
        backbone=False,
        tau=cfg.tau,
        thresh=cfg.thresh,
        soft_reset=not cfg.hard_reset,
        merge_polarities=cfg.merge_polarities,
        event_scale=1.0,
        len_sw=None,
        time_steps=None,
        conv_channels=tuple(cfg.conv_ch),
        conv_kernel_sizes=tuple(cfg.kernels),
        conv_strides=tuple(cfg.strides),
        conv_paddings=None,
        pool_kernel_sizes=tuple(cfg.pool_kernels),
        pool_strides=tuple(cfg.pool_strides),
        pool_paddings=tuple(cfg.pool_paddings),
        offset_dim=cfg.n_ch,
        offset_hidden=cfg.offset_hidden,
        offset_scale=1.0,
        offset_drop=0.0,
        spiking_heads=cfg.spiking_heads,
        head_rate_scale=cfg.head_rate_scale,
        use_bn=True,
        p_drop=cfg.p_drop,
        bias=False,
    )


def print_kv_block(title: str, items: Iterable[tuple[str, Any]]) -> None:
    """Print a compact key-value block."""
    print(f"[{title}]")
    for key, value in items:
        print(f"  {key:<14}: {value}")


def format_single_epoch_line(
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    lr: float,
    epoch_time: float,
    eta: float,
    saved: bool,
) -> str:
    """Format one epoch line for single-head training."""
    saved_msg = " | saved=1" if saved else ""
    return (
        f"[train] epoch={epoch:03d}/{total_epochs:03d} "
        f"tr_loss={train_metrics['loss']:.4f} tr_acc={train_metrics['acc']:.4f} | "
        f"va_loss={val_metrics['loss']:.4f} va_acc={val_metrics['acc']:.4f} | "
        f"lr={lr:.2e} time={format_seconds(epoch_time)} eta={format_seconds(eta)}{saved_msg}"
    )


def format_dual_epoch_line(
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    lr: float,
    epoch_time: float,
    eta: float,
    saved: bool,
) -> str:
    """Format one epoch line for dual-head training."""
    tr_h2 = f"{train_metrics['acc_h2']:.4f}" if np.isfinite(train_metrics["acc_h2"]) else "NA"
    va_h2 = f"{val_metrics['acc_h2']:.4f}" if np.isfinite(val_metrics["acc_h2"]) else "NA"
    saved_msg = " | saved=1" if saved else ""
    return (
        f"[train] epoch={epoch:03d}/{total_epochs:03d} "
        f"tr_loss={train_metrics['loss']:.4f} tr_h1={train_metrics['acc_h1']:.4f} tr_h2={tr_h2} | "
        f"va_loss={val_metrics['loss']:.4f} va_h1={val_metrics['acc_h1']:.4f} va_h2={va_h2} | "
        f"lr={lr:.2e} time={format_seconds(epoch_time)} eta={format_seconds(eta)}{saved_msg}"
    )


def fit_model(
    *,
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    cfg: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    metrics_dir: Path,
    ckpt_path: Path,
    run_epoch_fn: Callable[..., dict[str, Any]],
    metric_name: str = "loss",
    format_epoch_fn: Callable[[int, int, dict[str, Any], dict[str, Any], float, float, float, bool], str] | None = None,
    checkpoint_extra_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run train/validation epochs, checkpoint the best model, and save history."""
    metrics_dir = Path(metrics_dir)
    ckpt_path = Path(ckpt_path)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started_at = time.time()

    for epoch in range(1, int(cfg.epochs) + 1):
        ep_start = time.time()

        train_metrics = run_epoch_fn(train_loader, optimizer=optimizer, scaler=scaler, collect_predictions=False)
        val_metrics = run_epoch_fn(val_loader, optimizer=None, scaler=None, collect_predictions=False)

        score = float(val_metrics[metric_name])
        scheduler.step(score)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = score < best_val
        epoch_time = time.time() - ep_start

        if improved:
            best_val = score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
            "time_sec": epoch_time,
            "best_val_loss": best_val,
            "bad_epochs": bad_epochs,
        }
        history.append(row)
        save_json(metrics_dir / "history.json", history)

        if improved:
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val,
                "config": json_ready(cfg),
                "history": json_ready(history),
            }
            if checkpoint_extra_fn is not None:
                payload.update(checkpoint_extra_fn())
            torch.save(payload, ckpt_path)

        elapsed = time.time() - started_at
        avg_epoch = elapsed / epoch
        eta = avg_epoch * max(int(cfg.epochs) - epoch, 0)
        if format_epoch_fn is not None:
            print(format_epoch_fn(epoch, int(cfg.epochs), train_metrics, val_metrics, lr, epoch_time, eta, improved), flush=True)

        if bad_epochs >= int(cfg.patience):
            print(f"[early-stop] epoch={epoch} best_val_loss={best_val:.6f}", flush=True)
            break

    return {
        "history": history,
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "started_at": float(started_at),
    }


def save_classification_outputs(
    metrics_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    names: Sequence[str],
    prefix: str = "",
) -> None:
    """Save predictions, a classification report, and confusion matrices."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""

    np.save(metrics_dir / f"{stem}y_true.npy", y_true.astype(np.int64, copy=False))
    np.save(metrics_dir / f"{stem}y_pred.npy", y_pred.astype(np.int64, copy=False))

    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception as exc:
        (metrics_dir / f"{stem}classification_report.txt").write_text(
            f"sklearn is not available; report skipped. Error: {exc}\n",
            encoding="utf-8",
        )
        return

    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(names),
        digits=4,
        zero_division=0,
    )
    (metrics_dir / f"{stem}classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    np.save(metrics_dir / f"{stem}confusion_matrix.npy", cm)

    cm_norm = confusion_matrix(y_true, y_pred, labels=list(labels), normalize="true")
    np.save(metrics_dir / f"{stem}confusion_matrix_norm.npy", cm_norm)
