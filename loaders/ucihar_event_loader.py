#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UCI HAR event-loader utilities for Sigma-Delta event-based training."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:  # Torch is only required when DataLoader objects are requested.
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    DataLoader = None
    Dataset = object

try:
    from loaders import utils as u
except ModuleNotFoundError:  # Allows: python loaders/ucihar_event_loader.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from loaders import utils as u


SPLIT_NAMES = ("train", "val", "test")
ARRAY_KEYS = ("x_ev", "offset", "y", "domain")

ACTIVITY_LABELS: dict[int, str] = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

DEFAULT_THETA_UCIHAR = (0.151215, 0.297011, 0.135961, 0.091156, 0.067695, 0.066330)


@dataclass(frozen=True)
class UCIHAREventLoaderConfig:
    data_root: str = "data"
    dataset_dir: str = "UCI HAR Dataset"
    case: str = "random"
    target_domain: int = 0
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    val_ratio: float = 0.1
    seed: int = 0
    raw_win: int = 128
    ups: float = 5.0
    out_win: int | None = None
    interp_method: Literal["linear", "zoh", "cubic"] = "linear"
    theta_sd: tuple[float, ...] = DEFAULT_THETA_UCIHAR
    dead_zone: float = 0.5
    sd_init: str = "x0"
    n_ch: int = 6
    ev_dtype: str = "uint8"
    offset_dtype: str = "float32"
    cache_dir: str = "data/ucihar_cache"
    run_name: str = "default"
    verbose_every: int = 1000
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True

    def resolved_out_win(self) -> int:
        if self.out_win is not None:
            return int(self.out_win)
        from preprocess.interpolate_signal import derive_output_length

        return int(derive_output_length(self.raw_win, self.ups))


class UCIHAREventDataset(Dataset):  # type: ignore[misc]
    """Torch dataset wrapping cached UCI HAR event arrays."""

    def __init__(self, split: dict[str, np.ndarray]) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to build UCIHAREventDataset.")

        self.x_ev = torch.as_tensor(np.array(split["x_ev"], dtype=np.float32, copy=True))
        self.offset = torch.as_tensor(np.array(split["offset"], dtype=np.float32, copy=True))
        self.y = torch.as_tensor(np.array(split["y"], dtype=np.int64, copy=True))
        self.domain = torch.as_tensor(np.array(split["domain"], dtype=np.int64, copy=True))

        n = int(self.y.shape[0])
        if self.x_ev.shape[0] != n or self.offset.shape[0] != n or self.domain.shape[0] != n:
            raise ValueError("All split arrays must have the same first dimension.")
        if self.x_ev.ndim != 4 or self.x_ev.shape[2:] != (6, 2):
            raise ValueError(f"x_ev must have shape (N, Te, 6, 2). Got {tuple(self.x_ev.shape)}.")
        if self.offset.ndim != 2 or self.offset.shape[1] != 6:
            raise ValueError(f"offset must have shape (N, 6). Got {tuple(self.offset.shape)}.")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "x": self.x_ev[idx],
            "offset": self.offset[idx],
            "y": self.y[idx],
            "domain": self.domain[idx],
        }


def load_or_build_ucihar_event_splits(
    data_root: str | os.PathLike[str] = "data",
    dataset_dir: str = "UCI HAR Dataset",
    cache_dir: str | os.PathLike[str] = "data/ucihar_cache",
    run_name: str = "default",
    cfg: UCIHAREventLoaderConfig | None = None,
    refresh: bool = False,
    mmap_mode: str | None = None,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load cached UCI HAR event splits or build them from raw UCI HAR files."""
    u.ensure_project_root_on_path(__file__)
    cfg = UCIHAREventLoaderConfig(data_root=str(data_root), dataset_dir=dataset_dir, cache_dir=str(cache_dir), run_name=run_name) if cfg is None else cfg

    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / f"ucihar_event_loader_{run_name}"
    meta_path = out_dir / "metadata.json"

    expected = _metadata_template(data_root=data_root, dataset_dir=dataset_dir, cache_dir=cache_dir, run_name=run_name, cfg=cfg)
    cache_ok = _cache_is_valid(out_dir=out_dir, meta_path=meta_path, expected=expected)

    if cache_ok and not refresh:
        if verbose:
            print(f"[ucihar:cache] Loaded event splits from cache: {out_dir}")
        return load_event_splits(out_dir, mmap_mode=mmap_mode, verbose=verbose), u.read_json(meta_path)

    if verbose and out_dir.exists() and not cache_ok:
        print("[ucihar:cache] Existing cache has different settings. Rebuilding.")
    if verbose and not out_dir.exists():
        print("[ucihar:cache] Building event-loader cache.")

    if out_dir.exists():
        u.safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_ucihar_preprocessed(cfg=cfg, data_root=data_root, dataset_dir=dataset_dir, verbose=verbose)
    splits = {
        "train": build_event_split("train", raw.X_train, raw.y_train, raw.d_train, cfg, verbose=verbose),
        "val": build_event_split("val", raw.X_val, raw.y_val, raw.d_val, cfg, verbose=verbose),
        "test": build_event_split("test", raw.X_test, raw.y_test, raw.d_test, cfg, verbose=verbose),
    }

    metadata = {
        **expected,
        "split_summary": {name: summarize_split(split) for name, split in splits.items()},
        "preprocess_meta": u.json_ready(getattr(raw, "meta", {})),
    }
    save_event_splits(splits, out_dir, metadata=metadata, verbose=verbose)

    if verbose:
        print(f"[ucihar:cache] Event-loader cache ready: {out_dir}")
    return splits, metadata


def build_ucihar_training_loaders(
    data_root: str | os.PathLike[str] = "data",
    dataset_dir: str = "UCI HAR Dataset",
    cache_dir: str | os.PathLike[str] = "data/ucihar_cache",
    run_name: str = "default",
    cfg: UCIHAREventLoaderConfig | None = None,
    refresh: bool = False,
    mmap_mode: str | None = "r",
    verbose: bool = True,
    balanced_sampler: bool = False,
) -> dict[str, Any]:
    """Build/load cached event splits and return train/val/test DataLoaders."""
    cfg = UCIHAREventLoaderConfig(data_root=str(data_root), dataset_dir=dataset_dir, cache_dir=str(cache_dir), run_name=run_name) if cfg is None else cfg
    splits, metadata = load_or_build_ucihar_event_splits(
        data_root=data_root,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        run_name=run_name,
        cfg=cfg,
        refresh=refresh,
        mmap_mode=mmap_mode,
        verbose=verbose,
    )
    loaders = make_event_loaders(splits, cfg=cfg, balanced_sampler=balanced_sampler)
    return {
        "loaders": loaders,
        "splits": splits,
        "metadata": metadata,
        "class_weights": estimate_class_weights(splits["train"]),
    }


def make_event_loaders(
    splits: dict[str, dict[str, np.ndarray]],
    cfg: UCIHAREventLoaderConfig | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    balanced_sampler: bool = False,
) -> dict[str, Any]:
    """Create PyTorch DataLoaders from cached split arrays."""
    if DataLoader is None:
        raise ImportError("PyTorch is required to create DataLoader objects.")

    cfg = UCIHAREventLoaderConfig() if cfg is None else cfg
    bs = cfg.batch_size if batch_size is None else int(batch_size)
    nw = cfg.num_workers if num_workers is None else int(num_workers)
    pm = cfg.pin_memory if pin_memory is None else bool(pin_memory)

    sampler = make_weighted_sampler(splits["train"]) if balanced_sampler else None
    return {
        "train": DataLoader(UCIHAREventDataset(splits["train"]), batch_size=bs, shuffle=sampler is None, sampler=sampler, drop_last=False, num_workers=nw, pin_memory=pm),
        "val": DataLoader(UCIHAREventDataset(splits["val"]), batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pm),
        "test": DataLoader(UCIHAREventDataset(splits["test"]), batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pm),
    }


def build_event_split(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    cfg: UCIHAREventLoaderConfig,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Interpolate and Sigma-Delta encode one UCI HAR split."""
    from preprocess.event_encoding import sigma_delta
    from preprocess.interpolate_signal import interpolate_signal, make_resample_plan, resample_window_linear_exact

    X = _as_ucihar_x(X)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    domain = np.asarray(domain, dtype=np.int64).reshape(-1)

    if X.shape[0] != y.size or X.shape[0] != domain.size:
        raise ValueError(f"Inconsistent split sizes for {name}: X={X.shape[0]}, y={y.size}, domain={domain.size}.")
    if X.shape[1:] != (cfg.raw_win, cfg.n_ch):
        raise ValueError(f"Expected X shape (N, {cfg.raw_win}, {cfg.n_ch}) for {name}. Got {X.shape}.")

    theta = np.asarray(cfg.theta_sd, dtype=np.float32)
    if theta.shape != (cfg.n_ch,):
        raise ValueError(f"theta_sd must contain {cfg.n_ch} values. Got shape={theta.shape}.")
    if np.any(theta <= 0.0):
        raise ValueError("theta_sd values must be positive.")

    out_win = cfg.resolved_out_win()
    ev_dtype = np.dtype(cfg.ev_dtype)
    off_dtype = np.dtype(cfg.offset_dtype)
    ev = np.empty((X.shape[0], out_win, cfg.n_ch, 2), dtype=ev_dtype)
    offset = np.asarray(X[:, 0, :], dtype=off_dtype, order="C")
    plan = make_resample_plan(cfg.raw_win, out_win, method="linear", dtype=np.float32) if cfg.interp_method == "linear" else None

    if verbose:
        print(f"[ucihar:split:{name}] Encoding X={X.shape} -> x_ev={ev.shape} | UPS={cfg.ups} | Te={out_win} | dead_zone={cfg.dead_zone}")

    for i in range(X.shape[0]):
        if plan is not None:
            x_ip = resample_window_linear_exact(X[i], plan=plan)
        else:
            x_ip = interpolate_signal(X[i], out_len=out_win, method=cfg.interp_method, dtype=np.float32)

        out = sigma_delta(x_ip, theta=theta, init=cfg.sd_init, dead_zone=cfg.dead_zone, return_reconstruction=False)
        ev_i = out[0] if isinstance(out, (tuple, list)) else out
        ev[i] = np.asarray(ev_i, dtype=ev_dtype)

        if verbose and cfg.verbose_every and (i + 1) % int(cfg.verbose_every) == 0:
            print(f"[ucihar:split:{name}] Encoded windows: {i + 1}/{X.shape[0]}")

    split = {
        "x_ev": ev,
        "offset": offset,
        "y": y.astype(np.int64, copy=False),
        "domain": domain.astype(np.int64, copy=False),
    }

    if verbose:
        print(f"[ucihar:split:{name}] Done | N={len(y)} | counts={named_class_counts(split['y'])}")
    return split


def save_event_splits(
    splits: dict[str, dict[str, np.ndarray]],
    root: str | os.PathLike[str],
    metadata: dict[str, Any] | None = None,
    verbose: bool = True,
) -> None:
    """Save UCI HAR event split arrays as .npy files."""
    u.save_event_splits(splits, root, split_names=SPLIT_NAMES, array_keys=ARRAY_KEYS, metadata=metadata, verbose=verbose)


def load_event_splits(
    root: str | os.PathLike[str],
    mmap_mode: str | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Load cached UCI HAR event splits from .npy files."""
    return u.load_event_splits(root, split_names=SPLIT_NAMES, array_keys=ARRAY_KEYS, mmap_mode=mmap_mode, mmap_keys=None, verbose=verbose)


def summarize_split(split: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return shape and class-count information for one UCI HAR split."""
    y = np.asarray(split["y"], dtype=np.int64)
    return {
        "n": int(y.size),
        "x_ev_shape": list(np.asarray(split["x_ev"]).shape),
        "offset_shape": list(np.asarray(split["offset"]).shape),
        "class_counts": named_class_counts(y),
        "domain_count": int(np.unique(np.asarray(split["domain"])).size),
    }


def named_class_counts(y: np.ndarray) -> dict[str, int]:
    """Return class counts using UCI HAR activity names."""
    return u.named_class_counts(y, ACTIVITY_LABELS)


def estimate_class_weights(split: dict[str, np.ndarray], n_classes: int = 6) -> np.ndarray:
    """Compute balanced cross-entropy weights from a cached training split."""
    return u.balanced_class_weights(split["y"], n_classes=n_classes)


def make_weighted_sampler(split: dict[str, np.ndarray]) -> Any:
    """Create a balanced WeightedRandomSampler for the training split."""
    return u.weighted_random_sampler(split["y"], n_classes=6)


def _load_ucihar_preprocessed(
    cfg: UCIHAREventLoaderConfig,
    data_root: Path,
    dataset_dir: str,
    verbose: bool,
) -> Any:
    from preprocess.preprocess_ucihar import preprocess_ucihar

    return preprocess_ucihar(
        data_root=data_root,
        dataset_dir=dataset_dir,
        case=cfg.case,
        target_domain=cfg.target_domain,
        split=cfg.split,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        refresh_cache=False,
        model_shape=False,
        verbose=verbose,
    )


def _as_ucihar_x(X: np.ndarray) -> np.ndarray:
    """Normalize accepted UCI HAR shapes to (N, 128, 6)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and X.shape[2] == 1 and X.shape[3] == 6:
        X = X[:, :, 0, :]
    elif X.ndim == 4 and X.shape[1] == 1 and X.shape[-1] == 6:
        X = X[:, 0, :, :]
    if X.ndim != 3 or X.shape[1:] != (128, 6):
        raise ValueError(f"UCI HAR X must have shape (N,128,6) or (N,128,1,6). Got {X.shape}.")
    return X.astype(np.float32, copy=False)


def _metadata_template(
    data_root: Path,
    dataset_dir: str,
    cache_dir: Path,
    run_name: str,
    cfg: UCIHAREventLoaderConfig,
) -> dict[str, Any]:
    return {
        "dataset": "ucihar",
        "data_root": str(data_root),
        "dataset_dir": str(dataset_dir),
        "cache_dir": str(cache_dir),
        "run_name": str(run_name),
        "config": u.json_ready(asdict(cfg)),
        "array_keys": list(ARRAY_KEYS),
        "activity_labels": {str(k): v for k, v in ACTIVITY_LABELS.items()},
    }


def _cache_is_valid(out_dir: Path, meta_path: Path, expected: dict[str, Any]) -> bool:
    return u.event_cache_is_valid(
        out_dir,
        meta_path,
        expected,
        split_names=SPLIT_NAMES,
        array_keys=ARRAY_KEYS,
        metadata_keys=("config",),
    )


def main() -> None:
    """Build or reuse the default cache and print a short summary."""
    u.ensure_project_root_on_path(__file__)
    cfg = UCIHAREventLoaderConfig()
    splits, metadata = load_or_build_ucihar_event_splits(
        data_root=cfg.data_root,
        dataset_dir=cfg.dataset_dir,
        cache_dir=cfg.cache_dir,
        run_name=cfg.run_name,
        cfg=cfg,
        refresh=False,
        mmap_mode=None,
        verbose=True,
    )

    print("\n=== UCI HAR EVENT LOADER CHECK ===")
    for name in SPLIT_NAMES:
        print(f"{name:>5}: {summarize_split(splits[name])}")
    print(f"cache: {Path(cfg.cache_dir) / f'ucihar_event_loader_{cfg.run_name}'}")
    print(f"Te: {cfg.resolved_out_win()} | theta: {np.asarray(cfg.theta_sd, dtype=np.float32)}")
    print(f"metadata keys: {sorted(metadata.keys())}")

__all__ = [
    "ACTIVITY_LABELS", "ARRAY_KEYS", "DEFAULT_THETA_UCIHAR", "SPLIT_NAMES",
    "UCIHAREventDataset", "UCIHAREventLoaderConfig", "build_event_split",
    "build_ucihar_training_loaders", "estimate_class_weights",
    "load_event_splits", "load_or_build_ucihar_event_splits",
    "make_event_loaders", "make_weighted_sampler", "named_class_counts",
    "save_event_splits", "summarize_split",
]

if __name__ == "__main__":
    main()
