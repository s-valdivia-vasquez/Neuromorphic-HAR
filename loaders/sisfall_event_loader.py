#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SisFall event-loader pipeline for dual-head training."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  
    torch = None
    DataLoader = None
    Dataset = object

try:
    from loaders import utils as u
except ModuleNotFoundError:  
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from loaders import utils as u

from preprocess.event_encoding import sigma_delta
from preprocess.utils import stratified_file_split
from preprocess.interpolate_signal import make_linear_resample_plan, resample_window_linear_exact

H1_FALL = 0
H1_DYNAMIC = 1
H1_STATIC = 2

SRC_AUTOTAG = 0
SRC_MANUAL = 1

SPLIT_NAMES = ("train", "val", "test")
ARRAY_KEYS = ("x_ev", "offset", "y_h1", "y_h2", "src_h2", "w_h2", "file_idx", "start")

@dataclass(frozen=True)
class EventLoaderConfig:
    seed: int = 0
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    raw_win: int = 410
    out_win: int = 2048
    n_ch: int = 6
    stride_fall: int = 160
    stride_dynamic: int = 410
    stride_static: int = 205
    fall_policy: str = "contain_global_max"
    theta_sd: tuple[float, ...] = (61.000, 47.838, 51.000, 324.290, 201.000, 157.838)
    dead_zone: float = 0.5
    sd_init: str = "x0"
    manual_weight: float = 1.0
    autotag_weight: float = 0.4
    ambiguous_weight: float = 0.0
    ignore_ambiguous: bool = True
    acc: str = "ADXL345"
    dtype: str = "int32"
    ev_dtype: str = "uint8"
    offset_dtype: str = "float32"
    verbose_every: int = 2000
    batch_size: int = 512
    num_workers: int = 12
    pin_memory: bool = True

class MultiTaskEventDataset(Dataset):  # type: ignore[misc]
    """Torch dataset wrapping preprocessed event windows and multitask labels."""

    def __init__(self, split: dict[str, np.ndarray]) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to build MultiTaskEventDataset.")

        self.x_ev = torch.as_tensor(np.asarray(split["x_ev"], dtype=np.float32))
        self.offset = torch.as_tensor(np.asarray(split["offset"], dtype=np.float32))
        self.y_h1 = torch.as_tensor(np.asarray(split["y_h1"], dtype=np.int64))
        self.y_h2 = torch.as_tensor(np.asarray(split["y_h2"], dtype=np.float32))
        self.src_h2 = torch.as_tensor(np.asarray(split["src_h2"], dtype=np.uint8))
        self.w_h2 = torch.as_tensor(np.asarray(split["w_h2"], dtype=np.float32))
        self.file_idx = torch.as_tensor(np.asarray(split["file_idx"], dtype=np.int32))
        self.start = torch.as_tensor(np.asarray(split["start"], dtype=np.int32))

        n = int(self.y_h1.shape[0])
        if self.x_ev.shape[0] != n or self.offset.shape[0] != n:
            raise ValueError("All split arrays must have the same first dimension.")

    def __len__(self) -> int:
        return int(self.y_h1.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "x": self.x_ev[idx],
            "offset": self.offset[idx],
            "y_h1": self.y_h1[idx],
            "y_h2": self.y_h2[idx],
            "src_h2": self.src_h2[idx],
            "w_h2": self.w_h2[idx],
            "file_idx": self.file_idx[idx],
            "start": self.start[idx],
        }

class SingleHeadSisFallDataset(Dataset):  # type: ignore[misc]
    """Torch dataset that remaps SisFall multitask labels to four classes."""

    def __init__(
        self,
        split: dict[str, np.ndarray],
        ambiguous_policy: str = "drop",
        ambiguous_weight: float = 0.0,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to build SingleHeadSisFallDataset.")

        self.x_ev = torch.as_tensor(np.asarray(split["x_ev"], dtype=np.float32))
        self.offset = torch.as_tensor(np.asarray(split["offset"], dtype=np.float32))

        y_h1 = np.asarray(split["y_h1"], dtype=np.int64).reshape(-1)
        y_h2 = np.asarray(split["y_h2"], dtype=np.float32).reshape(-1)
        w_h2 = np.asarray(split["w_h2"], dtype=np.float32).reshape(-1)

        n = int(y_h1.shape[0])
        if self.x_ev.shape[0] != n or self.offset.shape[0] != n:
            raise ValueError("All split arrays must have the same first dimension.")

        y = np.full(n, -1, dtype=np.int64)
        weight = np.ones(n, dtype=np.float32)

        m_fall = y_h1 == H1_FALL
        m_dynamic = y_h1 == H1_DYNAMIC
        m_static = y_h1 == H1_STATIC
        m_stable = m_static & np.isclose(y_h2, 0.0)
        m_transition = m_static & np.isclose(y_h2, 1.0)
        m_ambiguous = m_static & ~(m_stable | m_transition)

        y[m_fall] = 0
        y[m_dynamic] = 1
        y[m_stable] = 2
        y[m_transition] = 3

        valid_static = m_stable | m_transition
        weight[valid_static] = np.where(w_h2[valid_static] > 0, w_h2[valid_static], 1.0)

        ambiguous_policy = str(ambiguous_policy).lower()
        if ambiguous_policy == "stable":
            y[m_ambiguous] = 2
            weight[m_ambiguous] = max(float(ambiguous_weight), 0.0)
        elif ambiguous_policy == "transition":
            y[m_ambiguous] = 3
            weight[m_ambiguous] = max(float(ambiguous_weight), 0.0)
        elif ambiguous_policy != "drop":
            raise ValueError("ambiguous_policy must be 'drop', 'stable', or 'transition'.")

        keep = y >= 0
        if not np.any(keep):
            raise ValueError("Single-head remapping removed all samples.")

        self.indices = np.nonzero(keep)[0].astype(np.int64, copy=False)
        self.y = torch.as_tensor(y[keep], dtype=torch.long)
        self.sample_weight = torch.as_tensor(weight[keep], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        src_idx = int(self.indices[idx])
        return {
            "x": self.x_ev[src_idx],
            "offset": self.offset[src_idx],
            "y": self.y[idx],
            "sample_weight": self.sample_weight[idx],
        }

def load_or_build_sisfall_event_splits(
    dataset_root: str | os.PathLike[str] = "data/SisFall_dataset",
    label_dir: str | os.PathLike[str] = "data/labels_transitions",
    cache_dir: str | os.PathLike[str] = "data/sisfall_cache",
    result_dir: str | os.PathLike[str] = "data/sisfall_tagging_results",
    run_name: str = "default",
    cfg: EventLoaderConfig | None = None,
    posture_cfg: Any | None = None,
    refresh: bool = False,
    refresh_posture: bool = False,
    mmap_mode: str | None = None,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load cached training splits or build the full preprocessing pipeline."""
    cfg = EventLoaderConfig() if cfg is None else cfg
    dataset_root = Path(dataset_root)
    label_dir = Path(label_dir)
    cache_dir = Path(cache_dir)
    result_dir = Path(result_dir)
    out_dir = cache_dir / f"sisfall_event_loader_{run_name}"
    meta_path = out_dir / "metadata.json"

    expected = _metadata_template(dataset_root, label_dir, result_dir, run_name, cfg)
    cache_ok = _cache_is_valid(out_dir, meta_path, expected)

    if cache_ok and not refresh and not refresh_posture:
        if verbose:
            print("[sisfall:cache] Loaded preprocessed event splits from cache.")
        return load_event_splits(out_dir, mmap_mode=mmap_mode, verbose=verbose), u.read_json(meta_path)

    if verbose and out_dir.exists() and not cache_ok:
        print("[sisfall:cache] Existing cache has different settings. Rebuilding.")
    if verbose and not out_dir.exists():
        print("[sisfall:cache] Building event-loader cache.")

    x, y, meta, skipped = _load_sisfall(dataset_root, cache_dir, cfg, verbose=verbose)
    posture = _load_or_build_posture_tags(
        dataset_root=dataset_root,
        label_dir=label_dir,
        result_dir=result_dir,
        cache_dir=cache_dir,
        run_name=run_name,
        cfg=cfg,
        posture_cfg=posture_cfg,
        refresh=refresh_posture,
        verbose=verbose,
    )

    ann = build_head2_annotation_map(
        autotag_records=posture.get("records", []),
        manual_records=posture.get("manual_records", []),
    )
    idx_tr, idx_va, idx_te = stratified_file_split(y, split=cfg.split, seed=cfg.seed)

    total_windows = (
        count_event_windows(x, y, idx_tr, cfg)
        + count_event_windows(x, y, idx_va, cfg)
        + count_event_windows(x, y, idx_te, cfg)
    )

    if out_dir.exists():
        u.safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, dict[str, np.ndarray]] = {}
    progress_offset = 0

    splits["train"] = build_event_split(
        x,
        y,
        meta,
        idx_tr,
        ann,
        cfg,
        name="train",
        verbose=verbose,
        progress_offset=progress_offset,
        progress_total=total_windows,
    )
    progress_offset += int(splits["train"]["x_ev"].shape[0])

    splits["val"] = build_event_split(
        x,
        y,
        meta,
        idx_va,
        ann,
        cfg,
        name="val",
        verbose=verbose,
        progress_offset=progress_offset,
        progress_total=total_windows,
    )
    progress_offset += int(splits["val"]["x_ev"].shape[0])

    splits["test"] = build_event_split(
        x,
        y,
        meta,
        idx_te,
        ann,
        cfg,
        name="test",
        verbose=verbose,
        progress_offset=progress_offset,
        progress_total=total_windows,
    )

    split_windows = {name: int(split["x_ev"].shape[0]) for name, split in splits.items()}
    split_windows["total"] = int(sum(split_windows.values()))

    metadata = {
        **expected,
        "raw_data": {"recordings": int(len(x)), "skipped": int(skipped)},
        "split_files": {"train": int(len(idx_tr)), "val": int(len(idx_va)), "test": int(len(idx_te))},
        "split_windows": split_windows,
        "split_summary": {name: summarize_split(split) for name, split in splits.items()},
        "posture_summary": posture.get("summary", {}),
    }

    save_event_splits(splits, out_dir, metadata=metadata, verbose=verbose)
    if verbose:
        print(f"[sisfall:cache] Event-loader cache ready: {out_dir}")
    return splits, metadata

def build_sisfall_training_loaders(
    dataset_root: str | os.PathLike[str] = "data/SisFall_dataset",
    label_dir: str | os.PathLike[str] = "data/labels_transitions",
    cache_dir: str | os.PathLike[str] = "data/sisfall_cache",
    result_dir: str | os.PathLike[str] = "data/sisfall_tagging_results",
    run_name: str = "default",
    cfg: EventLoaderConfig | None = None,
    posture_cfg: Any | None = None,
    refresh: bool = False,
    refresh_posture: bool = False,
    mmap_mode: str | None = "r",
    verbose: bool = True,
) -> dict[str, Any]:
    """Build or load preprocessed splits and return Torch DataLoaders."""
    cfg = EventLoaderConfig() if cfg is None else cfg
    splits, metadata = load_or_build_sisfall_event_splits(
        dataset_root=dataset_root,
        label_dir=label_dir,
        cache_dir=cache_dir,
        result_dir=result_dir,
        run_name=run_name,
        cfg=cfg,
        posture_cfg=posture_cfg,
        refresh=refresh,
        refresh_posture=refresh_posture,
        mmap_mode=mmap_mode,
        verbose=verbose,
    )
    loaders = make_event_loaders(splits, cfg=cfg)
    return {
        "loaders": loaders,
        "splits": splits,
        "metadata": metadata,
        "pos_weight_h2": estimate_head2_pos_weight(splits["train"]),
    }

def make_event_loaders(
    splits: dict[str, dict[str, np.ndarray]],
    cfg: EventLoaderConfig | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> dict[str, Any]:
    """Create train, validation, and test DataLoaders from cached split arrays."""
    if DataLoader is None:
        raise ImportError("PyTorch is required to create DataLoader objects.")

    cfg = EventLoaderConfig() if cfg is None else cfg
    bs = cfg.batch_size if batch_size is None else int(batch_size)
    nw = cfg.num_workers if num_workers is None else int(num_workers)
    pm = cfg.pin_memory if pin_memory is None else bool(pin_memory)

    return {
        "train": DataLoader(MultiTaskEventDataset(splits["train"]), batch_size=bs, shuffle=True, drop_last=False, num_workers=nw, pin_memory=pm),
        "val": DataLoader(MultiTaskEventDataset(splits["val"]), batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pm),
        "test": DataLoader(MultiTaskEventDataset(splits["test"]), batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pm),
    }

def make_single_head_event_loaders(
    splits: dict[str, dict[str, np.ndarray]],
    batch_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True,
    ambiguous_policy: str = "drop",
    ambiguous_weight: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create single-head SisFall datasets and DataLoaders from cached split arrays."""
    if DataLoader is None:
        raise ImportError("PyTorch is required to create DataLoader objects.")

    datasets = {
        name: SingleHeadSisFallDataset(
            split,
            ambiguous_policy=ambiguous_policy,
            ambiguous_weight=ambiguous_weight,
        )
        for name, split in splits.items()
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders, datasets

def count_event_windows(
    x: np.ndarray,
    y: np.ndarray,
    file_idx: Iterable[int],
    cfg: EventLoaderConfig,
) -> int:
    """Count the event windows generated by a set of recordings."""
    total = 0
    for fi in file_idx:
        fi = int(fi)
        xi = np.asarray(x[fi], dtype=np.float32)
        yi = int(y[fi])
        total += len(_window_starts_for_record(xi, yi, cfg))
    return int(total)


def build_event_split(
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
    file_idx: Iterable[int],
    ann: dict[tuple[Any, ...], dict[int, tuple[float, int]]],
    cfg: EventLoaderConfig,
    name: str = "split",
    verbose: bool = True,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> dict[str, np.ndarray]:
    """Segment, linearly interpolate, and Sigma-Delta encode one file split."""
    plan = make_linear_resample_plan(in_len=cfg.raw_win, out_len=cfg.out_win)
    theta = np.asarray(cfg.theta_sd, dtype=np.float32)
    ev_dtype = np.dtype(cfg.ev_dtype)
    off_dtype = np.dtype(cfg.offset_dtype)

    if theta.shape != (cfg.n_ch,):
        raise ValueError(f"theta_sd must contain {cfg.n_ch} values.")

    x_ev: list[np.ndarray] = []
    offs: list[np.ndarray] = []
    y1: list[int] = []
    y2: list[float] = []
    src2: list[int] = []
    w2: list[float] = []
    fidx: list[int] = []
    starts_out: list[int] = []

    h1_counts = {H1_FALL: 0, H1_DYNAMIC: 0, H1_STATIC: 0}
    h2_counts = {"stable_posture": 0, "postural_transition": 0, "ambiguous": 0, "manual": 0, "autotag": 0, "valid": 0}

    for fi in file_idx:
        fi = int(fi)
        xi = np.asarray(x[fi], dtype=np.float32)
        yi = int(y[fi])
        if xi.ndim != 2 or xi.shape[1] != cfg.n_ch or xi.shape[0] < cfg.raw_win:
            continue

        starts = _window_starts_for_record(xi, yi, cfg)
        rec_ann = ann.get(record_key(meta[fi]), {})
        for s in starts:
            x_raw = xi[s : s + cfg.raw_win]
            if x_raw.shape[0] != cfg.raw_win:
                continue

            x_ip = resample_window_linear_exact(x_raw, plan=plan)
            ev, _, _ = sigma_delta(x_ip, theta=theta, init=cfg.sd_init, dead_zone=cfg.dead_zone, return_reconstruction=False)
            y_h2, src_h2, w_h2 = _head2_target(yi, int(s), rec_ann, cfg, h2_counts)

            x_ev.append(np.asarray(ev, dtype=ev_dtype))
            offs.append(np.asarray(x_raw[0], dtype=off_dtype))
            y1.append(yi)
            y2.append(float(y_h2))
            src2.append(int(src_h2))
            w2.append(float(w_h2))
            fidx.append(fi)
            starts_out.append(int(s))
            h1_counts[yi] = h1_counts.get(yi, 0) + 1

            if verbose and cfg.verbose_every and len(x_ev) % int(cfg.verbose_every) == 0:
                current = int(progress_offset + len(x_ev))
                if progress_total is None:
                    print(f"[sisfall:split:{name}] Encoded windows: {current}")
                else:
                    print(f"[sisfall:split:{name}] Encoded windows: {current}/{int(progress_total)}")

    if not x_ev:
        raise RuntimeError(f"No windows were generated for split '{name}'.")

    out = {
        "x_ev": np.stack(x_ev).astype(ev_dtype, copy=False),
        "offset": np.stack(offs).astype(off_dtype, copy=False),
        "y_h1": np.asarray(y1, dtype=np.int64),
        "y_h2": np.asarray(y2, dtype=np.float32),
        "src_h2": np.asarray(src2, dtype=np.uint8),
        "w_h2": np.asarray(w2, dtype=np.float32),
        "file_idx": np.asarray(fidx, dtype=np.int32),
        "start": np.asarray(starts_out, dtype=np.int32),
    }

    if verbose:
        print(
            f"[sisfall:split:{name}] "
            f"N={out['x_ev'].shape[0]} | "
            f"H1 fall={h1_counts.get(H1_FALL, 0)}, "
            f"dynamic={h1_counts.get(H1_DYNAMIC, 0)}, "
            f"static={h1_counts.get(H1_STATIC, 0)} | "
            f"H2 valid={h2_counts['valid']}, "
            f"stable_posture={h2_counts['stable_posture']}, "
            f"postural_transition={h2_counts['postural_transition']}, "
            f"ambiguous={h2_counts['ambiguous']}"
        )
    return out

def build_head2_annotation_map(
    autotag_records: Iterable[Any] | None,
    manual_records: Iterable[Any] | None = None,
) -> dict[tuple[Any, ...], dict[int, tuple[float, int]]]:
    """Merge posture-window annotations, with manual labels taking priority."""
    ann: dict[tuple[Any, ...], dict[int, tuple[float, int]]] = {}

    def insert(records: Iterable[Any] | None, src: int) -> None:
        if records is None:
            return
        for rec in records:
            key = record_key(rec)
            starts = u.obj_get(rec, "starts")
            yw = u.obj_get(rec, "y_win")
            if starts is None or yw is None:
                continue
            starts = np.asarray(starts, dtype=np.int64).reshape(-1)
            yw = np.asarray(yw, dtype=np.float32).reshape(-1)
            if starts.shape[0] != yw.shape[0]:
                continue
            dst = ann.setdefault(key, {})
            for s, v in zip(starts, yw):
                if int(s) in dst and dst[int(s)][1] == SRC_MANUAL:
                    continue
                dst[int(s)] = (normalize_h2(v), int(src))

    insert(autotag_records, SRC_AUTOTAG)
    insert(manual_records, SRC_MANUAL)
    return ann

def save_event_splits(
    splits: dict[str, dict[str, np.ndarray]],
    root: str | os.PathLike[str],
    metadata: dict[str, Any] | None = None,
    verbose: bool = True,
) -> None:
    """Save SisFall event split arrays as .npy files."""
    u.save_event_splits(splits, root, split_names=SPLIT_NAMES, array_keys=ARRAY_KEYS, metadata=metadata, verbose=verbose)

def load_event_splits(
    root: str | os.PathLike[str],
    mmap_mode: str | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Load cached SisFall event splits from .npy files."""
    return u.load_event_splits(root, split_names=SPLIT_NAMES, array_keys=ARRAY_KEYS, mmap_mode=mmap_mode, mmap_keys=("x_ev",), verbose=verbose)

def summarize_split(split: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize class counts for one preprocessed split."""
    y1 = np.asarray(split["y_h1"])
    y2 = np.asarray(split["y_h2"])
    src = np.asarray(split["src_h2"])
    st = y1 == H1_STATIC
    valid = st & ~np.isclose(y2, 0.5)
    return {
        "n": int(len(y1)),
        "x_ev_shape": list(np.asarray(split["x_ev"]).shape),
        "h1": {
            "fall": int(np.sum(y1 == H1_FALL)),
            "normal_dynamic": int(np.sum(y1 == H1_DYNAMIC)),
            "normal_static": int(np.sum(y1 == H1_STATIC)),
        },
        "h2_static": {
            "valid": int(np.sum(valid)),
            "stable_posture": int(np.sum(st & np.isclose(y2, 0.0))),
            "postural_transition": int(np.sum(st & np.isclose(y2, 1.0))),
            "ambiguous": int(np.sum(st & np.isclose(y2, 0.5))),
            "manual": int(np.sum(valid & (src == SRC_MANUAL))),
            "autotag": int(np.sum(valid & (src == SRC_AUTOTAG))),
        },
    }

def estimate_head2_pos_weight(split: dict[str, np.ndarray]) -> float:
    """Estimate BCE pos_weight for valid static posture-transition labels."""
    y1 = np.asarray(split["y_h1"])
    y2 = np.asarray(split["y_h2"])
    m = (y1 == H1_STATIC) & ~np.isclose(y2, 0.5)
    if not np.any(m):
        return 1.0
    pos = int(np.sum(np.isclose(y2[m], 1.0)))
    neg = int(np.sum(np.isclose(y2[m], 0.0)))
    return 1.0 if pos == 0 else float(neg / max(pos, 1))

def record_key(obj: Any) -> tuple[Any, ...]:
    """Build a stable key shared by SisFall metadata and posture records."""
    f = u.obj_get(obj, "file", None) or u.obj_get(obj, "path", None) or u.obj_get(obj, "ruta", None)
    c = u.obj_get(obj, "code", None)
    s = u.obj_get(obj, "subject", None)
    r = u.obj_get(obj, "trial", None)
    return (u.basename_no_ext(f), u.canon(c), u.canon(s), u.canon(r))

def normalize_h2(v: Any) -> float:
    """Normalize posture-refinement labels to 0.0, 0.5, or 1.0."""
    fv = float(v)
    if np.isclose(fv, 1.0):
        return 1.0
    if np.isclose(fv, 0.0):
        return 0.0
    return 0.5

def _stride_for_h1(y_h1: int, cfg: EventLoaderConfig) -> int:
    if y_h1 == H1_FALL:
        stride = cfg.stride_fall
    elif y_h1 == H1_DYNAMIC:
        stride = cfg.stride_dynamic
    elif y_h1 == H1_STATIC:
        stride = cfg.stride_static
    else:
        return -1
    if int(stride) <= 0:
        raise ValueError("All strides must be positive.")
    return int(stride)


def _window_starts_for_record(xi: np.ndarray, y_h1: int, cfg: EventLoaderConfig) -> list[int]:
    if xi.ndim != 2 or xi.shape[1] != cfg.n_ch or xi.shape[0] < cfg.raw_win:
        return []

    stride = _stride_for_h1(int(y_h1), cfg)
    starts = list(range(0, xi.shape[0] - cfg.raw_win + 1, stride))

    if y_h1 == H1_FALL and cfg.fall_policy == "contain_global_max":
        abs_x = np.abs(xi)
        peak_idx = np.unique(np.where(abs_x == abs_x.max())[0])
        starts = [s for s in starts if np.any((peak_idx >= s) & (peak_idx < s + cfg.raw_win))]
    elif y_h1 == H1_FALL and cfg.fall_policy != "all":
        raise ValueError("fall_policy must be 'contain_global_max' or 'all'.")

    return starts


def _head2_target(
    y_h1: int,
    start: int,
    rec_ann: dict[int, tuple[float, int]],
    cfg: EventLoaderConfig,
    counts: dict[str, int],
) -> tuple[float, int, float]:
    y_h2 = 0.5
    src_h2 = SRC_AUTOTAG
    w_h2 = 0.0 if cfg.ignore_ambiguous else float(cfg.ambiguous_weight)

    if y_h1 != H1_STATIC:
        return y_h2, src_h2, w_h2

    if int(start) not in rec_ann:
        counts["ambiguous"] += 1
        return y_h2, src_h2, w_h2

    y_h2, src_h2 = rec_ann[int(start)]
    if np.isclose(y_h2, 1.0):
        w_h2 = float(cfg.manual_weight if src_h2 == SRC_MANUAL else cfg.autotag_weight)
        counts["postural_transition"] += 1
        counts["valid"] += 1
    elif np.isclose(y_h2, 0.0):
        w_h2 = float(cfg.manual_weight if src_h2 == SRC_MANUAL else cfg.autotag_weight)
        counts["stable_posture"] += 1
        counts["valid"] += 1
    else:
        y_h2 = 0.5
        counts["ambiguous"] += 1
        w_h2 = 0.0 if cfg.ignore_ambiguous else float(cfg.ambiguous_weight)

    if not np.isclose(y_h2, 0.5):
        counts["manual" if src_h2 == SRC_MANUAL else "autotag"] += 1

    return float(y_h2), int(src_h2), float(w_h2)

def _load_sisfall(
    dataset_root: Path,
    cache_dir: Path,
    cfg: EventLoaderConfig,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    from preprocess.preprocess_sisfall import load_sisfall_3class

    if verbose:
        print("[sisfall:data] Loading raw SisFall data.")
    x, y, meta, skipped = load_sisfall_3class(
        root=dataset_root,
        acc=cfg.acc,
        dtype=np.dtype(cfg.dtype).type,
        cache_dir=cache_dir,
        refresh=False,
    )
    return x, y, meta, int(skipped)

def _load_or_build_posture_tags(
    dataset_root: Path,
    label_dir: Path,
    result_dir: Path,
    cache_dir: Path,
    run_name: str,
    cfg: EventLoaderConfig,
    posture_cfg: Any | None,
    refresh: bool,
    verbose: bool,
) -> dict[str, Any]:
    from labeling.sisfall_posture_tagging import TaggingConfig, run_posture_tagging

    if posture_cfg is None:
        posture_cfg = TaggingConfig(acc=cfg.acc, dtype=cfg.dtype)
    if verbose:
        print("[sisfall:posture] Loading or building posture-refinement labels.")
    return run_posture_tagging(
        dataset_root=dataset_root,
        label_dir=label_dir,
        result_dir=result_dir,
        cache_dir=cache_dir,
        cfg=posture_cfg,
        run_name=run_name,
        refresh=refresh,
        save_plot=False,
        verbose=verbose,
    )

def _metadata_template(
    dataset_root: Path,
    label_dir: Path,
    result_dir: Path,
    run_name: str,
    cfg: EventLoaderConfig,
) -> dict[str, Any]:
    return {
        "version": 1,
        "run_name": str(run_name),
        "dataset_root": str(dataset_root),
        "label_dir": str(label_dir),
        "result_dir": str(result_dir),
        "config": u.json_ready(asdict(cfg)),
    }

def _cache_is_valid(root: Path, meta_path: Path, expected: dict[str, Any]) -> bool:
    return u.event_cache_is_valid(
        root,
        meta_path,
        expected,
        split_names=SPLIT_NAMES,
        array_keys=ARRAY_KEYS,
        metadata_keys=("version", "run_name", "config"),
    )

__all__ = [
    "ARRAY_KEYS", "EventLoaderConfig", "H1_DYNAMIC", "H1_FALL", "H1_STATIC",
    "MultiTaskEventDataset", "SRC_AUTOTAG", "SRC_MANUAL", "build_event_split",
    "count_event_windows",
    "build_head2_annotation_map", "build_sisfall_training_loaders",
    "estimate_head2_pos_weight", "load_event_splits", "load_or_build_sisfall_event_splits",
    "make_event_loaders", "normalize_h2", "record_key", "save_event_splits",
    "stratified_file_split", "summarize_split", "SingleHeadSisFallDataset",
    "make_single_head_event_loaders",
]
