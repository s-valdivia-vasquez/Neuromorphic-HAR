#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing utilities for the UCI HAR dataset.

This module reads the raw UCI HAR inertial-signal text files and converts them
into NumPy arrays suitable for the neuromorphic IMU pipeline used in this
repository.

Credits
-------
The original UCI HAR preprocessing structure was adapted from the SNN_HAR
repository by Intelligent Computing Lab / Panda Lab:

    https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR

This version was rewritten for this repository with the following changes:

1. It uses only 6 inertial channels instead of the original 9-channel setup.
   The selected channels are:

       body_gyro_x, body_gyro_y, body_gyro_z,
       total_acc_x, total_acc_y, total_acc_z

   UCI HAR includes both body_acc_* and total_acc_* signals. Using both groups
   gives 6 acceleration channels, which is redundant for this project because
   total_acc already provides the tri-axial accelerometer measurement used as
   the acceleration input, while body_acc is a derived acceleration component.
   The final representation therefore keeps one accelerometer triplet and one
   gyroscope triplet.

2. It returns arrays with shape (N, 128, 6) by default, preserving the temporal
   dimension and channel dimension explicitly.
3. It provides deterministic NumPy-only train/validation/test splitting.
4. It stores an optional compressed cache in data/ucihar_cache/.

Label convention
----------------
Activity labels are converted from the original UCI HAR range [1, 6] to
zero-based labels [0, 5]. Subject IDs are also converted from [1, 30] to
zero-based domain IDs [0, 29], matching the domain convention used in SNN_HAR.
"""

from __future__ import annotations

import argparse
from email import parser
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from preprocess.utils import class_counts, named_class_counts, stratified_split_indices, train_val_split_indices

Array = np.ndarray
CaseName = Literal["official", "random", "subject", "subject_large"]

ACTIVITY_LABELS: dict[int, str] = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

DEFAULT_SIGNAL_TYPES: tuple[str, ...] = (
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_",
)

DEFAULT_SMALL_SUBJECTS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_ALL_SUBJECTS: tuple[int, ...] = tuple(range(30))


@dataclass(frozen=True)
class UCIHARRawData:

    X_train: Array
    y_train: Array
    d_train: Array
    X_test: Array
    y_test: Array
    d_test: Array
    channel_names: tuple[str, ...]
    root: Path

    @property
    def X_all(self) -> Array:
        return np.concatenate((self.X_train, self.X_test), axis=0)

    @property
    def y_all(self) -> Array:
        return np.concatenate((self.y_train, self.y_test), axis=0)

    @property
    def d_all(self) -> Array:
        return np.concatenate((self.d_train, self.d_test), axis=0)


@dataclass(frozen=True)
class UCIHARSplits:
    """Prepared train/validation/test arrays."""

    X_train: Array
    y_train: Array
    d_train: Array
    X_val: Array
    y_val: Array
    d_val: Array
    X_test: Array
    y_test: Array
    d_test: Array
    meta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation for simple downstream use."""
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "d_train": self.d_train,
            "X_val": self.X_val,
            "y_val": self.y_val,
            "d_val": self.d_val,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "d_test": self.d_test,
            "meta": self.meta,
        }


# -----------------------------------------------------------------------------
# Path and raw-file loading
# -----------------------------------------------------------------------------


def resolve_ucihar_root(
    data_root: str | Path = "data",
    dataset_dir: str = "UCI HAR Dataset",
) -> Path:
    """Resolve and validate the UCI HAR dataset root directory."""
    data_root = Path(data_root)
    root = data_root / dataset_dir
    if root.is_dir():
        return root

    # Also allow passing the dataset folder directly through data_root.
    if data_root.name == dataset_dir and data_root.is_dir():
        return data_root

    raise FileNotFoundError(
        f"UCI HAR dataset directory not found. Tried: {root} and {data_root}. "
        "Expected a folder such as data/UCI HAR Dataset/."
    )


def _signal_file_paths(root: Path, split: Literal["train", "test"], signal_types: Sequence[str]) -> list[Path]:
    signal_dir = root / split / "Inertial Signals"
    paths = [signal_dir / f"{name}{split}.txt" for name in signal_types]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        msg = "Missing UCI HAR inertial signal files:\n" + "\n".join(missing)
        raise FileNotFoundError(msg)
    return paths


def _label_path(root: Path, split: Literal["train", "test"]) -> Path:
    path = root / split / f"y_{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    return path


def _subject_path(root: Path, split: Literal["train", "test"]) -> Path:
    path = root / split / f"subject_{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing subject file: {path}")
    return path


def _read_signal_file(path: Path, dtype: np.dtype[Any] | type = np.float32) -> Array:
    arr = np.loadtxt(path, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != 128:
        raise ValueError(f"Expected {path} to have shape (N, 128). Got {arr.shape}.")
    return np.asarray(arr, dtype=dtype)


def format_data_x(
    files: Sequence[str | Path],
    dtype: np.dtype[Any] | type = np.float32,
) -> Array:
    """
    Read UCI HAR signal files and return X with shape (N, 128, C).

    Each source file has shape (N, 128). Stacking C signal files along the last
    axis yields (N, 128, C), where C is usually 6 in this repository.
    """
    if not files:
        raise ValueError("files must contain at least one signal path.")

    arrays = [_read_signal_file(Path(path), dtype=dtype) for path in files]
    n_samples = {arr.shape[0] for arr in arrays}
    if len(n_samples) != 1:
        shapes = [arr.shape for arr in arrays]
        raise ValueError(f"All signal files must have the same number of samples. Got {shapes}.")

    return np.stack(arrays, axis=-1).astype(dtype, copy=False)


def format_data_y(
    file: str | Path,
    zero_based: bool = True,
    dtype: np.dtype[Any] | type = np.int64,
) -> Array:
    """Read labels or subject IDs and optionally convert them to zero-based IDs."""
    data = np.loadtxt(file, dtype=dtype).reshape(-1)
    if zero_based:
        data = data - 1
    return np.asarray(data, dtype=dtype)


def load_ucihar_split(
    root: str | Path,
    split: Literal["train", "test"],
    signal_types: Sequence[str] = DEFAULT_SIGNAL_TYPES,
    dtype: np.dtype[Any] | type = np.float32,
    zero_based: bool = True,
) -> tuple[Array, Array, Array]:
    """
    Load one official UCI HAR split.

    Returns
    -------
    X:
        Float array with shape (N, 128, 6) by default.
    y:
        Activity labels, zero-based by default.
    d:
        Subject/domain IDs, zero-based by default.
    """
    root = Path(root)
    files = _signal_file_paths(root, split, signal_types)
    X = format_data_x(files, dtype=dtype)
    y = format_data_y(_label_path(root, split), zero_based=zero_based, dtype=np.int64)
    d = format_data_y(_subject_path(root, split), zero_based=zero_based, dtype=np.int64)

    if X.shape[0] != y.size or X.shape[0] != d.size:
        raise ValueError(
            f"Inconsistent sample counts for split={split}: "
            f"X={X.shape[0]}, y={y.size}, d={d.size}"
        )
    return X, y, d


def _cache_path(cache_dir: str | Path, signal_types: Sequence[str], dtype: np.dtype[Any] | type) -> Path:
    dtype_name = np.dtype(dtype).name
    channel_tag = f"{len(signal_types)}ch"
    return Path(cache_dir) / f"ucihar_{channel_tag}_{dtype_name}.npz"


def load_ucihar_raw(
    data_root: str | Path = "data",
    dataset_dir: str = "UCI HAR Dataset",
    signal_types: Sequence[str] = DEFAULT_SIGNAL_TYPES,
    dtype: np.dtype[Any] | type = np.float32,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
    verbose: bool = True,
) -> UCIHARRawData:
    """Load official UCI HAR train/test files with optional compressed caching."""
    root = resolve_ucihar_root(data_root=data_root, dataset_dir=dataset_dir)
    if cache_dir is None:
        cache_dir = root.parent / "ucihar_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache = _cache_path(cache_dir, signal_types, dtype)
    expected_channels = tuple(signal_types)

    if cache.is_file() and not refresh_cache:
        cached = np.load(cache, allow_pickle=False)
        cached_channels = tuple(str(v) for v in cached["channel_names"].tolist())
        if cached_channels == expected_channels:
            if verbose:
                print(f"[UCIHAR] Loading cache: {cache}")
            return UCIHARRawData(
                X_train=cached["X_train"],
                y_train=cached["y_train"].astype(np.int64, copy=False),
                d_train=cached["d_train"].astype(np.int64, copy=False),
                X_test=cached["X_test"],
                y_test=cached["y_test"].astype(np.int64, copy=False),
                d_test=cached["d_test"].astype(np.int64, copy=False),
                channel_names=cached_channels,
                root=root,
            )

    if verbose:
        print(f"[UCIHAR] Reading raw files from: {root}")
        print(f"[UCIHAR] Channels: {', '.join(signal_types)}")

    X_train, y_train, d_train = load_ucihar_split(
        root=root,
        split="train",
        signal_types=signal_types,
        dtype=dtype,
        zero_based=True,
    )
    X_test, y_test, d_test = load_ucihar_split(
        root=root,
        split="test",
        signal_types=signal_types,
        dtype=dtype,
        zero_based=True,
    )

    np.savez_compressed(
        cache,
        X_train=X_train,
        y_train=y_train,
        d_train=d_train,
        X_test=X_test,
        y_test=y_test,
        d_test=d_test,
        channel_names=np.asarray(signal_types, dtype="U32"),
    )

    if verbose:
        print(f"[UCIHAR] Cache written: {cache}")
        print(f"[UCIHAR] train X={X_train.shape}, y={y_train.shape}, d={d_train.shape}")
        print(f"[UCIHAR] test  X={X_test.shape}, y={y_test.shape}, d={d_test.shape}")

    return UCIHARRawData(
        X_train=X_train,
        y_train=y_train,
        d_train=d_train,
        X_test=X_test,
        y_test=y_test,
        d_test=d_test,
        channel_names=tuple(signal_types),
        root=root,
    )

def subset_arrays(X: Array, y: Array, d: Array, idx: Array) -> tuple[Array, Array, Array]:
    """Index X/y/d with the same integer index vector."""
    idx = np.asarray(idx, dtype=np.int64)
    return X[idx], y[idx], d[idx]


def select_subjects(X: Array, y: Array, d: Array, subjects: Iterable[int]) -> tuple[Array, Array, Array]:
    """Select samples whose zero-based subject/domain ID is in subjects."""
    subject_set = np.asarray(list(subjects), dtype=np.int64)
    if subject_set.size == 0:
        raise ValueError("subjects must contain at least one subject ID.")
    mask = np.isin(d, subject_set)
    return X[mask], y[mask], d[mask]


# -----------------------------------------------------------------------------
# Public preprocessing API
# -----------------------------------------------------------------------------


def add_height_axis(X: Array) -> Array:
    """
    Convert X from (N, 128, 6) to (N, 128, 1, 6).

    This mirrors the shape used in the original SNN_HAR DataLoader while keeping
    the default preprocessing output simpler for event encoding and theta search.
    """
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1:] != (128, 6):
        raise ValueError(f"Expected X with shape (N, 128, 6). Got {X.shape}.")
    return X.reshape(X.shape[0], 128, 1, 6)


def maybe_add_height_axis(result: UCIHARSplits, model_shape: bool) -> UCIHARSplits:
    if not model_shape:
        return result
    return UCIHARSplits(
        X_train=add_height_axis(result.X_train),
        y_train=result.y_train,
        d_train=result.d_train,
        X_val=add_height_axis(result.X_val),
        y_val=result.y_val,
        d_val=result.d_val,
        X_test=add_height_axis(result.X_test),
        y_test=result.y_test,
        d_test=result.d_test,
        meta={**result.meta, "model_shape": True},
    )


def load_domain_data(
    domain_idx: int | str,
    data_root: str | Path = "data",
    dataset_dir: str = "UCI HAR Dataset",
    signal_types: Sequence[str] = DEFAULT_SIGNAL_TYPES,
    dtype: np.dtype[Any] | type = np.float32,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> tuple[Array, Array, Array]:
    """
    Load all samples from one zero-based subject/domain ID.

    This function keeps the domain-oriented behavior of the original SNN_HAR
    preprocessing but returns NumPy arrays only.
    """
    domain = int(domain_idx)
    raw = load_ucihar_raw(
        data_root=data_root,
        dataset_dir=dataset_dir,
        signal_types=signal_types,
        dtype=dtype,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )
    X, y, d = select_subjects(raw.X_all, raw.y_all, raw.d_all, [domain])
    if X.size == 0:
        raise ValueError(f"No samples found for zero-based subject/domain {domain}.")
    return X, y, d


def preprocess_ucihar(
    data_root: str | Path = "data",
    dataset_dir: str = "UCI HAR Dataset",
    case: CaseName = "official",
    target_domain: int | str = 0,
    split: Sequence[float] = (0.8, 0.1, 0.1),
    val_ratio: float = 0.1,
    seed: int = 0,
    signal_types: Sequence[str] = DEFAULT_SIGNAL_TYPES,
    dtype: np.dtype[Any] | type = np.float32,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
    model_shape: bool = False,
    verbose: bool = True,
) -> UCIHARSplits:
    """
    Prepare UCI HAR arrays for this repository.

    Parameters
    ----------
    data_root:
        Root data directory. By default the dataset is expected at
        data/UCI HAR Dataset/.
    dataset_dir:
        Name of the UCI HAR folder inside data_root.
    case:
        Split strategy:

        - "official": use the original UCI HAR train/test split and create a
          validation subset from the official train split.
        - "random": concatenate official train+test, then make a deterministic
          stratified train/validation/test split.
        - "subject": use subjects 0..4 as the domain pool, with target_domain
          as test and the remaining subjects split into train/validation.
        - "subject_large": use subjects 0..29 as the domain pool, with
          target_domain as test and the remaining subjects split into
          train/validation.

    target_domain:
        Zero-based subject/domain ID used by "subject" and "subject_large".
    split:
        Train/validation/test fractions for "random".
    val_ratio:
        Validation fraction for "official", "subject", and "subject_large".
    model_shape:
        If False, arrays are returned as (N, 128, 6). If True, arrays are
        returned as (N, 128, 1, 6), matching the original SNN_HAR loader shape.

    Returns
    -------
    UCIHARSplits
        Dataclass with X/y/domain arrays and metadata.
    """
    case = str(case).lower()
    if case not in {"official", "random", "subject", "subject_large"}:
        raise ValueError("case must be one of: official, random, subject, subject_large.")

    raw = load_ucihar_raw(
        data_root=data_root,
        dataset_dir=dataset_dir,
        signal_types=signal_types,
        dtype=dtype,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )

    if case == "official":
        train_idx, val_idx = train_val_split_indices(raw.y_train, val_ratio=val_ratio, seed=seed)
        X_train, y_train, d_train = subset_arrays(raw.X_train, raw.y_train, raw.d_train, train_idx)
        X_val, y_val, d_val = subset_arrays(raw.X_train, raw.y_train, raw.d_train, val_idx)
        X_test, y_test, d_test = raw.X_test, raw.y_test, raw.d_test

    elif case == "random":
        train_idx, val_idx, test_idx = stratified_split_indices(raw.y_all, split=split, seed=seed)
        X_train, y_train, d_train = subset_arrays(raw.X_all, raw.y_all, raw.d_all, train_idx)
        X_val, y_val, d_val = subset_arrays(raw.X_all, raw.y_all, raw.d_all, val_idx)
        X_test, y_test, d_test = subset_arrays(raw.X_all, raw.y_all, raw.d_all, test_idx)

    else:
        target = int(target_domain)
        subject_pool = DEFAULT_SMALL_SUBJECTS if case == "subject" else DEFAULT_ALL_SUBJECTS
        if target not in subject_pool:
            raise ValueError(f"target_domain={target} is not part of the selected {case} subject pool.")

        source_subjects = tuple(s for s in subject_pool if s != target)
        X_source, y_source, d_source = select_subjects(raw.X_all, raw.y_all, raw.d_all, source_subjects)
        X_test, y_test, d_test = select_subjects(raw.X_all, raw.y_all, raw.d_all, [target])

        train_idx, val_idx = train_val_split_indices(y_source, val_ratio=val_ratio, seed=seed)
        X_train, y_train, d_train = subset_arrays(X_source, y_source, d_source, train_idx)
        X_val, y_val, d_val = subset_arrays(X_source, y_source, d_source, val_idx)

    meta = {
        "dataset": "ucihar",
        "case": case,
        "target_domain": int(target_domain) if case in {"subject", "subject_large"} else None,
        "data_root": str(Path(data_root)),
        "dataset_root": str(raw.root),
        "channel_names": tuple(raw.channel_names),
        "activity_labels": ACTIVITY_LABELS,
        "label_zero_based": True,
        "domain_zero_based": True,
        "x_shape": "(N, 128, 6)" if not model_shape else "(N, 128, 1, 6)",
        "split": tuple(float(v) for v in split),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
    }

    result = UCIHARSplits(
        X_train=np.asarray(X_train, dtype=dtype),
        y_train=np.asarray(y_train, dtype=np.int64),
        d_train=np.asarray(d_train, dtype=np.int64),
        X_val=np.asarray(X_val, dtype=dtype),
        y_val=np.asarray(y_val, dtype=np.int64),
        d_val=np.asarray(d_val, dtype=np.int64),
        X_test=np.asarray(X_test, dtype=dtype),
        y_test=np.asarray(y_test, dtype=np.int64),
        d_test=np.asarray(d_test, dtype=np.int64),
        meta=meta,
    )

    result = maybe_add_height_axis(result, model_shape=model_shape)

    if verbose:
        print_summary(result)

    return result

def compute_class_weights(y: Array, scale: float = 100.0) -> Array:
    """
    Compute inverse-frequency class weights.

    This replaces the get_sample_weights/data_preprocess_utils dependency from
    the original code. The returned vector is indexed by class ID.
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if y.size == 0:
        raise ValueError("y must contain at least one label.")

    max_label = int(y.max())
    weights = np.zeros((max_label + 1,), dtype=np.float64)
    for label in np.unique(y):
        count = int(np.sum(y == label))
        weights[int(label)] = float(scale) / max(count, 1)
    return weights


def compute_sample_weights(y: Array, class_weights: Array | None = None, scale: float = 100.0) -> Array:
    """Return one weight per sample, suitable for a weighted sampler."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if class_weights is None:
        class_weights = compute_class_weights(y, scale=scale)
    return np.asarray(class_weights, dtype=np.float64)[y]


def print_summary(result: UCIHARSplits) -> None:
    """Print shapes and per-class distributions for quick inspection."""
    print("[UCIHAR] Prepared splits")
    print(f"  case: {result.meta.get('case')}")
    print(f"  channels: {', '.join(result.meta.get('channel_names', []))}")
    print(
        f"  X_train={result.X_train.shape}, "
        f"y_train={result.y_train.shape}, "
        f"counts={named_class_counts(result.y_train, ACTIVITY_LABELS)}"
    )
    print(
        f"  X_val  ={result.X_val.shape}, "
        f"y_val  ={result.y_val.shape}, "
        f"counts={named_class_counts(result.y_val, ACTIVITY_LABELS)}"
    )
    print(
        f"  X_test ={result.X_test.shape}, "
        f"y_test ={result.y_test.shape}, "
        f"counts={named_class_counts(result.y_test, ACTIVITY_LABELS)}"
    )


def save_preprocessed_npz(result: UCIHARSplits, output_path: str | Path) -> Path:
    """Save prepared splits to a compressed NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=result.X_train,
        y_train=result.y_train,
        d_train=result.d_train,
        X_val=result.X_val,
        y_val=result.y_val,
        d_val=result.d_val,
        X_test=result.X_test,
        y_test=result.y_test,
        d_test=result.d_test,
        meta_json=np.asarray(json.dumps(result.meta, ensure_ascii=False)),
    )
    return output_path


def load_preprocessed_npz(path: str | Path) -> UCIHARSplits:
    """Load a compressed NPZ generated by save_preprocessed_npz."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"].tolist()))
    return UCIHARSplits(
        X_train=data["X_train"],
        y_train=data["y_train"].astype(np.int64, copy=False),
        d_train=data["d_train"].astype(np.int64, copy=False),
        X_val=data["X_val"],
        y_val=data["y_val"].astype(np.int64, copy=False),
        d_val=data["d_val"].astype(np.int64, copy=False),
        X_test=data["X_test"],
        y_test=data["y_test"].astype(np.int64, copy=False),
        d_test=data["d_test"].astype(np.int64, copy=False),
        meta=meta,
    )


# Optional CLI for quick verification

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess UCI HAR 6-channel arrays.")
    parser.add_argument("--data-root", default="data", help="Root data directory.")
    parser.add_argument("--dataset-dir", default="UCI HAR Dataset", help="UCI HAR folder name inside data-root.")
    parser.add_argument("--case", default="official", choices=["official", "random", "subject", "subject_large"])
    parser.add_argument("--target-domain", default=0, type=int, help="Zero-based subject ID for subject cases.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--split",nargs=3,default=(0.8, 0.1, 0.1),type=float,metavar=("TRAIN", "VAL", "TEST"),help="Only used with --case random. Fractions for train/val/test after merging official train+test.",)
    parser.add_argument("--val-ratio",default=0.1,type=float,help="Only used with --case official, subject, and subject_large. Fraction of source training data used for validation.",)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--model-shape", action="store_true", help="Return/save X as (N,128,1,6) instead of (N,128,6).")
    parser.add_argument("--save", default="", help="Optional output NPZ path for the prepared splits.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = preprocess_ucihar(
        data_root=args.data_root,
        dataset_dir=args.dataset_dir,
        case=args.case,
        target_domain=args.target_domain,
        seed=args.seed,
        val_ratio=args.val_ratio,
        split=args.split,
        refresh_cache=args.refresh_cache,
        model_shape=args.model_shape,
        verbose=True,
    )
    if args.save:
        out = save_preprocessed_npz(result, args.save)
        print(f"[UCIHAR] Saved prepared splits to: {out}")


if __name__ == "__main__":
    main()
