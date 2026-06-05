"""Shared utilities for event-based dataset loaders."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from preprocess.utils import (
    class_counts,
    json_ready,
    load_json,
    named_class_counts,
    save_json,
    stratified_split_indices,
)

PathLike = str | os.PathLike[str]

__all__ = [
    "balanced_class_weights",
    "basename_no_ext",
    "canon",
    "class_counts",
    "ensure_project_root_on_path",
    "event_cache_is_valid",
    "load_event_splits",
    "load_json",
    "named_class_counts",
    "obj_get",
    "safe_rmtree",
    "save_event_splits",
    "save_json",
    "stratified_split_indices",
    "weighted_random_sampler",
]


def ensure_project_root_on_path(anchor_file: PathLike | None = None) -> None:
    anchor = Path(anchor_file or __file__).resolve()
    root = anchor.parents[1] if len(anchor.parents) >= 2 else Path.cwd()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def save_event_splits(
    splits: dict[str, dict[str, np.ndarray]],
    root: PathLike,
    *,
    split_names: Sequence[str],
    array_keys: Sequence[str],
    metadata: dict[str, Any] | None = None,
    verbose: bool = True,
) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for split_name in split_names:
        split_dir = root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for key in array_keys:
            array = np.asarray(splits[split_name][key])
            path = split_dir / f"{key}.npy"
            np.save(path, array)

            if verbose:
                print(f"[cache:save] {path} | shape={array.shape} | dtype={array.dtype}")

    if metadata is not None:
        save_json(root / "metadata.json", metadata)


def load_event_splits(
    root: PathLike,
    *,
    split_names: Sequence[str],
    array_keys: Sequence[str],
    mmap_mode: str | None = None,
    mmap_keys: Iterable[str] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    root = Path(root)
    mmap_key_set = None if mmap_keys is None else set(mmap_keys)
    splits: dict[str, dict[str, np.ndarray]] = {}

    for split_name in split_names:
        split_dir = root / split_name
        split: dict[str, np.ndarray] = {}

        for key in array_keys:
            use_mmap = mmap_key_set is None or key in mmap_key_set
            split[key] = np.load(
                split_dir / f"{key}.npy",
                mmap_mode=mmap_mode if use_mmap else None,
            )

        splits[split_name] = split

        if verbose:
            preview = ", ".join(f"{key}={np.asarray(split[key]).shape}" for key in array_keys[:3])
            print(f"[cache:load] {split_name}: {preview}")

    return splits


def _normalize_cache_value(value: Any) -> Any:
    value = json_ready(value)

    if isinstance(value, dict):
        return {str(key): _normalize_cache_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_normalize_cache_value(item) for item in value]

    if isinstance(value, str):
        return value.replace("\\", "/").rstrip("/")

    return value


def event_cache_is_valid(
    root: PathLike,
    meta_path: PathLike,
    expected: dict[str, Any],
    *,
    split_names: Sequence[str],
    array_keys: Sequence[str],
    metadata_keys: Sequence[str] = ("config",),
) -> bool:
    root = Path(root)
    meta_path = Path(meta_path)

    if not root.is_dir() or not meta_path.is_file():
        return False

    try:
        current_meta = _normalize_cache_value(load_json(meta_path))
        expected_meta = _normalize_cache_value(expected)
    except Exception:
        return False

    if any(current_meta.get(key) != expected_meta.get(key) for key in metadata_keys):
        return False

    return all(
        (root / split_name / f"{key}.npy").is_file()
        for split_name in split_names
        for key in array_keys
    )


def _remove_readonly(func: Any, path: str, _: Any) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    func(path)


def safe_rmtree(path: PathLike, *, retries: int = 5, delay: float = 0.5) -> None:
    path = Path(path)

    if not path.exists():
        return

    attempts = max(1, int(retries))
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            shutil.rmtree(path, onerror=_remove_readonly)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(float(delay))

    raise PermissionError(
        f"Could not remove cache directory '{path}'. On Windows, this usually "
        "means that OneDrive, File Explorer, a notebook, TensorBoard, or another "
        "Python process still has cache files open. Close those processes or move "
        "--cache-dir outside OneDrive and try again."
    ) from last_error


def balanced_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=int(n_classes)).astype(np.float64)

    if np.any(counts <= 0):
        raise ValueError(f"Cannot estimate class weights with empty classes: counts={counts.tolist()}")

    weights = counts.sum() / (int(n_classes) * counts)
    return weights.astype(np.float32)


def weighted_random_sampler(y: np.ndarray, n_classes: int) -> Any:
    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
    except Exception as exc:
        raise ImportError("PyTorch is required to create a WeightedRandomSampler.") from exc

    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=int(n_classes)).astype(np.float64)

    if np.any(counts <= 0):
        raise ValueError(f"Cannot use weighted sampling with empty classes: counts={counts.tolist()}")

    sample_weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=labels.size, replacement=True)


def obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    if hasattr(obj, key):
        return getattr(obj, key)

    try:
        dtype = getattr(obj, "dtype", None)
        if dtype is not None and dtype.names is not None and key in dtype.names:
            value = obj[key]
            return value.item() if np.ndim(value) == 0 else value
    except Exception:
        return default

    return default


def basename_no_ext(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    return Path(str(value)).stem


def canon(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    return str(value).strip().upper()