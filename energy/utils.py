#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for SCN energy estimation."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from preprocess.utils import json_ready

ENERGY = {
    "E_DENSE_MAC": 1.000,
    "E_SNN_AC": 0.175,
    "E_LIF": 0.383,
    "E_ISPAD": 0.107,
    "E_WSPAD": 1.712,
}


def discover_run_dirs(runs_root: str | Path | None = None, run_dir: str | Path | None = None) -> list[Path]:
    if run_dir is not None:
        path = Path(run_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {path}")
        return [path]

    if runs_root is None:
        raise ValueError("Provide either runs_root or run_dir.")

    root = Path(runs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Runs root not found: {root}")

    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "config.json").is_file())


def save_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(str(key))
                seen.add(str(key))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})
    return path


def load_checkpoint(model: nn.Module, ckpt_path: str | Path, device: torch.device) -> Any:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return ckpt


def count_nonzero_numel(x: torch.Tensor | None) -> tuple[int, int]:
    if x is None:
        return 0, 0
    with torch.no_grad():
        return int(torch.count_nonzero(x).item()), int(x.numel())


def first_module_of_type(module: nn.Module, cls: type[nn.Module]) -> nn.Module | None:
    return next((m for m in module.modules() if isinstance(m, cls)), None)


def first_module_by_class_name(module: nn.Module, class_name: str) -> nn.Module | None:
    return next((m for m in module.modules() if m.__class__.__name__ == class_name), None)


def scalar_or_string(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) or v is None for v in value):
        return str(tuple(value))
    return None


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _csv_value(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, (dict, list, tuple, set)):
        return str(value)
    return value
