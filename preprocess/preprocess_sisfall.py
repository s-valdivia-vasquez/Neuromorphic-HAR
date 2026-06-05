"""SisFall dataset loader utilities with optional caching."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np


ACC_COLS = {
    "ADXL345": [0, 1, 2],
    "MMA8451Q": [6, 7, 8],
}
GYR_COLS = [3, 4, 5]

DYNAMIC_CODES = {"D01", "D02", "D03", "D04", "D05", "D06", "D18", "D19"}
STATIC_CODES = {"D07", "D08", "D09", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17"}
LABELS = {
    0: "FALL",
    1: "NORMAL_DYNAMIC",
    2: "NORMAL_STATIC",
}

FILE_RE = re.compile(r"^(D\d{2}|F\d{2})_(SA\d{2}|SE\d{2})_(R\d{2})\.txt$", re.IGNORECASE)
INT_RE = re.compile(r"[-+]?\d+")


def load_sisfall_3class(
    root: str | os.PathLike[str],
    acc: str = "ADXL345",
    dtype: np.dtype[Any] | type | str = np.int32,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    """Load SisFall TXT files as fall, dynamic ADL, and static ADL classes."""
    root = Path(root)
    acc = acc.upper()
    dt = np.dtype(dtype)

    if acc not in ACC_COLS:
        raise ValueError(f"Unknown accelerometer '{acc}'. Expected one of {sorted(ACC_COLS)}.")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    if cache_dir is None:
        cache_dir = root.parent / "sisfall_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"sisfall_3class_{acc.lower()}_{dt.name}.npz"
    if cache_path.exists() and not refresh:
        d = np.load(cache_path, allow_pickle=True)
        x = d["x"]
        y = d["y"].astype(np.int64)
        meta = list(d["meta"].tolist())
        skipped = int(d["skipped"])
        return x, y, meta, skipped

    xs: list[np.ndarray] = []
    ys: list[int] = []
    meta: list[dict[str, Any]] = []
    skipped = 0
    cols = ACC_COLS[acc] + GYR_COLS

    # Traverse files and keep only valid SisFall recordings.
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower() == "desktop.ini" or not fn.lower().endswith(".txt"):
                continue

            mt = FILE_RE.match(fn)
            if mt is None:
                skipped += 1
                continue

            code = mt.group(1).upper()
            if code.startswith("F"):
                lab = 0
            elif code in DYNAMIC_CODES:
                lab = 1
            elif code in STATIC_CODES:
                lab = 2
            else:
                skipped += 1
                continue

            fp = Path(dp) / fn
            rows: list[list[int]] = []

            # Parse the first nine integer fields from each valid line.
            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        nums = INT_RE.findall(line)
                        if len(nums) >= 9:
                            rows.append([int(v) for v in nums[:9]])
            except OSError:
                skipped += 1
                continue

            if not rows:
                skipped += 1
                continue

            arr = np.asarray(rows, dtype=dt)
            if arr.ndim != 2 or arr.shape[1] != 9:
                skipped += 1
                continue

            x = arr[:, cols]
            xs.append(x)
            ys.append(lab)
            meta.append(
                {
                    "file": str(fp),
                    "code": code,
                    "subject": mt.group(2).upper(),
                    "trial": mt.group(3).upper(),
                    "label_id": int(lab),
                    "label": LABELS[int(lab)],
                    "acc_sensor": acc,
                    "channels": cols,
                    "T": int(x.shape[0]),
                }
            )

    x_obj = np.asarray(xs, dtype=object)
    y_arr = np.asarray(ys, dtype=np.int64)
    meta_obj = np.asarray(meta, dtype=object)

    np.savez_compressed(
        cache_path,
        x=x_obj,
        y=y_arr,
        meta=meta_obj,
        skipped=np.asarray(skipped),
    )

    return x_obj, y_arr, meta, skipped
