"""Signal interpolation utilities for fixed-length preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from scipy.interpolate import interp1d as _interp1d
except Exception:  # pragma: no cover - optional dependency
    _interp1d = None

InterpMethod = Literal["linear", "zoh", "cubic"]


@dataclass(frozen=True)
class ResamplePlan:
    """Precomputed time grid and weights for repeated interpolation calls."""

    in_len: int
    out_len: int
    method: str
    t_in: np.ndarray
    t_out: np.ndarray
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    alpha: np.ndarray | None = None
    index: np.ndarray | None = None


def has_cubic_support() -> bool:
    """Return True when SciPy is available for cubic interpolation."""
    return _interp1d is not None


def derive_output_length(in_len: int, factor: float, min_len: int = 2) -> int:
    """Return the target length obtained from an interpolation factor."""
    if int(in_len) < 1:
        raise ValueError("in_len must be >= 1.")
    if float(factor) <= 0.0:
        raise ValueError("factor must be > 0.")
    if int(min_len) < 1:
        raise ValueError("min_len must be >= 1.")
    return max(int(round(float(in_len) * float(factor))), int(min_len))


def make_resample_plan(
    in_len: int,
    out_len: int,
    method: InterpMethod = "linear",
    dtype: np.dtype | type = np.float32,
) -> ResamplePlan:
    """Build a reusable interpolation plan for signals with fixed length."""
    in_len = int(in_len)
    out_len = int(out_len)
    method = str(method).lower()

    if method not in {"linear", "zoh", "cubic"}:
        raise ValueError("method must be 'linear', 'zoh', or 'cubic'.")
    if in_len < 2:
        raise ValueError("in_len must be >= 2.")
    if out_len < 1:
        raise ValueError("out_len must be >= 1.")

    t_in = np.arange(in_len, dtype=dtype)
    t_out = np.linspace(0.0, float(in_len - 1), num=out_len, endpoint=True, dtype=dtype)

    if method == "linear":
        left = np.floor(t_out).astype(np.int64)
        right = np.minimum(left + 1, in_len - 1)
        alpha = (t_out - left).astype(dtype)[:, None]
        return ResamplePlan(in_len, out_len, method, t_in, t_out, left=left, right=right, alpha=alpha)

    if method == "zoh":
        idx = np.floor(t_out).astype(np.int64)
        idx = np.clip(idx, 0, in_len - 1)
        return ResamplePlan(in_len, out_len, method, t_in, t_out, index=idx)

    return ResamplePlan(in_len, out_len, method, t_in, t_out)


def make_linear_resample_plan(
    in_len: int,
    out_len: int,
    dtype: np.dtype | type = np.float32,
) -> ResamplePlan:
    """Build a reusable exact linear interpolation plan."""
    return make_resample_plan(in_len=in_len, out_len=out_len, method="linear", dtype=dtype)


def interpolate_signal(
    x: np.ndarray,
    out_len: int | None = None,
    method: InterpMethod = "linear",
    plan: ResamplePlan | None = None,
    factor: float | None = None,
    dtype: np.dtype | type = np.float32,
    return_times: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a 1-D or multi-channel signal to a target length."""
    x0, squeeze = _as_signal(x, dtype=dtype)

    if plan is None:
        if out_len is None:
            if factor is None:
                raise ValueError("out_len, factor, or plan must be provided.")
            out_len = derive_output_length(x0.shape[0], factor)
        plan = make_resample_plan(x0.shape[0], int(out_len), method=method, dtype=dtype)
    elif x0.shape[0] != plan.in_len:
        raise ValueError(f"Signal length {x0.shape[0]} does not match plan.in_len={plan.in_len}.")

    y = _apply_plan(x0, plan, dtype=dtype)
    if squeeze:
        y = y[:, 0]

    if return_times:
        return plan.t_in.copy(), plan.t_out.copy(), y
    return y


def resample_window_linear_exact(x: np.ndarray, plan: ResamplePlan) -> np.ndarray:
    """Apply a precomputed linear plan to one signal window."""
    if plan.method != "linear":
        raise ValueError("plan.method must be 'linear'.")
    return interpolate_signal(x, plan=plan, dtype=np.float32)  # type: ignore[return-value]


def interpolate_to_times(
    x: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
    method: InterpMethod = "linear",
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Interpolate a signal from arbitrary source times to target times."""
    x0, squeeze = _as_signal(x, dtype=dtype)
    t_src = np.asarray(t_src, dtype=dtype).reshape(-1)
    t_dst = np.asarray(t_dst, dtype=dtype).reshape(-1)
    method = str(method).lower()

    if x0.shape[0] != t_src.size:
        raise ValueError("x and t_src must have the same temporal length.")
    if t_src.size < 2:
        raise ValueError("t_src must contain at least two samples.")
    if t_dst.size < 1:
        raise ValueError("t_dst must contain at least one sample.")
    if np.any(np.diff(t_src) < 0):
        raise ValueError("t_src must be sorted in ascending order.")

    if method == "linear":
        y = np.empty((t_dst.size, x0.shape[1]), dtype=dtype)
        for c in range(x0.shape[1]):
            y[:, c] = np.interp(t_dst, t_src, x0[:, c]).astype(dtype)
    elif method == "zoh":
        idx = np.searchsorted(t_src, t_dst, side="right") - 1
        idx = np.clip(idx, 0, t_src.size - 1)
        y = x0[idx].astype(dtype, copy=False)
    elif method == "cubic":
        y = _cubic_interp(x0, t_src, t_dst, dtype=dtype)
    else:
        raise ValueError("method must be 'linear', 'zoh', or 'cubic'.")

    return y[:, 0] if squeeze else y


def interpolation_error(x_ref: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """Compute basic reconstruction errors between two aligned signals."""
    a = np.asarray(x_ref, dtype=np.float32)
    b = np.asarray(x_hat, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shapes must match, got {a.shape} and {b.shape}.")

    e = a - b
    den = max(float(np.sum(a * a)), 1e-12)
    return {
        "mae": float(np.mean(np.abs(e))),
        "mse": float(np.mean(e * e)),
        "rel_err": float(np.sum(e * e) / den),
    }


def _as_signal(x: np.ndarray, dtype: np.dtype | type) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=dtype)
    if x.ndim == 1:
        if x.shape[0] < 2:
            raise ValueError("A 1-D signal must contain at least two samples.")
        return x[:, None], True
    if x.ndim == 2:
        if x.shape[0] < 2:
            raise ValueError("A 2-D signal must contain at least two time samples.")
        return x, False
    raise ValueError(f"Signal must have shape (T,) or (T, C), got {x.shape}.")


def _apply_plan(x: np.ndarray, plan: ResamplePlan, dtype: np.dtype | type) -> np.ndarray:
    if plan.method == "linear":
        if plan.left is None or plan.right is None or plan.alpha is None:
            raise ValueError("Invalid linear resampling plan.")
        y0 = x[plan.left]
        y1 = x[plan.right]
        y = y0 + plan.alpha * (y1 - y0)
        return y.astype(dtype, copy=False)

    if plan.method == "zoh":
        if plan.index is None:
            raise ValueError("Invalid zero-order hold resampling plan.")
        return x[plan.index].astype(dtype, copy=False)

    if plan.method == "cubic":
        return _cubic_interp(x, plan.t_in, plan.t_out, dtype=dtype)

    raise ValueError("method must be 'linear', 'zoh', or 'cubic'.")


def _cubic_interp(
    x: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
    dtype: np.dtype | type,
) -> np.ndarray:
    if _interp1d is None:
        raise RuntimeError("Cubic interpolation requires scipy.")

    kind = "cubic" if x.shape[0] >= 4 else "linear"
    y = np.empty((t_dst.size, x.shape[1]), dtype=dtype)
    for c in range(x.shape[1]):
        f = _interp1d(
            t_src,
            x[:, c],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        y[:, c] = np.asarray(f(t_dst), dtype=dtype)
    return y


__all__ = [
    "InterpMethod",
    "ResamplePlan",
    "derive_output_length",
    "has_cubic_support",
    "interpolate_signal",
    "interpolate_to_times",
    "interpolation_error",
    "make_linear_resample_plan",
    "make_resample_plan",
    "resample_window_linear_exact",
]
