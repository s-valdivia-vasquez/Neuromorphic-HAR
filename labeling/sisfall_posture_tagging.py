"""Window-level posture refinement tagging for SisFall recordings."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import as_strided

LABEL_MAP = {
    0.0: "stable_posture",
    0.5: "ambiguous",
    1.0: "postural_transition",
}


@dataclass(frozen=True)
class TaggingConfig:
    # Sensor sampling rate in Hz.
    fs: float = 200.0 
    # Window length in samples.
    win: int = 410
    # Distance between consecutive windows in samples.
    hop: int = 205
    # Padding step used to provide context near signal boundaries.
    pad_hop: int = 205
    # Number of repeated padding blocks at each boundary.
    pad_rep: int = 2
    # Number of PCA components appended to normalized features.
    pca_k: int = 16
    # Downsampling factor for gravity direction variation features.
    ds_grav: int = 16
    # Logistic regression learning rate.
    lr: float = 0.05
    # Number of optimization steps for each binary classifier.
    steps: int = 2500
    # L2 regularization strength for logistic regression weights.
    l2: float = 2e-3
    # Moving average size applied to output probabilities.
    smooth_k: int = 3
    # Probability threshold for postural transition detection.
    th_pt: float = 0.60
    # Probability threshold for stable posture detection.
    th_sp: float = 0.60
    # Minimum probability gap between both posture classes.
    margin: float = 0.15
    # Maximum probability for both classes before forcing ambiguity.
    amb_ceil: float = 0.55
    # Enable hysteresis over class probabilities.
    use_hyst: bool = True
    # Hysteresis activation threshold for postural transition.
    th_pt_on: float = 0.60
    # Hysteresis deactivation threshold for postural transition.
    th_pt_off: float = 0.40
    # Hysteresis activation threshold for stable posture.
    th_sp_on: float = 0.60
    # Hysteresis deactivation threshold for stable posture.
    th_sp_off: float = 0.40
    # Manual-label aggregation mode for window targets.
    ywin_mode: str = "q"
    # Quantile used when ywin_mode is set to q.
    ywin_q: float = 0.8
    # Gaussian width factor for manual postural-transition intervals.
    sigma_frac: float = 0.40
    # Accelerometer source used from each SisFall sample.
    acc: str = "ADXL345"
    # Numeric dtype used when loading raw SisFall values.
    dtype: str = "int32"



def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if k <= 1:
        return x
    w = np.ones(int(k), np.float32) / float(k)
    return np.convolve(x, w, "same").astype(np.float32)


def _hyst(p: np.ndarray, on: float, off: float) -> np.ndarray:
    y = np.zeros_like(p, np.uint8)
    st = False
    for i, v in enumerate(p):
        if not st and v >= on:
            st = True
        elif st and v <= off:
            st = False
        y[i] = 1 if st else 0
    return y


def _rz(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    med = np.median(a, axis=0, keepdims=True)
    mad = np.median(np.abs(a - med), axis=0, keepdims=True) + 1e-6
    return ((a - med) / (1.4826 * mad)).astype(np.float32, copy=False)


def _meta_key(m: dict[str, Any]) -> str | None:
    for k in ("file", "path", "filepath", "fname", "filename"):
        if k in m:
            return os.path.basename(str(m[k]))
    return None


def _load_labels(label_dir: str | os.PathLike[str]) -> dict[str, dict[str, list[dict[str, float]]]]:
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    labels: dict[str, dict[str, list[dict[str, float]]]] = {}
    for p in sorted(label_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        src = d.get("labels", d)
        labels[p.with_suffix(".txt").name] = {
            "postural_transition": list(
                src.get("postural_transition", src.get("transition", src.get("transitions", [])))
            ),
            "stable_posture": list(src.get("stable_posture", src.get("no_transition", []))),
            "ambiguous": list(src.get("ambiguous", [])),
        }
    return labels


def _window_score(
    n: int,
    intervals: list[dict[str, float]],
    starts: np.ndarray,
    cfg: TaggingConfig,
    mode: str | None = None,
) -> np.ndarray:
    y = np.zeros(int(n), np.float32)
    for it in intervals:
        a = max(0, int(np.floor(float(it["t0"]) * cfg.fs)))
        b = min(int(n), int(np.ceil(float(it["t1"]) * cfg.fs)))
        if b > a:
            y[a:b] = 1.0

    starts = np.asarray(starts, np.int64)
    if starts.size == 0:
        return np.empty(0, np.float32)

    mode = cfg.ywin_mode if mode is None else mode
    if mode == "center":
        c = cfg.win // 2
        return np.asarray([y[s + c] for s in starts], np.float32)
    if mode == "max":
        return np.asarray([y[s : s + cfg.win].max() for s in starts], np.float32)
    if mode == "q":
        q = float(np.clip(cfg.ywin_q, 0.0, 1.0))
        return np.asarray([np.quantile(y[s : s + cfg.win], q) for s in starts], np.float32)

    cs = np.concatenate(([0.0], y.cumsum(dtype=np.float64)))
    return ((cs[starts + cfg.win] - cs[starts]) / float(cfg.win)).astype(np.float32)


def _gaussian_score(
    n: int,
    intervals: list[dict[str, float]],
    starts: np.ndarray,
    cfg: TaggingConfig,
    mode: str | None = None,
) -> np.ndarray:
    t = np.arange(int(n), dtype=np.float32) / float(cfg.fs)
    y = np.zeros(int(n), np.float32)
    for it in intervals:
        t0, t1 = float(it["t0"]), float(it["t1"])
        if t1 <= t0:
            continue
        c = 0.5 * (t0 + t1)
        s = max((t1 - t0) * cfg.sigma_frac, 1.0 / cfg.fs)
        g = np.exp(-0.5 * ((t - c) / s) ** 2).astype(np.float32)
        g *= (t >= t0) & (t <= t1)
        y = np.maximum(y, g)

    tmp = cfg if mode is None else TaggingConfig(**{**asdict(cfg), "ywin_mode": mode})
    return _window_score_from_sample(y, starts, tmp)


def _window_score_from_sample(y: np.ndarray, starts: np.ndarray, cfg: TaggingConfig) -> np.ndarray:
    starts = np.asarray(starts, np.int64)
    if starts.size == 0:
        return np.empty(0, np.float32)
    if cfg.ywin_mode == "center":
        return np.asarray([y[s + cfg.win // 2] for s in starts], np.float32)
    if cfg.ywin_mode == "max":
        return np.asarray([y[s : s + cfg.win].max() for s in starts], np.float32)
    if cfg.ywin_mode == "q":
        q = float(np.clip(cfg.ywin_q, 0.0, 1.0))
        return np.asarray([np.quantile(y[s : s + cfg.win], q) for s in starts], np.float32)
    cs = np.concatenate(([0.0], np.asarray(y, np.float32).cumsum(dtype=np.float64)))
    return ((cs[starts + cfg.win] - cs[starts]) / float(cfg.win)).astype(np.float32)


def _extract_features(x6: np.ndarray, cfg: TaggingConfig) -> tuple[np.ndarray, np.ndarray] | None:
    x6 = np.asarray(x6, np.float32)
    if x6.ndim != 2 or x6.shape[1] != 6 or x6.shape[0] < max(cfg.win, cfg.pad_hop):
        return None

    xp = np.ascontiguousarray(
        np.concatenate(
            [
                np.tile(x6[: cfg.pad_hop], (cfg.pad_rep, 1)),
                x6,
                np.tile(x6[-cfg.pad_hop :], (cfg.pad_rep, 1)),
            ],
            axis=0,
        )
    )
    off = cfg.pad_hop * cfg.pad_rep
    n, c = xp.shape
    nw = 1 + (n - cfg.win) // cfg.hop
    if nw < 3:
        return None

    starts0 = np.arange(nw, dtype=np.int64) * cfg.hop
    xw = as_strided(
        xp,
        shape=(nw, cfg.win, c),
        strides=(xp.strides[0] * cfg.hop, xp.strides[0], xp.strides[1]),
    )

    eps = 1e-8
    bands = ((0.5, 3), (3, 8), (8, 20), (20, 50))
    acc = xw[:, :, :3]
    gyr = xw[:, :, 3:]
    an = np.sqrt(np.sum(acc * acc, axis=2) + eps)
    gn = np.sqrt(np.sum(gyr * gyr, axis=2) + eps)

    eg = np.mean(np.sum(gyr * gyr, axis=2), axis=1)
    jk = np.diff(acc, axis=1) * cfg.fs
    ej = np.mean(np.sum(jk * jk, axis=2), axis=1)

    g = acc / (np.linalg.norm(acc, axis=2, keepdims=True) + eps)
    gd = g[:, :: cfg.ds_grav, :]
    dtv = np.sum(
        np.arccos(np.clip(np.sum(gd[:, 1:] * gd[:, :-1], axis=2), -1.0, 1.0)),
        axis=1,
    )

    gm = np.mean(acc, axis=1)
    gm /= np.linalg.norm(gm, axis=1, keepdims=True) + eps
    dst = np.zeros(nw, np.float32)
    dst[1:] = np.arccos(np.clip(np.sum(gm[1:] * gm[:-1], axis=1), -1.0, 1.0))

    aq = np.quantile(an, [0.25, 0.75], axis=1)
    gq = np.quantile(gn, [0.25, 0.75], axis=1)
    ast = np.stack(
        [an.mean(1), an.std(1), np.sqrt((an * an).mean(1)), an.max(1) - an.min(1), aq[1] - aq[0]],
        axis=1,
    ).astype(np.float32)
    gst = np.stack(
        [gn.mean(1), gn.std(1), np.sqrt((gn * gn).mean(1)), gn.max(1) - gn.min(1), gq[1] - gq[0]],
        axis=1,
    ).astype(np.float32)

    ff = np.fft.rfftfreq(cfg.win, d=1.0 / cfg.fs)
    pa = (np.abs(np.fft.rfft(an, axis=1)) ** 2).astype(np.float32)
    pg = (np.abs(np.fft.rfft(gn, axis=1)) ** 2).astype(np.float32)
    ba = np.stack(
        [pa[:, (ff >= lo) & (ff < hi)].mean(axis=1) if np.any((ff >= lo) & (ff < hi)) else np.zeros(nw) for lo, hi in bands],
        axis=1,
    ).astype(np.float32)
    bg = np.stack(
        [pg[:, (ff >= lo) & (ff < hi)].mean(axis=1) if np.any((ff >= lo) & (ff < hi)) else np.zeros(nw) for lo, hi in bands],
        axis=1,
    ).astype(np.float32)

    f = np.concatenate(
        [_rz(eg)[:, None], _rz(ej)[:, None], _rz(dtv)[:, None], _rz(dst)[:, None], _rz(ast), _rz(gst), _rz(ba), _rz(bg)],
        axis=1,
    ).astype(np.float32)

    pp, cc, nn, ss = f[:-2], f[1:-1], f[2:], starts0[1:-1]
    dp, dn = cc - pp, nn - cc
    nc = np.linalg.norm(cc, ord=1, axis=1) + 1e-6
    npv = np.linalg.norm(pp, ord=1, axis=1) + 1e-6
    nnv = np.linalg.norm(nn, ord=1, axis=1) + 1e-6
    fc = np.concatenate(
        [
            cc,
            dp,
            dn,
            ((cc * pp).sum(axis=1) / (nc * npv))[:, None],
            ((cc * nn).sum(axis=1) / (nc * nnv))[:, None],
            np.sqrt((dp * dp).sum(axis=1) + 1e-6)[:, None],
            np.sqrt((dn * dn).sum(axis=1) + 1e-6)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    starts = ss - off
    keep = (starts >= 0) & (starts + cfg.win <= x6.shape[0])
    if not np.any(keep):
        return None
    return starts[keep].astype(np.int64), fc[keep].astype(np.float32)


def _fit_logreg(z: np.ndarray, y: np.ndarray, cfg: TaggingConfig) -> tuple[np.ndarray, float]:
    y = np.asarray(y, np.float32).reshape(-1)
    w = np.zeros(z.shape[1], np.float32)
    b = 0.0
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    cw = neg / (pos + 1e-6)
    wt = 1.0 + (cw - 1.0) * y
    for _ in range(int(cfg.steps)):
        p = _sigmoid(z @ w + b)
        d = (p - y) * wt
        w -= cfg.lr * ((z.T @ d) / len(y) + cfg.l2 * w)
        b -= cfg.lr * d.mean()
    return w.astype(np.float32), float(b)


def _cache_id(cfg: TaggingConfig, dataset_root: Path, label_dir: Path) -> str:
    msg = {
        "cfg": asdict(cfg),
        "dataset_root": str(dataset_root),
        "label_dir": str(label_dir),
    }
    raw = json.dumps(msg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _load_dataset(
    dataset_root: str | os.PathLike[str],
    acc: str,
    dtype: str,
    cache_dir: str | os.PathLike[str],
    refresh_raw: bool,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    from preprocess.preprocess_sisfall import load_sisfall_3class

    kwargs: dict[str, Any] = {"root": dataset_root, "acc": acc, "dtype": np.dtype(dtype).type}
    sig = inspect.signature(load_sisfall_3class)
    if "cache_dir" in sig.parameters:
        kwargs["cache_dir"] = cache_dir
    if "refresh" in sig.parameters:
        kwargs["refresh"] = refresh_raw

    x, y, meta, skipped = load_sisfall_3class(**kwargs)
    if verbose:
        print("Raw SisFall data ready.")
    return x, y, meta, skipped


def build_manual_posture_records(
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
    labels: dict[str, dict[str, list[dict[str, float]]]],
    cfg: TaggingConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    idx = {_meta_key(m): i for i, m in enumerate(meta) if _meta_key(m) is not None}
    recs: list[dict[str, Any]] = []
    skip = {"not_found": 0, "non_static_dataset_label": 0, "too_short": 0}

    for base, labs in sorted(labels.items()):
        i = idx.get(base)
        if i is None:
            skip["not_found"] += 1
            continue
        if int(y[i]) != 2:
            skip["non_static_dataset_label"] += 1
            continue
        n = int(x[i].shape[0])
        if n < cfg.win:
            skip["too_short"] += 1
            continue

        starts = np.arange(0, n - cfg.win + 1, cfg.hop, dtype=np.int64)
        pt = _gaussian_score(n, labs["postural_transition"], starts, cfg, mode=cfg.ywin_mode)
        pt_max = _gaussian_score(n, labs["postural_transition"], starts, cfg, mode="max")
        sp = _window_score(n, labs["stable_posture"], starts, cfg, mode="mean")
        amb = _window_score(n, labs["ambiguous"], starts, cfg, mode="mean")

        yw = np.full(starts.shape[0], 0.5, np.float32)
        yw[(sp >= 0.60) & (amb < 0.25)] = 0.0
        yw[pt >= 0.50] = 1.0

        recs.append(
            {
                "file": meta[i]["file"],
                "code": meta[i]["code"],
                "subject": meta[i]["subject"],
                "trial": meta[i]["trial"],
                "starts": starts.astype(np.int64),
                "y_win": yw.astype(np.float32),
                "label_map": LABEL_MAP,
                "score_postural_transition": pt.astype(np.float32),
                "score_postural_transition_max": pt_max.astype(np.float32),
                "cov_stable_posture": sp.astype(np.float32),
                "cov_ambiguous": amb.astype(np.float32),
            }
        )
    return recs, skip


def train_and_tag_postures(
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
    labels: dict[str, dict[str, list[dict[str, float]]]],
    cfg: TaggingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idx = {_meta_key(m): i for i, m in enumerate(meta) if _meta_key(m) is not None}
    feats = [_extract_features(x[i], cfg) for i in range(len(x))]

    xf: list[np.ndarray] = []
    yt: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    files_used = 0
    skipped_clear = 0

    for base, labs in labels.items():
        i = idx.get(base)
        if i is None or feats[i] is None:
            continue
        starts, f = feats[i]
        n = int(x[i].shape[0])
        pt = _window_score(n, labs["postural_transition"], starts, cfg)
        sp = _window_score(n, labs["stable_posture"], starts, cfg)
        amb = _window_score(n, labs["ambiguous"], starts, cfg, mode="mean")

        clear_pt = (pt >= 0.50) & (sp <= 0.05) & (amb <= 0.10)
        clear_sp = (sp >= 0.60) & (pt <= 0.05) & (amb <= 0.10)
        keep = clear_pt | clear_sp
        if not np.any(keep):
            skipped_clear += 1
            continue
        xf.append(f[keep].astype(np.float32))
        yt.append(clear_pt[keep].astype(np.float32))
        ys.append(clear_sp[keep].astype(np.float32))
        files_used += 1

    if not xf:
        raise RuntimeError("No clear windows were found for posture tagger training.")

    xf0 = np.concatenate(xf, axis=0).astype(np.float32)
    yt0 = np.concatenate(yt, axis=0).astype(np.float32)
    ys0 = np.concatenate(ys, axis=0).astype(np.float32)

    mu = xf0.mean(axis=0, keepdims=True).astype(np.float32)
    sd = (xf0.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    z = np.nan_to_num(((xf0 - mu) / sd).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    wp = None
    if cfg.pca_k > 0:
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        wp = vt[: min(cfg.pca_k, vt.shape[0])].T.astype(np.float32)
        z = np.concatenate([z, z @ wp], axis=1).astype(np.float32)

    w_pt, b_pt = _fit_logreg(z, yt0, cfg)
    w_sp, b_sp = _fit_logreg(z, ys0, cfg)

    recs: list[dict[str, Any]] = []
    n_files = 0
    n_win = 0
    for i in range(len(x)):
        if int(y[i]) != 2 or feats[i] is None:
            continue
        starts, f = feats[i]
        zt = np.nan_to_num(((f - mu) / sd).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if wp is not None:
            zt = np.concatenate([zt, zt @ wp], axis=1).astype(np.float32)

        p_pt = _smooth(_sigmoid(zt @ w_pt + b_pt), cfg.smooth_k)
        p_sp = _smooth(_sigmoid(zt @ w_sp + b_sp), cfg.smooth_k)

        if cfg.use_hyst:
            yh_pt = _hyst(p_pt, cfg.th_pt_on, cfg.th_pt_off)
            yh_sp = _hyst(p_sp, cfg.th_sp_on, cfg.th_sp_off)
        else:
            yh_pt = (p_pt >= cfg.th_pt).astype(np.uint8)
            yh_sp = (p_sp >= cfg.th_sp).astype(np.uint8)

        pred_pt = ((p_pt >= cfg.th_pt) & ((p_pt - p_sp) >= cfg.margin)) | (
            (yh_pt > 0) & (yh_sp == 0) & ((p_pt - p_sp) >= 0.05)
        )
        pred_sp = ((p_sp >= cfg.th_sp) & ((p_sp - p_pt) >= cfg.margin)) | (
            (yh_sp > 0) & (yh_pt == 0) & ((p_sp - p_pt) >= 0.05)
        )

        yw = np.full_like(p_pt, 0.5, dtype=np.float32)
        yw[pred_sp & ~pred_pt] = 0.0
        yw[pred_pt & ~pred_sp] = 1.0
        yw[(p_pt <= cfg.amb_ceil) & (p_sp <= cfg.amb_ceil)] = 0.5

        recs.append(
            {
                "file": meta[i]["file"],
                "code": meta[i]["code"],
                "subject": meta[i]["subject"],
                "trial": meta[i]["trial"],
                "starts": starts.astype(np.int64),
                "y_win": yw.astype(np.float32),
                "label_map": LABEL_MAP,
                "p_postural_transition": p_pt.astype(np.float32),
                "p_stable_posture": p_sp.astype(np.float32),
                "y_hat_postural_transition": yh_pt.astype(np.uint8),
                "y_hat_stable_posture": yh_sp.astype(np.uint8),
            }
        )
        n_files += 1
        n_win += len(yw)

    model = {
        "mu": mu,
        "sd": sd,
        "wp": np.asarray([] if wp is None else wp, dtype=np.float32),
        "w_postural_transition": w_pt,
        "b_postural_transition": b_pt,
        "w_stable_posture": w_sp,
        "b_stable_posture": b_sp,
        "training_windows": int(len(xf0)),
        "training_files": int(files_used),
        "skipped_without_clear_windows": int(skipped_clear),
        "tagged_files": int(n_files),
        "tagged_windows": int(n_win),
    }
    return recs, model


def evaluate_window_tags(
    x: np.ndarray,
    meta: list[dict[str, Any]],
    labels: dict[str, dict[str, list[dict[str, float]]]],
    recs: list[dict[str, Any]],
    cfg: TaggingConfig,
) -> dict[str, Any]:
    """Evaluate predictions over manually clear windows."""
    idx = {_meta_key(m): i for i, m in enumerate(meta) if _meta_key(m) is not None}
    rec_map = {(_meta_key(r), str(r["code"]).upper(), str(r["subject"]).upper(), str(r["trial"]).upper()): r for r in recs}

    yt, yp, yo, pp, ps = [], [], [], [], []
    for base, labs in labels.items():
        i = idx.get(base)
        if i is None:
            continue
        m = meta[i]
        r = rec_map.get((base, str(m["code"]).upper(), str(m["subject"]).upper(), str(m["trial"]).upper()))
        if r is None:
            continue

        starts = np.asarray(r["starts"], np.int64)
        yw = np.asarray(r["y_win"], np.float32)
        n = int(x[i].shape[0])
        pt = _window_score(n, labs["postural_transition"], starts, cfg, mode="mean")
        sp = _window_score(n, labs["stable_posture"], starts, cfg, mode="mean")
        amb = _window_score(n, labs["ambiguous"], starts, cfg, mode="mean")

        clear_pt = (pt >= 0.50) & (sp <= 0.05) & (amb <= 0.10)
        clear_sp = (sp >= 0.60) & (pt <= 0.05) & (amb <= 0.10)
        keep = clear_pt | clear_sp
        if not np.any(keep):
            continue

        yt.append(clear_pt[keep].astype(np.uint8))
        yp.append(np.isclose(yw[keep], 1.0).astype(np.uint8))
        yo.append(yw[keep].astype(np.float32))
        pp.append(np.asarray(r["p_postural_transition"], np.float32)[keep])
        ps.append(np.asarray(r["p_stable_posture"], np.float32)[keep])

    if not yt:
        return {
            "n": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ambiguous_on_clear": 0,
            "p_postural_transition_mean": 0.0,
            "p_stable_posture_mean": 0.0,
        }

    yt0 = np.concatenate(yt)
    yp0 = np.concatenate(yp)
    yo0 = np.concatenate(yo)
    pp0 = np.concatenate(pp)
    ps0 = np.concatenate(ps)

    tp = int(np.sum((yp0 == 1) & (yt0 == 1)))
    tn = int(np.sum((yp0 == 0) & (yt0 == 0)))
    fp = int(np.sum((yp0 == 1) & (yt0 == 0)))
    fn = int(np.sum((yp0 == 0) & (yt0 == 1)))
    acc = (tp + tn) / max(1, len(yt0))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)

    return {
        "n": int(len(yt0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "ambiguous_on_clear": int(np.sum(np.isclose(yo0, 0.5))),
        "p_postural_transition_mean": float(pp0.mean()),
        "p_stable_posture_mean": float(ps0.mean()),
    }


def summarize_records(recs: list[dict[str, Any]]) -> dict[str, int]:
    y = np.concatenate([np.asarray(r["y_win"], np.float32) for r in recs]) if recs else np.asarray([], np.float32)
    return {
        "files": int(len(recs)),
        "windows": int(y.size),
        "stable_posture": int(np.sum(np.isclose(y, 0.0))),
        "postural_transition": int(np.sum(np.isclose(y, 1.0))),
        "ambiguous": int(np.sum(np.isclose(y, 0.5))),
    }

def run_posture_tagging(
    dataset_root: str | os.PathLike[str] = "data/SisFall_dataset",
    label_dir: str | os.PathLike[str] = "data/labels_transitions",
    result_dir: str | os.PathLike[str] = "data/sisfall_tagging_results",
    cache_dir: str | os.PathLike[str] = "data/sisfall_cache",
    cfg: TaggingConfig | None = None,
    run_name: str = "default",
    refresh: bool = False,
    refresh_raw: bool = False,
    save_plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg = TaggingConfig() if cfg is None else cfg
    dataset_root = Path(dataset_root)
    label_dir = Path(label_dir)
    result_dir = Path(result_dir)
    cache_dir = Path(cache_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cid = _cache_id(cfg, dataset_root, label_dir)
    stem = f"posture_{run_name}"
    cache_root = cache_dir / stem
    out_root = result_dir / stem
    cache_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    rec_cache = cache_root / "records.npy"
    manual_cache = cache_root / "manual_records.npy"
    model_cache = cache_root / "model.npz"
    summary_cache = cache_root / "summary.json"

    rec_out = out_root / "records.npy"
    manual_out = out_root / "manual_records.npy"
    summary_out = out_root / "summary.json"
    plot_out = out_root / "clear_window_probabilities.png"

    cache_ok = False
    if rec_cache.exists() and model_cache.exists() and summary_cache.exists() and not refresh:
        summary = json.loads(summary_cache.read_text(encoding="utf-8"))
        cache_ok = (
            summary.get("cache_id") == cid
            and summary.get("config") == asdict(cfg)
            and summary.get("dataset_root") == str(dataset_root)
            and summary.get("label_dir") == str(label_dir)
        )
        if cache_ok:
            recs = list(np.load(rec_cache, allow_pickle=True).tolist())
            manual = list(np.load(manual_cache, allow_pickle=True).tolist()) if manual_cache.exists() else []
            if verbose:
                print("Posture tagging artifacts loaded from cache.")
                s = summary.get("autotagged_records", {})
                print(f"   Files tagged: {s.get('files', 0)} | windows: {s.get('windows', 0)}")
                print(
                    "   Labels: "
                    f"stable_posture={s.get('stable_posture', 0)} | "
                    f"postural_transition={s.get('postural_transition', 0)} | ambiguous={s.get('ambiguous', 0)}"
                )
                wv = summary.get("evaluation", {}).get("windows", {})
                if wv:
                    print(
                        "   Window-level check: "
                        f"N={wv.get('n', 0)} | accuracy={wv.get('accuracy', 0.0):.3f} | "
                        f"precision={wv.get('precision', 0.0):.3f} | "
                        f"recall={wv.get('recall', 0.0):.3f} | f1={wv.get('f1', 0.0):.3f}"
                    )
            np.save(rec_out, np.asarray(recs, dtype=object), allow_pickle=True)
            if manual:
                np.save(manual_out, np.asarray(manual, dtype=object), allow_pickle=True)
            summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return {
                "records": recs,
                "manual_records": manual,
                "summary": summary,
                "paths": {
                    "records": str(rec_out),
                    "manual_records": str(manual_out),
                    "model_cache": str(model_cache),
                    "summary": str(summary_out),
                    "plot": str(plot_out) if save_plot else None,
                    "output_dir": str(out_root),
                    "cache_dir": str(cache_root),
                },
            }
        if verbose:
            print("Posture cache found with different settings; rebuilding and overwriting it.")

    if verbose and not cache_ok:
        print("SisFall posture tagging")
        print("   Preparing data, manual annotations, and posture artifacts.")

    x, y, meta, skipped = _load_dataset(dataset_root, cfg.acc, cfg.dtype, cache_dir, refresh_raw, verbose)
    labels = _load_labels(label_dir)
    manual, manual_skip = build_manual_posture_records(x, y, meta, labels, cfg)
    recs, model = train_and_tag_postures(x, y, meta, labels, cfg)
    win_eval = evaluate_window_tags(x, meta, labels, recs, cfg)

    summary = {
        "run_name": run_name,
        "cache_id": cid,
        "dataset_root": str(dataset_root),
        "label_dir": str(label_dir),
        "label_map": LABEL_MAP,
        "raw_data": {"recordings": int(len(x)), "skipped": int(skipped)},
        "manual_annotations": {"files": int(len(labels)), "skipped": manual_skip},
        "manual_records": summarize_records(manual),
        "autotagged_records": summarize_records(recs),
        "model": {k: v for k, v in model.items() if not isinstance(v, np.ndarray)},
        "evaluation": {"windows": win_eval},
        "config": asdict(cfg),
    }

    np.save(rec_cache, np.asarray(recs, dtype=object), allow_pickle=True)
    np.save(manual_cache, np.asarray(manual, dtype=object), allow_pickle=True)
    np.save(rec_out, np.asarray(recs, dtype=object), allow_pickle=True)
    np.save(manual_out, np.asarray(manual, dtype=object), allow_pickle=True)
    np.savez_compressed(model_cache, **{k: v for k, v in model.items() if isinstance(v, np.ndarray)})
    summary_cache.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if save_plot:
        _save_probability_plot(recs, x, meta, labels, cfg, plot_out)

    if verbose:
        s = summary["autotagged_records"]
        print("Tagging completed.")
        print(f"   Files tagged: {s['files']} | windows: {s['windows']}")
        print(
            "   Labels: "
            f"stable_posture={s['stable_posture']} | "
            f"postural_transition={s['postural_transition']} | ambiguous={s['ambiguous']}"
        )
        wv = win_eval
        print(
            "   Window-level check: "
            f"N={wv['n']} | accuracy={wv['accuracy']:.3f} | "
            f"precision={wv['precision']:.3f} | recall={wv['recall']:.3f} | f1={wv['f1']:.3f}"
        )
        print(f"   Results: {out_root}")
        print(f"   Cache: {cache_root}")

    return {
        "records": recs,
        "manual_records": manual,
        "model": model,
        "summary": summary,
        "paths": {
            "records": str(rec_out),
            "manual_records": str(manual_out),
            "model_cache": str(model_cache),
            "summary": str(summary_out),
            "plot": str(plot_out) if save_plot else None,
            "output_dir": str(out_root),
            "cache_dir": str(cache_root),
        },
    }


def _save_probability_plot(
    recs: list[dict[str, Any]],
    x: np.ndarray,
    meta: list[dict[str, Any]],
    labels: dict[str, dict[str, list[dict[str, float]]]],
    cfg: TaggingConfig,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    idx = {_meta_key(m): i for i, m in enumerate(meta) if _meta_key(m) is not None}
    rec_map = {(_meta_key(r), str(r["code"]).upper(), str(r["subject"]).upper(), str(r["trial"]).upper()): r for r in recs}
    y_true, pp, ps = [], [], []
    for base, labs in labels.items():
        i = idx.get(base)
        if i is None:
            continue
        m = meta[i]
        r = rec_map.get((base, str(m["code"]).upper(), str(m["subject"]).upper(), str(m["trial"]).upper()))
        if r is None:
            continue
        starts = np.asarray(r["starts"], np.int64)
        n = int(x[i].shape[0])
        pt = _window_score(n, labs["postural_transition"], starts, cfg, mode="mean")
        sp = _window_score(n, labs["stable_posture"], starts, cfg, mode="mean")
        amb = _window_score(n, labs["ambiguous"], starts, cfg, mode="mean")
        keep_pt = (pt >= 0.50) & (sp <= 0.05) & (amb <= 0.10)
        keep_sp = (sp >= 0.60) & (pt <= 0.05) & (amb <= 0.10)
        keep = keep_pt | keep_sp
        if not np.any(keep):
            continue
        y_true.append(keep_pt[keep].astype(np.uint8))
        pp.append(np.asarray(r["p_postural_transition"], np.float32)[keep])
        ps.append(np.asarray(r["p_stable_posture"], np.float32)[keep])

    if not y_true:
        return
    yt = np.concatenate(y_true)
    ptt = np.concatenate(pp)
    pss = np.concatenate(ps)
    plt.figure(figsize=(5.6, 5.6))
    plt.scatter(pss[yt == 0], ptt[yt == 0], s=10, alpha=0.40, label="stable_posture")
    plt.scatter(pss[yt == 1], ptt[yt == 1], s=10, alpha=0.40, label="postural_transition")
    plt.xlabel("p(stable_posture)")
    plt.ylabel("p(postural_transition)")
    plt.title("Clear windows probability map")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

