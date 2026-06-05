"""Utilities for Sigma-Delta theta optimization."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from preprocess.event_encoding import sigma_delta
from preprocess.utils import class_counts, json_ready, load_json, save_json_atomic, split_counts, stratified_split_indices

from preprocess.interpolate_signal import derive_output_length, interpolate_signal, interpolate_to_times, make_resample_plan


SISFALL_LABELS = {
    0: "FALL",
    1: "NORMAL_DYNAMIC",
    2: "NORMAL_STATIC",
}

UCIHAR_LABELS = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


@dataclass(frozen=True)
class DatasetDefaults:
    raw_win: int
    label_names: dict[int, str]
    theta_init: tuple[float, ...]
    init_step: float
    min_step: float
    round_decimals: int
    theta_max_scale: float
    train_per_class: int
    val_per_class: int
    test_per_class: int
    max_epochs: int
    patience_global: int
    patience_lr: int
    alpha_worst: float = 0.5
    theta_min: float = 1e-3
    theta_max_pctl: float = 95.0
    theta_max_floor: float = 0.02
    theta_max_init_margin: float = 1.75
    window_stride: int | None = None


DATASET_DEFAULTS: dict[str, DatasetDefaults] = {
    "sisfall": DatasetDefaults(
        raw_win=410,
        label_names=SISFALL_LABELS,
        theta_init=(41.000, 47.838, 51.000, 274.290, 161.000, 157.838),
        init_step=10.0,
        min_step=9e-4,
        round_decimals=3,
        theta_max_scale=0.5,
        train_per_class=682,
        val_per_class=170,
        test_per_class=170,
        max_epochs=200,
        patience_global=20,
        patience_lr=5,
        window_stride=205,
    ),
    "ucihar": DatasetDefaults(
        raw_win=128,
        label_names=UCIHAR_LABELS,
        theta_init=(0.381, 0.4185323, 0.2887801, 0.281, 0.180, 0.154),
        init_step=0.020,
        min_step=5e-4,
        round_decimals=4,
        theta_max_scale=1.25,
        train_per_class=512,
        val_per_class=160,
        test_per_class=160,
        max_epochs=120,
        patience_global=18,
        patience_lr=4,
    ),
}


@dataclass
class RunConfig:
    dataset: str
    data_root: Path
    ups: float
    dead_zone: float
    results_root: Path
    workers: int
    executor: str
    keep_worker_cache: bool
    chunk_size: int
    seed: int
    resume: bool
    out_len_override: int | None
    interp_method: str
    sd_init: str
    max_epochs: int
    patience_global: int
    patience_lr: int
    init_step: float
    min_step: float
    step_shrink: float
    improve_tol: float
    alpha_worst: float
    round_decimals: int
    theta_init: tuple[float, ...]
    theta_min: float
    theta_max: tuple[float, ...] | None
    theta_max_pctl: float
    theta_max_scale: float
    theta_max_floor: float
    theta_max_init_margin: float
    train_per_class: int
    val_per_class: int
    test_per_class: int
    final_test_all: bool
    window_stride: int | None
    fall_policy: str
    acc_sensor: str
    case: str
    target_domain: int
    split: tuple[float, float, float]
    val_ratio: float
    batch_size: int
    print_every_epoch: bool

    @classmethod
    def from_args(cls, args: Any) -> "RunConfig":
        dataset = str(args.dataset).lower()
        if dataset not in DATASET_DEFAULTS:
            raise ValueError(f"Unknown dataset: {dataset}")
        d = DATASET_DEFAULTS[dataset]

        def pick(name: str, default: Any) -> Any:
            value = getattr(args, name)
            return default if value is None else value

        theta_init = pick("theta_init", d.theta_init)
        theta_max = getattr(args, "theta_max", None)

        return cls(
            dataset=dataset,
            data_root=Path(args.data_root),
            ups=float(args.ups),
            dead_zone=float(args.dead_zone),
            results_root=Path(args.results_root),
            workers=max(1, int(args.workers)),
            executor=str(args.executor).lower(),
            keep_worker_cache=bool(args.keep_worker_cache),
            chunk_size=max(1, int(args.chunk_size)),
            seed=int(args.seed),
            resume=not bool(args.force),
            out_len_override=getattr(args, "out_len", None),
            interp_method=str(args.interp_method),
            sd_init=str(args.sd_init),
            max_epochs=int(pick("max_epochs", d.max_epochs)),
            patience_global=int(pick("patience_global", d.patience_global)),
            patience_lr=int(pick("patience_lr", d.patience_lr)),
            init_step=float(pick("init_step", d.init_step)),
            min_step=float(pick("min_step", d.min_step)),
            step_shrink=float(args.step_shrink),
            improve_tol=float(args.improve_tol),
            alpha_worst=float(pick("alpha_worst", d.alpha_worst)),
            round_decimals=int(pick("round_decimals", d.round_decimals)),
            theta_init=tuple(float(v) for v in theta_init),
            theta_min=float(pick("theta_min", d.theta_min)),
            theta_max=None if theta_max is None else tuple(float(v) for v in theta_max),
            theta_max_pctl=float(pick("theta_max_pctl", d.theta_max_pctl)),
            theta_max_scale=float(pick("theta_max_scale", d.theta_max_scale)),
            theta_max_floor=float(pick("theta_max_floor", d.theta_max_floor)),
            theta_max_init_margin=float(pick("theta_max_init_margin", d.theta_max_init_margin)),
            train_per_class=int(pick("train_per_class", d.train_per_class)),
            val_per_class=int(pick("val_per_class", d.val_per_class)),
            test_per_class=int(pick("test_per_class", d.test_per_class)),
            final_test_all=not bool(args.sampled_test_only),
            window_stride=pick("window_stride", d.window_stride),
            fall_policy=str(args.fall_policy),
            acc_sensor=str(args.acc_sensor),
            case=str(args.case),
            target_domain=int(args.target_domain),
            split=tuple(float(v) for v in args.split),
            val_ratio=float(args.val_ratio),
            batch_size=int(args.batch_size),
            print_every_epoch=not bool(args.quiet_epochs),
        )

    @property
    def defaults(self) -> DatasetDefaults:
        return DATASET_DEFAULTS[self.dataset]

    @property
    def raw_win(self) -> int:
        return self.defaults.raw_win

    @property
    def label_names(self) -> dict[int, str]:
        return self.defaults.label_names

    @property
    def out_len(self) -> int:
        if self.out_len_override is not None:
            return int(self.out_len_override)
        if self.dataset == "sisfall" and self.raw_win == 410 and abs(float(self.ups) - 5.0) < 1e-12:
            return 2048
        return int(derive_output_length(self.raw_win, self.ups))

    @property
    def result_dir(self) -> Path:
        tag = f"{self.dataset}_ups{safe_tag(self.ups)}_dz{safe_tag(self.dead_zone)}"
        return self.results_root / tag


@dataclass
class SplitWindows:
    x_native: np.ndarray
    x_interp: np.ndarray
    y: np.ndarray
    chunks: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x_native = as_window_array(self.x_native)
        self.x_interp = as_window_array(self.x_interp)
        self.y = normalize_labels(self.y)
        n = self.x_native.shape[0]
        if self.x_interp.shape[0] != n or self.y.shape[0] != n:
            raise ValueError("x_native, x_interp and y must have the same number of samples.")
        if self.x_native.shape[-1] != self.x_interp.shape[-1]:
            raise ValueError("Native and interpolated windows must have the same number of channels.")


@dataclass
class PreparedData:
    train: SplitWindows
    val: SplitWindows
    test_sampled: SplitWindows
    test_full: SplitWindows
    label_names: dict[int, str]
    raw_win: int
    out_len: int
    t_native: np.ndarray
    t_interp: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return int(self.train.x_native.shape[-1])

    @property
    def splits(self) -> dict[str, SplitWindows]:
        return {
            "train": self.train,
            "val": self.val,
            "test_sampled": self.test_sampled,
            "test_full": self.test_full,
        }


def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{name}_{stamp}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def safe_tag(x: Any) -> str:
    return str(x).replace(".", "p").replace("-", "m").replace("/", "_")


def is_tensor_like(x: Any) -> bool:
    return hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy")


def normalize_labels(y: Any) -> np.ndarray:
    if is_tensor_like(y):
        y = y.detach().cpu().numpy()
    arr = np.asarray(y).reshape(-1).astype(np.int64, copy=False)
    if arr.size and arr.min() == 1 and arr.max() == np.unique(arr).size:
        arr = arr - 1
    return arr


def as_window_array(x: Any, n_channels: int = 6) -> np.ndarray:
    if is_tensor_like(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[2] == 1 and arr.shape[3] == n_channels:
        arr = arr[:, :, 0, :]
    elif arr.ndim == 4 and arr.shape[1] == 1 and arr.shape[-1] == n_channels:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3 or arr.shape[-1] != n_channels:
        raise ValueError(f"Expected shape [N, T, {n_channels}], got {arr.shape}.")
    return arr.astype(np.float32, copy=False)

def collect_loader(loader: Iterable[Any], n_channels: int = 6) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for batch in loader:
        if isinstance(batch, Mapping):
            xb = next((batch[k] for k in ("x", "X", "data", "inputs") if k in batch), None)
            yb = next((batch[k] for k in ("y", "label", "labels", "target", "targets") if k in batch), None)
            if xb is None or yb is None:
                raise ValueError("Loader batches must contain x/y-like keys.")
        else:
            xb, yb = batch[:2]
        xs.append(as_window_array(xb, n_channels=n_channels))
        ys.append(normalize_labels(yb))
    if not xs:
        raise RuntimeError("The loader produced no batches.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def make_chunks(n: int, chunk_size: int) -> list[tuple[int, int]]:
    step = max(1, int(chunk_size))
    return [(i, min(i + step, int(n))) for i in range(0, int(n), step)]

def balanced_indices(y: Any, n_per_class: int, seed: int) -> np.ndarray:
    yy = normalize_labels(y)
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for cls in sorted(np.unique(yy).tolist()):
        pool = np.where(yy == cls)[0]
        take = min(int(n_per_class), int(pool.size))
        if take > 0:
            selected.append(rng.choice(pool, size=take, replace=False))
    if not selected:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(selected).astype(np.int64)
    rng.shuffle(out)
    return out


def clip_theta(theta: Sequence[float], theta_min: float, theta_max: Sequence[float], decimals: int) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float32)
    hi = np.asarray(theta_max, dtype=np.float32)
    if th.shape != hi.shape:
        raise ValueError(f"theta shape {th.shape} does not match theta_max shape {hi.shape}.")
    th = np.clip(th, float(theta_min), hi)
    th = np.round(th, int(decimals))
    th = np.clip(th, float(theta_min), hi)
    return th.astype(np.float32, copy=False)


def compute_theta_max(x_train: np.ndarray, cfg: RunConfig) -> np.ndarray:
    if cfg.theta_max is not None:
        return np.asarray(cfg.theta_max, dtype=np.float32)
    x = as_window_array(x_train)
    flat = x.reshape(-1, x.shape[-1])
    amp = np.percentile(np.abs(flat), cfg.theta_max_pctl, axis=0).astype(np.float32)
    theta_init = np.asarray(cfg.theta_init, dtype=np.float32)
    theta_max = np.maximum(cfg.theta_max_floor, cfg.theta_max_scale * amp)
    theta_max = np.maximum(theta_max, cfg.theta_max_init_margin * theta_init)
    return theta_max.astype(np.float32, copy=False)


def build_split(x_native: np.ndarray, y: np.ndarray, cfg: RunConfig) -> SplitWindows:
    x_native = as_window_array(x_native)
    plan = make_resample_plan(x_native.shape[1], cfg.out_len, method=cfg.interp_method, dtype=np.float32)
    x_interp = np.empty((x_native.shape[0], cfg.out_len, x_native.shape[-1]), dtype=np.float32)
    for i in range(x_native.shape[0]):
        x_interp[i] = interpolate_signal(x_native[i], plan=plan, dtype=np.float32)
    return SplitWindows(
        x_native=x_native,
        x_interp=x_interp,
        y=normalize_labels(y),
        chunks=make_chunks(len(y), cfg.chunk_size),
    )


def prepare_data(cfg: RunConfig, logger: logging.Logger) -> PreparedData:
    if cfg.dataset == "sisfall":
        return prepare_sisfall(cfg, logger)
    if cfg.dataset == "ucihar":
        return prepare_ucihar(cfg, logger)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def _time_grids(cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    plan = make_resample_plan(cfg.raw_win, cfg.out_len, method=cfg.interp_method, dtype=np.float32)
    return plan.t_in.copy(), plan.t_out.copy()


def resolve_sisfall_root(data_root: Path) -> Path:
    candidates = [
        data_root / "SisFall_dataset",
        data_root / "SisFall",
        data_root / "sisfall",
        data_root,
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return candidates[0]


def _fall_peak_index(x: np.ndarray) -> int:
    acc = np.asarray(x[:, :3], dtype=np.float32)
    return int(np.argmax(np.max(np.abs(acc), axis=1)))


def _window_starts(x: np.ndarray, label: int, cfg: RunConfig) -> list[int]:
    win = int(cfg.raw_win)
    if x.shape[0] < win:
        return []
    stride = int(cfg.window_stride or win)
    starts = list(range(0, x.shape[0] - win + 1, max(1, stride)))
    if label == 0 and cfg.fall_policy == "contain_peak":
        peak = _fall_peak_index(x)
        selected = [s for s in starts if s <= peak < s + win]
        if selected:
            return selected
        centered = int(np.clip(peak - win // 2, 0, x.shape[0] - win))
        return [centered]
    return starts


def prepare_sisfall(cfg: RunConfig, logger: logging.Logger) -> PreparedData:
    from preprocess.preprocess_sisfall import load_sisfall_3class

    root = resolve_sisfall_root(cfg.data_root)
    cache_dir = cfg.data_root / "sisfall_cache"
    x_files, y_files, _, skipped = load_sisfall_3class(
        root=root,
        acc=cfg.acc_sensor,
        dtype=np.int32,
        cache_dir=cache_dir,
        refresh=False,
    )
    y_files = normalize_labels(y_files)
    idx_tr, idx_va, idx_te = stratified_split_indices(y_files, (0.8, 0.1, 0.1), cfg.seed)

    def collect(file_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for i in file_idx:
            seq = np.asarray(x_files[int(i)], dtype=np.float32)
            lab = int(y_files[int(i)])
            for start in _window_starts(seq, lab, cfg):
                xs.append(seq[start : start + cfg.raw_win])
                ys.append(lab)
        if not xs:
            raise RuntimeError("No SisFall windows were generated. Check data-root, window-stride, and fall-policy.")
        return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)

    xtr_full, ytr_full = collect(idx_tr)
    xva_full, yva_full = collect(idx_va)
    xte_full, yte_full = collect(idx_te)

    tr_idx = balanced_indices(ytr_full, cfg.train_per_class, cfg.seed + 101)
    va_idx = balanced_indices(yva_full, cfg.val_per_class, cfg.seed + 202)
    te_idx = balanced_indices(yte_full, cfg.test_per_class, cfg.seed + 303)
    tf_idx = np.arange(yte_full.size, dtype=np.int64) if cfg.final_test_all else te_idx

    logger.info("SisFall root=%s | files=%d | skipped=%d", root, len(x_files), skipped)
    logger.info("SisFall file counts | train=%s | val=%s | test=%s", class_counts(y_files[idx_tr], SISFALL_LABELS), class_counts(y_files[idx_va], SISFALL_LABELS), class_counts(y_files[idx_te], SISFALL_LABELS))
    logger.info("SisFall window counts full | train=%s | val=%s | test=%s", class_counts(ytr_full, SISFALL_LABELS), class_counts(yva_full, SISFALL_LABELS), class_counts(yte_full, SISFALL_LABELS))
    logger.info("SisFall sampled windows | train=%d | val=%d | test_sampled=%d | test_full=%d", tr_idx.size, va_idx.size, te_idx.size, tf_idx.size)

    t_native, t_interp = _time_grids(cfg)
    return PreparedData(
        train=build_split(xtr_full[tr_idx], ytr_full[tr_idx], cfg),
        val=build_split(xva_full[va_idx], yva_full[va_idx], cfg),
        test_sampled=build_split(xte_full[te_idx], yte_full[te_idx], cfg),
        test_full=build_split(xte_full[tf_idx], yte_full[tf_idx], cfg),
        label_names=SISFALL_LABELS,
        raw_win=cfg.raw_win,
        out_len=cfg.out_len,
        t_native=t_native,
        t_interp=t_interp,
        metadata={
            "dataset": "sisfall",
            "dataset_root": str(root),
            "cache_dir": str(cache_dir),
            "acc_sensor": cfg.acc_sensor,
            "window_stride": cfg.window_stride,
            "fall_policy": cfg.fall_policy,
            "counts_full": {
                "train": class_counts(ytr_full, SISFALL_LABELS),
                "val": class_counts(yva_full, SISFALL_LABELS),
                "test": class_counts(yte_full, SISFALL_LABELS),
            },
        },
    )

def _call_preprocess_ucihar(cfg: RunConfig) -> Any:
    from preprocess.preprocess_ucihar import preprocess_ucihar

    return preprocess_ucihar(
        data_root=str(cfg.data_root),
        case=cfg.case,
        target_domain=cfg.target_domain,
        split=cfg.split,
        val_ratio=cfg.val_ratio,
        seed=int(cfg.seed),
        model_shape=False,
        verbose=False,
    )

def prepare_ucihar(cfg: RunConfig, logger: logging.Logger) -> PreparedData:
    raw = _call_preprocess_ucihar(cfg)
    x_train = as_window_array(raw.X_train)
    y_train = normalize_labels(raw.y_train)
    x_val = as_window_array(raw.X_val)
    y_val = normalize_labels(raw.y_val)
    x_test = as_window_array(raw.X_test)
    y_test = normalize_labels(raw.y_test)

    if x_train.shape[1] != cfg.raw_win:
        raise ValueError(
            f"Expected UCI HAR temporal length {cfg.raw_win}, got {x_train.shape[1]}."
        )

    tr_idx = balanced_indices(y_train, cfg.train_per_class, cfg.seed + 111)
    va_idx = balanced_indices(y_val, cfg.val_per_class, cfg.seed + 222)
    te_idx = balanced_indices(y_test, cfg.test_per_class, cfg.seed + 333)
    tf_idx = np.arange(y_test.size, dtype=np.int64) if cfg.final_test_all else te_idx

    logger.info("UCI HAR data_root=%s", cfg.data_root)
    logger.info(
        "UCI HAR arrays | train=%s | val=%s | test=%s",
        x_train.shape,
        x_val.shape,
        x_test.shape,
    )
    logger.info(
        "UCI HAR counts | train=%s | val=%s | test=%s",
        class_counts(y_train, UCIHAR_LABELS),
        class_counts(y_val, UCIHAR_LABELS),
        class_counts(y_test, UCIHAR_LABELS),
    )
    logger.info(
        "UCI HAR sampled windows | train=%d | val=%d | test_sampled=%d | test_full=%d",
        tr_idx.size,
        va_idx.size,
        te_idx.size,
        tf_idx.size,
    )

    t_native, t_interp = _time_grids(cfg)

    return PreparedData(
        train=build_split(x_train[tr_idx], y_train[tr_idx], cfg),
        val=build_split(x_val[va_idx], y_val[va_idx], cfg),
        test_sampled=build_split(x_test[te_idx], y_test[te_idx], cfg),
        test_full=build_split(x_test[tf_idx], y_test[tf_idx], cfg),
        label_names=UCIHAR_LABELS,
        raw_win=cfg.raw_win,
        out_len=cfg.out_len,
        t_native=t_native,
        t_interp=t_interp,
        metadata={
            "dataset": "ucihar",
            "data_root": str(cfg.data_root),
            "case": cfg.case,
            "target_domain": cfg.target_domain if cfg.case in {"subject", "subject_large"} else None,
            "split": cfg.split,
            "val_ratio": cfg.val_ratio,
            "counts_full": {
                "train": class_counts(y_train, UCIHAR_LABELS),
                "val": class_counts(y_val, UCIHAR_LABELS),
                "test": class_counts(y_test, UCIHAR_LABELS),
            },
        },
    )

_WORKER_SPLITS: dict[str, SplitWindows] = {}
_WORKER_T_NATIVE: np.ndarray | None = None
_WORKER_T_INTERP: np.ndarray | None = None
_WORKER_LABEL_NAMES: dict[int, str] = {}
_WORKER_SD_INIT = "x0"
_WORKER_INTERP_METHOD = "linear"


def _worker_init(
    splits: dict[str, SplitWindows],
    t_native: np.ndarray,
    t_interp: np.ndarray,
    label_names: dict[int, str],
    sd_init: str,
    interp_method: str,
) -> None:
    global _WORKER_SPLITS, _WORKER_T_NATIVE, _WORKER_T_INTERP, _WORKER_LABEL_NAMES, _WORKER_SD_INIT, _WORKER_INTERP_METHOD
    _WORKER_SPLITS = splits
    _WORKER_T_NATIVE = np.asarray(t_native, dtype=np.float32)
    _WORKER_T_INTERP = np.asarray(t_interp, dtype=np.float32)
    _WORKER_LABEL_NAMES = dict(label_names)
    _WORKER_SD_INIT = str(sd_init)
    _WORKER_INTERP_METHOD = str(interp_method)


def _objective_chunk(job: tuple[str, int, int, np.ndarray, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_name, a, b, theta, dead_zone = job
    if _WORKER_T_NATIVE is None or _WORKER_T_INTERP is None:
        raise RuntimeError("Worker state is not initialized.")

    split = _WORKER_SPLITS[split_name]
    n_classes = max(_WORKER_LABEL_NAMES) + 1
    theta = np.asarray(theta, dtype=np.float32)

    err = np.zeros(n_classes, dtype=np.float64)
    eng = np.zeros(n_classes, dtype=np.float64)
    nwin = np.zeros(n_classes, dtype=np.int64)
    ev_sum = np.zeros(n_classes, dtype=np.float64)
    ev_sumsq = np.zeros(n_classes, dtype=np.float64)
    ev_n = np.zeros(n_classes, dtype=np.int64)

    for i in range(int(a), int(b)):
        cls = int(split.y[i])
        x_up = split.x_interp[i]
        events, _, rec_up = sigma_delta(
            x_up,
            theta=theta,
            init=_WORKER_SD_INIT,
            return_reconstruction=True,
            dead_zone=float(dead_zone),
        )
        if rec_up is None:
            raise RuntimeError("sigma_delta returned no reconstruction trace.")

        rec_native = interpolate_to_times(
            rec_up,
            t_src=_WORKER_T_INTERP,
            t_dst=_WORKER_T_NATIVE,
            method=_WORKER_INTERP_METHOD,
        )
        x_ref = split.x_native[i]
        diff = (x_ref - rec_native).astype(np.float64, copy=False)
        ref = x_ref.astype(np.float64, copy=False)

        err[cls] += float(np.sum(diff * diff))
        eng[cls] += float(np.sum(ref * ref))
        nwin[cls] += 1

        ev_count = int(np.count_nonzero(events))
        ev_sum[cls] += ev_count
        ev_sumsq[cls] += ev_count * ev_count
        ev_n[cls] += 1

    return err, eng, nwin, ev_sum, ev_sumsq, ev_n

def _worker_init_from_files(
    split_files: Mapping[str, Mapping[str, Any]],
    t_native: np.ndarray,
    t_interp: np.ndarray,
    label_names: dict[int, str],
    sd_init: str,
    interp_method: str,
) -> None:
    """Initialize a process worker from memory-mapped .npy files.

    This avoids sending large interpolated SisFall/UCI HAR arrays through the
    Windows spawn pipe, which is the usual cause of truncated pickle errors.
    """
    splits: dict[str, SplitWindows] = {}
    for name, spec in split_files.items():
        x_native = np.load(str(spec["x_native"]), mmap_mode="r")
        x_interp = np.load(str(spec["x_interp"]), mmap_mode="r")
        y = np.load(str(spec["y"]), mmap_mode="r")
        chunks = [tuple(map(int, chunk)) for chunk in spec["chunks"]]
        splits[str(name)] = SplitWindows(x_native=x_native, x_interp=x_interp, y=y, chunks=chunks)
    _worker_init(splits, t_native, t_interp, label_names, sd_init, interp_method)



def event_stats(ev_sum: np.ndarray, ev_sumsq: np.ndarray, ev_n: np.ndarray, labels: Mapping[int, str]) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    for cls in sorted(labels):
        n = int(ev_n[cls]) if cls < ev_n.size else 0
        if n == 0:
            by_class[str(cls)] = {"label": labels[cls], "mean": 0.0, "std": 0.0, "n": 0}
        else:
            mean = float(ev_sum[cls] / n)
            var = max(0.0, float(ev_sumsq[cls] / n - mean * mean))
            by_class[str(cls)] = {"label": labels[cls], "mean": mean, "std": float(np.sqrt(var)), "n": n}

    total = int(ev_n.sum())
    if total == 0:
        global_stats = {"mean": 0.0, "std": 0.0, "n": 0}
    else:
        mean = float(ev_sum.sum() / total)
        var = max(0.0, float(ev_sumsq.sum() / total - mean * mean))
        global_stats = {"mean": mean, "std": float(np.sqrt(var)), "n": total}
    return {"by_class": by_class, "global": global_stats}


class ThetaOptimizer:
    def __init__(self, data: PreparedData, cfg: RunConfig, logger: logging.Logger) -> None:
        self.data = data
        self.cfg = cfg
        self.logger = logger
        self.result_dir = cfg.result_dir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.result_dir / "checkpoint.json"
        self.summary_path = self.result_dir / "summary.json"
        self.worker_cache_dir = self.result_dir / "_worker_cache"

        self.theta_init = np.asarray(cfg.theta_init, dtype=np.float32)
        self.theta_max = compute_theta_max(data.train.x_native, cfg)

        _worker_init(self._splits_with_chunks(), data.t_native, data.t_interp, data.label_names, cfg.sd_init, cfg.interp_method)

    def _splits_with_chunks(self) -> dict[str, SplitWindows]:
        splits = self.data.splits
        for split in splits.values():
            split.chunks = make_chunks(split.y.size, self.cfg.chunk_size)
        return splits

    def _executor_kind(self) -> str:
        kind = str(self.cfg.executor).lower()
        if kind not in {"auto", "process", "thread", "none"}:
            raise ValueError("executor must be one of: auto, process, thread, none")
        if self.cfg.workers <= 1 or kind == "none":
            return "none"
        if kind == "auto":
            return "process"
        return kind

    def _write_worker_cache(self) -> dict[str, dict[str, Any]]:
        """Write objective arrays to .npy files for process workers.

        ProcessPoolExecutor on Windows starts fresh Python interpreters with
        spawn. Passing multi-hundred-MB arrays as initializer arguments can fail
        with `_pickle.UnpicklingError: pickle data was truncated`. The cache
        keeps the process payload small and lets each worker memory-map the
        arrays locally.
        """
        if self.worker_cache_dir.exists():
            shutil.rmtree(self.worker_cache_dir)
        self.worker_cache_dir.mkdir(parents=True, exist_ok=True)

        specs: dict[str, dict[str, Any]] = {}
        for name, split in self._splits_with_chunks().items():
            prefix = self.worker_cache_dir / name
            x_native_path = prefix.with_name(prefix.name + "_x_native.npy")
            x_interp_path = prefix.with_name(prefix.name + "_x_interp.npy")
            y_path = prefix.with_name(prefix.name + "_y.npy")
            np.save(x_native_path, np.asarray(split.x_native, dtype=np.float32))
            np.save(x_interp_path, np.asarray(split.x_interp, dtype=np.float32))
            np.save(y_path, np.asarray(split.y, dtype=np.int64))
            specs[name] = {
                "x_native": str(x_native_path),
                "x_interp": str(x_interp_path),
                "y": str(y_path),
                "chunks": [(int(a), int(b)) for a, b in split.chunks],
            }
        return specs

    def _cleanup_worker_cache(self) -> None:
        if not self.cfg.keep_worker_cache and self.worker_cache_dir.exists():
            shutil.rmtree(self.worker_cache_dir, ignore_errors=True)

    def _pool(self) -> Any:
        kind = self._executor_kind()
        if kind == "none":
            self.logger.info("executor=none | workers=1 | no multiprocessing")
            return None

        if kind == "thread":
            from concurrent.futures import ThreadPoolExecutor

            self.logger.info("executor=thread | workers=%d", self.cfg.workers)
            _worker_init(self._splits_with_chunks(), self.data.t_native, self.data.t_interp, self.data.label_names, self.cfg.sd_init, self.cfg.interp_method)
            return ThreadPoolExecutor(max_workers=int(self.cfg.workers))

        from concurrent.futures import ProcessPoolExecutor

        split_files = self._write_worker_cache()
        self.logger.info("executor=process | workers=%d | worker_cache=%s", self.cfg.workers, self.worker_cache_dir)
        return ProcessPoolExecutor(
            max_workers=int(self.cfg.workers),
            initializer=_worker_init_from_files,
            initargs=(split_files, self.data.t_native, self.data.t_interp, self.data.label_names, self.cfg.sd_init, self.cfg.interp_method),
        )

    def objective(self, split_name: str, theta: np.ndarray, pool: Any) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
        split = self.data.splits[split_name]
        jobs = [(split_name, a, b, np.asarray(theta, dtype=np.float32), self.cfg.dead_zone) for a, b in split.chunks]
        parts = [_objective_chunk(job) for job in jobs] if pool is None else list(pool.map(_objective_chunk, jobs))

        n_classes = max(self.data.label_names) + 1
        err = np.zeros(n_classes, dtype=np.float64)
        eng = np.zeros(n_classes, dtype=np.float64)
        nwin = np.zeros(n_classes, dtype=np.int64)
        ev_sum = np.zeros(n_classes, dtype=np.float64)
        ev_sumsq = np.zeros(n_classes, dtype=np.float64)
        ev_n = np.zeros(n_classes, dtype=np.int64)

        for e, g, n, es, ess, en in parts:
            err += e
            eng += g
            nwin += n
            ev_sum += es
            ev_sumsq += ess
            ev_n += en

        rel = err / np.maximum(eng, 1e-12)
        present = np.where(nwin > 0)[0]
        if present.size == 0:
            raise RuntimeError(f"Split {split_name!r} has no evaluable windows.")
        rel_present = rel[present]
        score = float(rel_present.mean() + self.cfg.alpha_worst * rel_present.max())
        return score, rel, nwin, event_stats(ev_sum, ev_sumsq, ev_n, self.data.label_names)

    def optimize(self) -> dict[str, Any]:
        if self.cfg.resume and self.checkpoint_path.exists():
            ck = load_json(self.checkpoint_path)
            if ck.get("status") == "done":
                self.logger.info("Completed checkpoint found: %s", self.checkpoint_path)
                return ck

        t0 = time.perf_counter()
        theta = clip_theta(self.theta_init, self.cfg.theta_min, self.theta_max, self.cfg.round_decimals)
        theta0 = theta.copy()
        theta_history = [theta.copy()]
        history: list[dict[str, Any]] = []

        self.logger.info("=" * 100)
        self.logger.info("Theta optimization | dataset=%s | UPS=%s | Te=%d | dead_zone=%s", self.cfg.dataset, self.cfg.ups, self.data.out_len, self.cfg.dead_zone)
        self.logger.info("theta_init=%s", theta)
        self.logger.info("theta_max =%s", self.theta_max)
        self.logger.info("splits | train=%d | val=%d | test_sampled=%d | test_full=%d", self.data.train.y.size, self.data.val.y.size, self.data.test_sampled.y.size, self.data.test_full.y.size)

        pool = self._pool()
        try:
            tr_sc, tr_rel, tr_nw, _ = self.objective("train", theta, pool)
            va_sc, va_rel, va_nw, _ = self.objective("val", theta, pool)

            best = {
                "theta": theta.copy(),
                "val": float(va_sc),
                "epoch": 0,
                "train": float(tr_sc),
                "rel_train": tr_rel.copy(),
                "rel_val": va_rel.copy(),
                "nwin_train": tr_nw.copy(),
                "nwin_val": va_nw.copy(),
            }

            step = float(np.round(self.cfg.init_step, self.cfg.round_decimals))
            no_imp_global = 0
            no_imp_lr = 0
            self.logger.info("initial | train=%.6e | val=%.6e", tr_sc, va_sc)

            for epoch in range(1, int(self.cfg.max_epochs) + 1):
                epoch_t0 = time.perf_counter()
                improved_train = False

                for ch in range(self.data.n_channels):
                    base = float(theta[ch])
                    best_ch_score = float(tr_sc)
                    best_ch_theta = theta.copy()
                    best_ch_rel = tr_rel.copy()

                    for value in (base - step, base + step):
                        cand = theta.copy()
                        cand[ch] = float(value)
                        cand = clip_theta(cand, self.cfg.theta_min, self.theta_max, self.cfg.round_decimals)
                        sc, rel, _, _ = self.objective("train", cand, pool)
                        if sc + self.cfg.improve_tol < best_ch_score:
                            best_ch_score = float(sc)
                            best_ch_theta = cand.copy()
                            best_ch_rel = rel.copy()

                    if best_ch_score + self.cfg.improve_tol < tr_sc:
                        theta = best_ch_theta.copy()
                        tr_sc = float(best_ch_score)
                        tr_rel = best_ch_rel.copy()
                        improved_train = True

                va_sc, va_rel, va_nw, _ = self.objective("val", theta, pool)
                improved_val = va_sc + self.cfg.improve_tol < float(best["val"])
                lr_reduced = False

                if improved_val:
                    best.update({
                        "theta": theta.copy(),
                        "val": float(va_sc),
                        "epoch": int(epoch),
                        "train": float(tr_sc),
                        "rel_train": tr_rel.copy(),
                        "rel_val": va_rel.copy(),
                        "nwin_train": tr_nw.copy(),
                        "nwin_val": va_nw.copy(),
                    })
                    no_imp_global = 0
                    no_imp_lr = 0
                    status = "improved"
                else:
                    no_imp_global += 1
                    no_imp_lr += 1
                    status = "no_improvement"
                    if no_imp_lr >= self.cfg.patience_lr:
                        new_step = max(self.cfg.min_step, float(np.round(step * self.cfg.step_shrink, self.cfg.round_decimals)))
                        if new_step < step:
                            step = new_step
                            lr_reduced = True
                        no_imp_lr = 0

                theta_history.append(theta.copy())
                row = {
                    "epoch": int(epoch),
                    "step": float(step),
                    "theta": theta.copy(),
                    "train_score": float(tr_sc),
                    "val_score": float(va_sc),
                    "best_val": float(best["val"]),
                    "rel_val": va_rel.copy(),
                    "status": status,
                    "lr_reduced": bool(lr_reduced),
                }
                history.append(row)
                self._save_checkpoint("running", theta0, theta, best, theta_history, history, epoch, tr_sc, va_sc, tr_rel, va_rel)

                if self.cfg.print_every_epoch:
                    theta_str = " | ".join(f"θ{i}={theta[i]:.{self.cfg.round_decimals}f}" for i in range(self.data.n_channels))
                    extra = " | step_down" if lr_reduced else ""
                    self.logger.info(
                        "epoch=%03d | train=%.6e | val=%.6e | best=%.6e | step=%.*f | patience=%d/%d | %s%s | %.1fs | %s",
                        epoch,
                        tr_sc,
                        va_sc,
                        float(best["val"]),
                        self.cfg.round_decimals,
                        step,
                        no_imp_global,
                        self.cfg.patience_global,
                        status,
                        extra,
                        time.perf_counter() - epoch_t0,
                        theta_str,
                    )

                if no_imp_global >= self.cfg.patience_global:
                    self.logger.info("early stop: patience_global=%d", self.cfg.patience_global)
                    break
                if step <= self.cfg.min_step and not improved_train:
                    self.logger.info("early stop: min_step reached with no train improvement")
                    break

            result = self._finalize(theta0, np.asarray(best["theta"], dtype=np.float32), theta_history, history, int(best["epoch"]), pool, time.perf_counter() - t0)
            save_json_atomic(self.checkpoint_path, result)
            save_json_atomic(self.summary_path, result)
            return result
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=False)
            self._cleanup_worker_cache()

    def _save_checkpoint(
        self,
        status: str,
        theta0: np.ndarray,
        theta: np.ndarray,
        best: dict[str, Any],
        theta_history: list[np.ndarray],
        history: list[dict[str, Any]],
        epoch: int,
        train_score: float,
        val_score: float,
        rel_train: np.ndarray,
        rel_val: np.ndarray,
    ) -> None:
        save_json_atomic(
            self.checkpoint_path,
            {
                "status": status,
                "dataset": self.cfg.dataset,
                "ups": float(self.cfg.ups),
                "dead_zone": float(self.cfg.dead_zone),
                "theta_init": theta0,
                "theta_current": theta,
                "theta_best": best["theta"],
                "theta_max": self.theta_max,
                "theta_history": np.asarray(theta_history, dtype=np.float32),
                "last_epoch": int(epoch),
                "best_epoch": int(best["epoch"]),
                "scores_partial": {"train": float(train_score), "val": float(val_score), "best_val": float(best["val"])},
                "rel_err_partial": {"train": rel_train, "val": rel_val},
                "history": history,
            },
        )

    def _finalize(
        self,
        theta0: np.ndarray,
        theta_best: np.ndarray,
        theta_history: list[np.ndarray],
        history: list[dict[str, Any]],
        best_epoch: int,
        pool: Any,
        elapsed: float,
    ) -> dict[str, Any]:
        scores: dict[str, float] = {}
        rel_err: dict[str, np.ndarray] = {}
        n_windows: dict[str, np.ndarray] = {}
        ev_stats: dict[str, Any] = {}

        for split_name in ("train", "val", "test_sampled", "test_full"):
            score, rel, nwin, stats = self.objective(split_name, theta_best, pool)
            scores[split_name] = float(score)
            rel_err[split_name] = rel
            n_windows[split_name] = nwin
            ev_stats[split_name] = stats

        result = {
            "status": "done",
            "timestamp": datetime.now().isoformat(),
            "dataset": self.cfg.dataset,
            "ups": float(self.cfg.ups),
            "raw_win": int(self.data.raw_win),
            "out_len": int(self.data.out_len),
            "dead_zone": float(self.cfg.dead_zone),
            "interp_method": self.cfg.interp_method,
            "sd_init": self.cfg.sd_init,
            "theta_init": theta0,
            "theta_best": theta_best,
            "theta_max": self.theta_max,
            "theta_history": np.asarray(theta_history, dtype=np.float32),
            "best_epoch": int(best_epoch),
            "scores": scores,
            "rel_err": rel_err,
            "n_windows": n_windows,
            "event_stats": ev_stats,
            "history": history,
            "metadata": self.data.metadata,
            "config": {
                "workers": int(self.cfg.workers),
                "executor": self._executor_kind(),
                "chunk_size": int(self.cfg.chunk_size),
                "max_epochs": int(self.cfg.max_epochs),
                "patience_global": int(self.cfg.patience_global),
                "patience_lr": int(self.cfg.patience_lr),
                "init_step": float(self.cfg.init_step),
                "min_step": float(self.cfg.min_step),
                "alpha_worst": float(self.cfg.alpha_worst),
                "step_shrink": float(self.cfg.step_shrink),
                "improve_tol": float(self.cfg.improve_tol),
                "round_decimals": int(self.cfg.round_decimals),
                "theta_min": float(self.cfg.theta_min),
                "theta_max_pctl": float(self.cfg.theta_max_pctl),
                "theta_max_scale": float(self.cfg.theta_max_scale),
                "theta_max_floor": float(self.cfg.theta_max_floor),
                "theta_max_init_margin": float(self.cfg.theta_max_init_margin),
                "train_per_class": int(self.cfg.train_per_class),
                "val_per_class": int(self.cfg.val_per_class),
                "test_per_class": int(self.cfg.test_per_class),
                "final_test_all": bool(self.cfg.final_test_all),
            },
            "elapsed_sec": float(elapsed),
        }

        self.logger.info("=" * 100)
        self.logger.info("Finished | best_epoch=%d | theta_best=%s", best_epoch, theta_best)
        self.logger.info("scores=%s", {k: f"{v:.6e}" for k, v in scores.items()})
        self.logger.info("summary=%s", self.summary_path)
        self.logger.info("elapsed=%.2f min", elapsed / 60.0)
        return result
