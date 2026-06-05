#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theta optimization for Sigma-Delta event encoding.

This script optimizes the 6-channel threshold vector theta used by the
Sigma-Delta encoder. It prepares the selected dataset, interpolates each
IMU window to the requested temporal resolution, reconstructs the signal
from events, and minimizes the class-balanced relative reconstruction error.

Examples:

    # SisFall
    python optimize_theta.py --dataset sisfall --ups 5 --dead-zone 0.5

    # UCI HAR
    python optimize_theta.py --dataset ucihar --ups 5 --dead-zone 0.5

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _csv_float_list(value: str | None) -> tuple[float, ...] | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = tuple(float(v.strip()) for v in str(value).split(",") if v.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats, for example: 1,2,3,4,5,6") from exc
    if len(out) != 6:
        raise argparse.ArgumentTypeError("theta vectors must contain exactly 6 values.")
    return out


def build_parser() -> argparse.ArgumentParser:
    cpu_default = max(1, min(8, (os.cpu_count() or 2) - 1))
    p = argparse.ArgumentParser(
        description="Optimize one Sigma-Delta theta vector for one dataset, one UPS value, and one dead-zone value.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required run identity.
    p.add_argument("--dataset", choices=("sisfall", "ucihar"), required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--ups", type=float, required=True)
    p.add_argument("--dead-zone", type=float, required=True)

    # Output and execution.
    p.add_argument("--results-root", default="results/theta_optimization")
    p.add_argument("--workers", type=int, default=cpu_default)
    p.add_argument("--executor", choices=("auto", "process", "thread", "none"), default="auto")
    p.add_argument("--keep-worker-cache", action="store_true", help="Keep temporary .npy arrays used by process workers.")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Ignore a completed checkpoint and run again.")
    p.add_argument("--quiet-epochs", action="store_true", help="Do not print every optimization epoch.")

    # Interpolation / encoding.
    p.add_argument("--out-len", type=int, default=None, help="Interpolated temporal length. If omitted, it is inferred from dataset and UPS.")
    p.add_argument("--interp-method", choices=("linear", "zoh", "cubic"), default="linear")
    p.add_argument("--sd-init", choices=("x0", "zero"), default="x0")

    # Optimization controls. If omitted, dataset-specific defaults are used.
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience-global", type=int, default=None)
    p.add_argument("--patience-lr", type=int, default=None)
    p.add_argument("--init-step", type=float, default=None)
    p.add_argument("--min-step", type=float, default=None)
    p.add_argument("--step-shrink", type=float, default=0.5)
    p.add_argument("--improve-tol", type=float, default=1e-7)
    p.add_argument("--alpha-worst", type=float, default=None)
    p.add_argument("--round-decimals", type=int, default=None)

    # Theta bounds / initialization. If omitted, dataset-specific defaults are used.
    p.add_argument("--theta-init", type=_csv_float_list, default=None)
    p.add_argument("--theta-min", type=float, default=None)
    p.add_argument("--theta-max", type=_csv_float_list, default=None)
    p.add_argument("--theta-max-pctl", type=float, default=None)
    p.add_argument("--theta-max-scale", type=float, default=None)
    p.add_argument("--theta-max-floor", type=float, default=None)
    p.add_argument("--theta-max-init-margin", type=float, default=None)

    # Balanced sampling. Values mean windows per class for both datasets.
    p.add_argument("--train-per-class", type=int, default=None)
    p.add_argument("--val-per-class", type=int, default=None)
    p.add_argument("--test-per-class", type=int, default=None)
    p.add_argument("--sampled-test-only", action="store_true", help="Do not evaluate the full test split at the end.")

    # SisFall window extraction. These are ignored by already-windowed datasets.
    p.add_argument("--window-stride", type=int, default=None)
    p.add_argument("--fall-policy", choices=("contain_peak", "all"), default="contain_peak")
    p.add_argument("--acc-sensor", choices=("ADXL345", "MMA8451Q"), default="ADXL345")

    # UCI HAR preprocessing options.
    p.add_argument("--case", default="random", choices=("official", "random", "subject", "subject_large"))
    p.add_argument("--target-domain", type=int, default=0)
    p.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    return p


def main() -> dict:
    args = build_parser().parse_args()

    repo_root = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    for path in (repo_root, cwd):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from theta_optimization.utils import RunConfig, ThetaOptimizer, prepare_data, setup_logger

    cfg = RunConfig.from_args(args)
    logger = setup_logger("optimize_theta", cfg.result_dir / "logs")

    logger.info("dataset=%s | data_root=%s | UPS=%s | dead_zone=%s", cfg.dataset, cfg.data_root, cfg.ups, cfg.dead_zone)
    logger.info("results=%s | workers=%d | executor=%s | seed=%d | resume=%s", cfg.result_dir, cfg.workers, cfg.executor, cfg.seed, cfg.resume)
    logger.info("raw_win=%d | out_len=%d | interp=%s | sd_init=%s", cfg.raw_win, cfg.out_len, cfg.interp_method, cfg.sd_init)
    logger.info("train/val/test per class=%s/%s/%s", cfg.train_per_class, cfg.val_per_class, cfg.test_per_class)
    logger.info("max_epochs=%d | patience=%d | patience_lr=%d | init_step=%s | min_step=%s", cfg.max_epochs, cfg.patience_global, cfg.patience_lr, cfg.init_step, cfg.min_step)

    data = prepare_data(cfg, logger)
    result = ThetaOptimizer(data, cfg, logger).optimize()
    logger.info("theta_best=%s", result.get("theta_best"))
    return result


if __name__ == "__main__":
    main()
