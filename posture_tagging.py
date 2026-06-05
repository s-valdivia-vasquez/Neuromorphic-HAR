"""Command-line entry point for SisFall posture refinement tagging.

Examples:
    python posture_tagging.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeling.sisfall_posture_tagging import TaggingConfig, run_posture_tagging
from preprocess.utils import load_json

def _path_value(cfg: dict[str, Any], key: str, default: str, override: str | None) -> str:
    if override is not None:
        return override
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    return str(cfg.get(key, paths.get(key, default)))


def _build_config(args: argparse.Namespace, cfg_json: dict[str, Any]) -> TaggingConfig:
    vals = asdict(TaggingConfig())
    vals.update({k: v for k, v in cfg_json.items() if k in vals})
    vals.update({k: v for k, v in cfg_json.get("posture_tagging", {}).items() if k in vals})

    for k in vals:
        v = getattr(args, k, None)
        if v is not None:
            vals[k] = v

    return TaggingConfig(**vals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SisFall window-level stable-posture and postural-transition autotagging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Optional JSON config file. Values under 'posture_tagging' are supported.")
    p.add_argument("--dataset-root", default=None, help="Root directory containing SisFall TXT files.")
    p.add_argument("--label-dir", default=None, help="Directory containing manual posture-label JSON files.")
    p.add_argument("--result-dir", default=None, help="Directory where posture-tagging outputs are written.")
    p.add_argument("--cache-dir", default=None, help="Directory used for reusable posture-tagging artifacts.")
    p.add_argument("--run-name", default=None, help="Name used to group output and cache artifacts.")
    p.add_argument("--refresh", action="store_true", help="Rebuild posture-tagging artifacts even when a valid cache exists.")
    p.add_argument("--refresh-raw", action="store_true", help="Ask the raw SisFall loader to rebuild its own cache when supported.")
    p.add_argument("--quiet", action="store_true", help="Disable high-level progress messages.")
    p.add_argument("--save-plot", action="store_true", help="Save a probability scatter plot over clear manual windows.")

    p.add_argument("--fs", type=float, default=None, help="Sensor sampling rate in Hz.")
    p.add_argument("--win", type=int, default=None, help="Window length in samples.")
    p.add_argument("--hop", type=int, default=None, help="Distance between consecutive windows in samples.")
    p.add_argument("--pad-hop", dest="pad_hop", type=int, default=None, help="Padding step used to provide context near signal boundaries.")
    p.add_argument("--pad-rep", dest="pad_rep", type=int, default=None, help="Number of repeated padding blocks at each boundary.")
    p.add_argument("--pca-k", dest="pca_k", type=int, default=None, help="Number of PCA components appended to normalized features; use 0 to disable PCA features.")
    p.add_argument("--ds-grav", dest="ds_grav", type=int, default=None, help="Downsampling factor for gravity direction variation features.")

    p.add_argument("--lr", type=float, default=None, help="Learning rate for the internal logistic-regression classifiers.")
    p.add_argument("--steps", type=int, default=None, help="Number of optimization steps for each binary classifier.")
    p.add_argument("--l2", type=float, default=None, help="L2 regularization strength for logistic-regression weights.")
    p.add_argument("--smooth-k", dest="smooth_k", type=int, default=None, help="Moving average size applied to output probabilities.")

    p.add_argument("--th-pt", dest="th_pt", type=float, default=None, help="Probability threshold for postural transition detection.")
    p.add_argument("--th-sp", dest="th_sp", type=float, default=None, help="Probability threshold for stable posture detection.")
    p.add_argument("--margin", type=float, default=None, help="Minimum probability gap between both posture classes.")
    p.add_argument("--amb-ceil", dest="amb_ceil", type=float, default=None, help="Maximum probability for both classes before forcing ambiguity.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-hyst", dest="use_hyst", action="store_true", default=None, help="Enable hysteresis over class probabilities.")
    g.add_argument("--no-hyst", dest="use_hyst", action="store_false", default=None, help="Disable hysteresis over class probabilities.")
    p.add_argument("--th-pt-on", dest="th_pt_on", type=float, default=None, help="Hysteresis activation threshold for postural transition.")
    p.add_argument("--th-pt-off", dest="th_pt_off", type=float, default=None, help="Hysteresis deactivation threshold for postural transition.")
    p.add_argument("--th-sp-on", dest="th_sp_on", type=float, default=None, help="Hysteresis activation threshold for stable posture.")
    p.add_argument("--th-sp-off", dest="th_sp_off", type=float, default=None, help="Hysteresis deactivation threshold for stable posture.")

    p.add_argument("--ywin-mode", dest="ywin_mode", choices=["mean", "center", "max", "q"], default=None, help="Manual-label aggregation mode for window targets.")
    p.add_argument("--ywin-q", dest="ywin_q", type=float, default=None, help="Quantile used when ywin-mode is q.")
    p.add_argument("--sigma-frac", dest="sigma_frac", type=float, default=None, help="Gaussian width factor for manual postural-transition intervals.")

    p.add_argument("--acc", choices=["ADXL345", "MMA8451Q"], default=None, help="Accelerometer source used from each SisFall sample.")
    p.add_argument("--dtype", default=None, help="Numeric dtype used when loading raw SisFall values.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_json = {} if args.config is None else load_json(args.config)
    cfg = _build_config(args, cfg_json)

    dataset_root = _path_value(cfg_json, "dataset_root", "data/SisFall_dataset", args.dataset_root)
    label_dir = _path_value(cfg_json, "label_dir", "data/labels_transitions", args.label_dir)
    result_dir = _path_value(cfg_json, "result_dir", "data/sisfall_tagging_results", args.result_dir)
    cache_dir = _path_value(cfg_json, "cache_dir", "data/sisfall_cache", args.cache_dir)
    run_name = args.run_name or str(cfg_json.get("run_name", "default"))

    run_posture_tagging(
        dataset_root=dataset_root,
        label_dir=label_dir,
        result_dir=result_dir,
        cache_dir=cache_dir,
        cfg=cfg,
        run_name=run_name,
        refresh=args.refresh,
        refresh_raw=args.refresh_raw,
        save_plot=args.save_plot,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
