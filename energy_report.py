#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command line entry point for SCN energy reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


class ArgumentFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate normalized inference-energy reports for trained SCN runs.",
        formatter_class=ArgumentFormatter,
        epilog=(
            "Examples:\n"
            "  python energy_report.py --dataset ucihar "
            "  python energy_report.py --dataset sisfall"
        ),
    )
    parser.add_argument("--dataset", choices=("ucihar", "sisfall"), required=True, help="Dataset used to select runs, defaults and output names.")
    parser.add_argument("--train-module", default=None, help="Override the default training module for the selected dataset.")
    parser.add_argument("--runs-root", default="runs", help="Directory containing run folders with config.json files.")
    parser.add_argument("--run-dir", default=None, help="Evaluate a single run directory instead of all runs under runs-root.")
    parser.add_argument("--project-root", default=None, help="Project root added to sys.path before importing local modules.")
    parser.add_argument("--device", default="auto", help="Device used for inference: auto, cpu, cuda:0, etc.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size stored in config.json.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers used by the report.")
    parser.add_argument("--max-batches", type=int, default=0, help="Use 0 for the full test split, or a positive value for a quick pass.")
    parser.add_argument("--include-data-movement", action=argparse.BooleanOptionalAction, default=True, help="Include scratchpad-access energy per SOP.")
    parser.add_argument("--no-data-movement", action="store_true", help="Compatibility alias for --no-include-data-movement.")
    parser.add_argument("--offset-as-sparse", action="store_true", help="Treat the offset branch as sparse instead of dense analog input.")
    parser.add_argument("--skip-incomplete", action=argparse.BooleanOptionalAction, default=True, help="Skip runs without checkpoints.")
    parser.add_argument("--output-prefix", default="energy", help="Prefix for generated report files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    add_project_root(args.project_root)

    from energy.measure_consumption import evaluate_runs

    result = evaluate_runs(
        train_module=args.train_module,
        runs_root=args.runs_root,
        run_dir=args.run_dir,
        device=args.device,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches if args.max_batches > 0 else None,
        include_data_movement=bool(args.include_data_movement and not args.no_data_movement),
        offset_as_dense_mac=not args.offset_as_sparse,
        skip_incomplete=args.skip_incomplete,
        output_prefix=args.output_prefix,
    )

    if result["summaries"]:
        print(f"[saved] {result['summary_csv']}")
        print(f"[saved] {result['summary_json']}")
    if result["failed"]:
        print("[failed]")
        for item in result["failed"]:
            print(f"  {item['run_dir']}: {item['error']}")


def add_project_root(project_root: str | None) -> None:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


if __name__ == "__main__":
    main()
