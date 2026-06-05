#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Energy-estimation pipeline for trained SCN runs.

The operation-level accounting follows the SNN-HAR energy analysis of
Li et al. (2023), which estimates compute and data-movement energy for
spiking HAR models using the SATA/SATA_Sim sparsity-aware simulator.

References:
    Li Y, Yin R, Kim Y, Panda P. Efficient human activity recognition with
    spatio-temporal spiking neural networks. Frontiers in Neuroscience.
    2023;17:1233037. doi: 10.3389/fnins.2023.1233037

    Yin R, Moitra A, Bhattacharjee A, Kim Y, Panda P. SATA: Sparsity-Aware
    Training Accelerator for Spiking Neural Networks. IEEE Transactions on
    Computer-Aided Design of Integrated Circuits and Systems.
    2023;42(6):1926-1938. doi: 10.1109/TCAD.2022.3213211

Code references:
    SNN_HAR:
        https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR

    SATA_Sim:
        https://github.com/RuokaiYin/SATA_Sim

Notes:
    Energy values are normalized to one dense ANN MAC. The implementation
    uses the normalized PE-level costs reported by Li et al. for accumulate,
    LIF update, input-scratchpad access, and weight-scratchpad access.
    This is an inference-stage estimate and does not include the analog
    event-generation front-end.
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from preprocess.utils import json_ready, load_json, save_json
from energy.utils import (
    ENERGY,
    count_nonzero_numel,
    discover_run_dirs,
    first_module_by_class_name,
    first_module_of_type,
    load_checkpoint,
    safe_float,
    save_csv,
    scalar_or_string,
)


class SCNEnergyProfiler:
    """Forward-hook profiler for SCN inference energy."""

    def __init__(self, model: nn.Module, include_data_movement: bool = True, offset_as_dense_mac: bool = True) -> None:
        self.model = model
        self.include_data_movement = bool(include_data_movement)
        self.offset_as_dense_mac = bool(offset_as_dense_mac)
        self.layer_stats = defaultdict(self._new_layer_stat)
        self.activation_stats = defaultdict(lambda: {"nonzero": 0, "numel": 0, "calls": 0, "last_shape": ""})
        self.handles: list[Any] = []
        self.total_samples = 0

    @staticmethod
    def _new_layer_stat() -> dict[str, Any]:
        return {
            "kind": "",
            "group": "",
            "dense_macs": 0.0,
            "effective_sops": 0.0,
            "input_nonzero": 0,
            "input_numel": 0,
            "lif_ops": 0,
            "lif_nonzero": 0,
            "lif_numel": 0,
            "calls": 0,
            "last_input_shape": "",
            "last_output_shape": "",
            "last_lif_shape": "",
            "notes": "",
        }

    def register(self) -> None:
        for idx in range(1, 4):
            block = getattr(self.model, f"conv_block{idx}", None)
            self._register_conv_block(f"conv{idx}", block)
            self._register_activation(f"conv{idx}_pool_out", block)

        if hasattr(self.model, "offset_proj_spk"):
            self._register_spiking_linear_block(
                "offset_spk",
                self.model.offset_proj_spk,
                group="offset",
                analog_or_dense_input=self.offset_as_dense_mac,
                notes="Offset branch counted as dense MACs when offset_as_dense_mac is enabled.",
            )

        if getattr(self.model, "spiking_heads", False):
            for name in ("head1_spk", "head2_spk"):
                if hasattr(self.model, name):
                    self._register_spiking_linear_block(name, getattr(self.model, name), group="head", analog_or_dense_input=False)
        else:
            for name in ("head1", "head2"):
                if hasattr(self.model, name):
                    self._register_dense_linear(f"{name}_dense", getattr(self.model, name), group="head")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def add_batch(self, batch_size: int) -> None:
        self.total_samples += int(batch_size)

    def _register_conv_block(self, name: str, block: nn.Module | None) -> None:
        if block is None:
            return
        conv = first_module_of_type(block, nn.Conv1d)
        lif = first_module_by_class_name(block, "LIFSpike")
        if conv is None:
            return

        def conv_hook(mod: nn.Conv1d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x = inputs[0]
            dense_macs = float(x.shape[0] * output.shape[-1] * mod.out_channels * (mod.in_channels // mod.groups) * mod.kernel_size[0])
            nonzero, numel = count_nonzero_numel(x)
            stat = self.layer_stats[name]
            stat.update(kind="spiking_conv1d", group="backbone", last_input_shape=str(tuple(x.shape)), last_output_shape=str(tuple(output.shape)))
            stat["dense_macs"] += dense_macs
            stat["effective_sops"] += dense_macs * _rate(nonzero, numel, default=0.0)
            stat["input_nonzero"] += nonzero
            stat["input_numel"] += numel
            stat["calls"] += 1

        self.handles.append(conv.register_forward_hook(conv_hook))
        if lif is not None:
            self.handles.append(lif.register_forward_hook(lambda _m, _i, y, layer=name: self._add_lif(layer, y)))

    def _register_activation(self, name: str, module: nn.Module | None) -> None:
        if module is None:
            return

        def hook(_mod: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            nonzero, numel = count_nonzero_numel(output)
            stat = self.activation_stats[name]
            stat["nonzero"] += nonzero
            stat["numel"] += numel
            stat["calls"] += 1
            stat["last_shape"] = str(tuple(output.shape))

        self.handles.append(module.register_forward_hook(hook))

    def _register_spiking_linear_block(self, name: str, block: nn.Module, group: str, analog_or_dense_input: bool, notes: str = "") -> None:
        fc = getattr(block, "fc", None) or first_module_of_type(block, nn.Linear)
        lif = getattr(block, "lif", None)
        if fc is None:
            return

        def fc_hook(_mod: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x = inputs[0]
            dense_macs = float(x.shape[0] * x.shape[-1] * output.shape[-1])
            nonzero, numel = count_nonzero_numel(x)
            stat = self.layer_stats[name]
            stat.update(
                kind="spiking_linear_dense_input" if analog_or_dense_input else "spiking_linear_sparse_input",
                group=group,
                last_input_shape=str(tuple(x.shape)),
                last_output_shape=str(tuple(output.shape)),
                notes=notes,
            )
            stat["dense_macs"] += dense_macs
            stat["effective_sops"] += dense_macs if analog_or_dense_input else dense_macs * _rate(nonzero, numel, default=0.0)
            stat["input_nonzero"] += nonzero
            stat["input_numel"] += numel
            stat["calls"] += 1

        self.handles.append(fc.register_forward_hook(fc_hook))
        if lif is not None:
            self.handles.append(lif.register_forward_hook(lambda _m, _i, y, layer=name: self._add_lif(layer, y)))

    def _register_dense_linear(self, name: str, layer: nn.Linear, group: str) -> None:
        def hook(_mod: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x = inputs[0]
            dense_macs = float(x.shape[0] * x.shape[-1] * output.shape[-1])
            nonzero, numel = count_nonzero_numel(x)
            stat = self.layer_stats[name]
            stat.update(kind="dense_linear", group=group, last_input_shape=str(tuple(x.shape)), last_output_shape=str(tuple(output.shape)))
            stat["dense_macs"] += dense_macs
            stat["effective_sops"] += dense_macs
            stat["input_nonzero"] += nonzero
            stat["input_numel"] += numel
            stat["calls"] += 1

        self.handles.append(layer.register_forward_hook(hook))

    def _add_lif(self, layer: str, output: torch.Tensor) -> None:
        nonzero, numel = count_nonzero_numel(output)
        stat = self.layer_stats[layer]
        stat["lif_ops"] += numel
        stat["lif_nonzero"] += nonzero
        stat["lif_numel"] += numel
        stat["last_lif_shape"] = str(tuple(output.shape))

    def summarize(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        n_samples = max(int(self.total_samples), 1)
        rows: list[dict[str, Any]] = []
        totals = defaultdict(float)
        group_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for name, stat in self.layer_stats.items():
            energy = self._layer_energy(stat)
            group = stat["group"] or "other"
            row = {
                "layer": name,
                "group": group,
                "kind": stat["kind"],
                "calls": stat["calls"],
                "input_rate": _rate(stat["input_nonzero"], stat["input_numel"]),
                "lif_output_rate": _rate(stat["lif_nonzero"], stat["lif_numel"]),
                "dense_equiv_macs_total": stat["dense_macs"],
                "effective_sops_total": stat["effective_sops"],
                "lif_ops_total": stat["lif_ops"],
                "dense_equiv_macs_per_sample": stat["dense_macs"] / n_samples,
                "effective_sops_per_sample": stat["effective_sops"] / n_samples,
                "lif_ops_per_sample": stat["lif_ops"] / n_samples,
                "last_input_shape": stat["last_input_shape"],
                "last_output_shape": stat["last_output_shape"],
                "last_lif_shape": stat["last_lif_shape"],
                "notes": stat["notes"],
            }
            for key, value in energy.items():
                row[f"{key}_total"] = value
                row[f"{key}_per_sample"] = value / n_samples
                totals[key] += value
                group_totals[group][key] += value
            for key in ("dense_macs", "effective_sops", "lif_ops"):
                totals[key] += float(stat[key])
                group_totals[group][key] += float(stat[key])
            rows.append(row)

        dense_equiv = float(totals["dense_macs"])
        compute = float(totals["energy_compute_norm"])
        pe_total = float(totals["energy_pe_total_norm"])
        summary = {
            "n_samples": n_samples,
            "dense_equiv_macs_total": dense_equiv,
            "effective_sops_total": float(totals["effective_sops"]),
            "lif_ops_total": float(totals["lif_ops"]),
            "dense_equiv_macs_per_sample": dense_equiv / n_samples,
            "effective_sops_per_sample": float(totals["effective_sops"]) / n_samples,
            "lif_ops_per_sample": float(totals["lif_ops"]) / n_samples,
            "energy_compute_norm_total": compute,
            "energy_data_norm_total": float(totals["energy_data_norm"]),
            "energy_pe_total_norm_total": pe_total,
            "energy_compute_norm_per_sample": compute / n_samples,
            "energy_data_norm_per_sample": float(totals["energy_data_norm"]) / n_samples,
            "energy_pe_total_norm_per_sample": pe_total / n_samples,
            "energy_compute_reduction_vs_dense_equiv": 1.0 - compute / dense_equiv if dense_equiv > 0 else np.nan,
            "energy_pe_total_reduction_vs_dense_equiv": 1.0 - pe_total / dense_equiv if dense_equiv > 0 else np.nan,
            "include_data_movement": self.include_data_movement,
            "offset_as_dense_mac": self.offset_as_dense_mac,
            "energy_constants": dict(ENERGY),
            "activation_summary": self._activation_summary(),
            "group_totals": {g: {f"{k}_per_sample": float(v) / n_samples for k, v in vals.items()} for g, vals in group_totals.items()},
        }
        return summary, rows

    def _activation_summary(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "rate": _rate(stat["nonzero"], stat["numel"]),
                "nonzero": int(stat["nonzero"]),
                "numel": int(stat["numel"]),
                "calls": int(stat["calls"]),
                "last_shape": stat["last_shape"],
            }
            for name, stat in self.activation_stats.items()
        }

    def _layer_energy(self, stat: dict[str, Any]) -> dict[str, float]:
        kind = stat["kind"]
        dense_macs = float(stat["dense_macs"])
        effective_sops = float(stat["effective_sops"])
        lif_ops = float(stat["lif_ops"])
        if kind in {"spiking_conv1d", "spiking_linear_sparse_input"}:
            compute = effective_sops * ENERGY["E_SNN_AC"] + lif_ops * ENERGY["E_LIF"]
            data = effective_sops * (ENERGY["E_ISPAD"] + ENERGY["E_WSPAD"]) if self.include_data_movement else 0.0
        elif kind == "spiking_linear_dense_input":
            compute = dense_macs * ENERGY["E_DENSE_MAC"] + lif_ops * ENERGY["E_LIF"]
            data = 0.0
        elif kind == "dense_linear":
            compute = dense_macs * ENERGY["E_DENSE_MAC"]
            data = 0.0
        else:
            compute = data = 0.0
        return {"energy_compute_norm": compute, "energy_data_norm": data, "energy_pe_total_norm": compute + data}


def evaluate_run(
    train_module: str,
    run_dir: str | Path,
    device: str | torch.device = "auto",
    dataset: str = "auto",
    batch_size: int | None = None,
    num_workers: int = 0,
    max_batches: int | None = None,
    include_data_movement: bool = True,
    offset_as_dense_mac: bool = True,
    output_prefix: str = "energy",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = _load_run_cfg(run_dir, batch_size=batch_size, num_workers=num_workers)
    dataset_name = infer_dataset(dataset, train_module, cfg, run_dir)
    dev = resolve_device(device)
    train_mod = importlib.import_module(train_module)

    model = train_mod.build_model(cfg, dev)
    load_checkpoint(model, run_dir / "checkpoints" / getattr(cfg, "ckpt_name", "best_model.pt"), dev)

    loader = train_mod.build_data(cfg, dev)["loaders"]["test"]
    profiler = SCNEnergyProfiler(model, include_data_movement=include_data_movement, offset_as_dense_mac=offset_as_dense_mac)
    profiler.register()
    try:
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if max_batches is not None and int(max_batches) > 0 and index >= int(max_batches):
                    break
                x = batch["x"].to(dev, non_blocking=True).float()
                offset = batch["offset"].to(dev, non_blocking=True).float()
                profiler.add_batch(int(x.shape[0]))
                model(x, offset=offset)
    finally:
        profiler.remove()

    summary, layer_rows = profiler.summarize()
    run_summary = _make_run_summary(run_dir, cfg, dataset_name, summary, _read_json(run_dir / "metrics" / "test_metrics.json", default={}))
    metrics_dir = run_dir / "metrics"
    report_path = metrics_dir / f"{output_prefix}_report_{dataset_name}.json"
    layers_path = metrics_dir / f"{output_prefix}_layers_{dataset_name}.csv"
    save_json(report_path, {"summary": summary, "run_summary": run_summary, "layers": layer_rows})
    save_csv(layer_rows, layers_path)
    return {**run_summary, "energy_report_path": str(report_path), "energy_layers_path": str(layers_path)}


def evaluate_runs(
    train_module: str | None = None,
    runs_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    device: str | torch.device = "auto",
    dataset: str = "auto",
    batch_size: int | None = None,
    num_workers: int = 0,
    max_batches: int | None = None,
    include_data_movement: bool = True,
    offset_as_dense_mac: bool = True,
    skip_incomplete: bool = True,
    output_prefix: str = "energy",
) -> dict[str, Any]:
    dataset = dataset.lower()
    if train_module is None:
        if dataset == "auto":
            raise ValueError("Use a dataset or provide train_module.")
        train_module = f"train_{dataset}"

    run_dirs = discover_run_dirs(runs_root=runs_root, run_dir=run_dir)
    if dataset != "auto":
        run_dirs = [path for path in run_dirs if dataset in path.name.lower()]

    summaries: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for index, path in enumerate(run_dirs, start=1):
        ckpt_path = _checkpoint_path(path)
        if not ckpt_path.is_file():
            message = f"Missing checkpoint: {ckpt_path}"
            if skip_incomplete:
                print(f"[skip {index}/{len(run_dirs)}] {path.name} | {message}")
                continue
            raise FileNotFoundError(message)
        try:
            print(f"[run {index}/{len(run_dirs)}] {path.name}")
            row = evaluate_run(
                train_module=train_module,
                run_dir=path,
                device=device,
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                max_batches=max_batches,
                include_data_movement=include_data_movement,
                offset_as_dense_mac=offset_as_dense_mac,
                output_prefix=output_prefix,
            )
            summaries.append(row)
            print(_format_energy_line(row))
        except Exception as exc:
            print(f"[failed] {path}: {exc}")
            failed.append({"run_dir": str(path), "error": str(exc)})

    output_root = Path(runs_root) if run_dir is None else Path(run_dir).parent
    dataset_name = _summary_dataset_name(dataset, train_module, summaries)
    summary_csv = output_root / f"{output_prefix}_summary_{dataset_name}.csv"
    summary_json = output_root / f"{output_prefix}_summary_{dataset_name}.json"
    summaries = sorted(summaries, key=lambda row: safe_float(row.get("energy_compute_norm_per_sample"), default=float("inf")))
    if summaries:
        save_csv(summaries, summary_csv)
        save_json(summary_json, {"summaries": summaries, "failed": failed})
    return {"summaries": summaries, "failed": failed, "summary_csv": str(summary_csv), "summary_json": str(summary_json)}


def infer_dataset(dataset: str, train_module: str, cfg: Any, run_dir: str | Path | None = None) -> str:
    if dataset != "auto":
        return dataset.lower()
    text = " ".join(str(x).lower() for x in (train_module, getattr(cfg, "run", ""), getattr(cfg, "requested_run", ""), run_dir or ""))
    if "ucihar" in text or "uci" in text:
        return "ucihar"
    if "sisfall" in text:
        return "sisfall"
    return "scn"


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _load_run_cfg(run_dir: Path, batch_size: int | None, num_workers: int) -> SimpleNamespace:
    cfg = SimpleNamespace(**load_json(run_dir / "config.json"))
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if hasattr(cfg, "workers"):
        cfg.workers = int(num_workers)
    if hasattr(cfg, "num_workers"):
        cfg.num_workers = int(num_workers)
    for name in ("refresh_cache", "refresh_posture"):
        if hasattr(cfg, name):
            setattr(cfg, name, False)
    if hasattr(cfg, "quiet_loader"):
        cfg.quiet_loader = True
    return cfg


def _checkpoint_path(run_dir: Path) -> Path:
    cfg = _read_json(run_dir / "config.json", default={})
    return run_dir / "checkpoints" / str(cfg.get("ckpt_name", "best_model.pt"))


def _make_run_summary(run_dir: Path, cfg: Any, dataset: str, summary: dict[str, Any], test_metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"run_name": getattr(cfg, "run", run_dir.name), "run_dir": str(run_dir), "dataset": dataset}
    for key in _config_summary_keys(dataset):
        if hasattr(cfg, key):
            value = scalar_or_string(getattr(cfg, key))
            if value is not None:
                row[key] = value
    for key, value in test_metrics.items():
        value = scalar_or_string(value)
        if value is not None:
            row[f"test_{key}" if not str(key).startswith("test_") else str(key)] = value
    for key, value in summary.items():
        if key not in {"activation_summary", "group_totals", "energy_constants"}:
            row[key] = value
    for name, values in summary.get("activation_summary", {}).items():
        row[f"{name}_rate"] = values.get("rate", np.nan)
    for group, values in summary.get("group_totals", {}).items():
        for key, value in values.items():
            row[f"{group}_{key}"] = value
    return json_ready(row)


def _config_summary_keys(dataset: str) -> tuple[str, ...]:
    common = (
        "model_name",
        "head_mode",
        "spiking_heads",
        "n_ch",
        "n_classes",
        "conv_ch",
        "kernels",
        "strides",
        "pool_kernels",
        "pool_strides",
        "pool_paddings",
        "tau",
        "thresh",
        "hard_reset",
        "p_drop",
        "merge_polarities",
        "offset_hidden",
        "head_rate_scale",
        "batch_size",
    )
    if dataset == "ucihar":
        return common + ("case", "target_domain", "ups", "dead_zone", "theta", "cache_name")
    if dataset == "sisfall":
        return common + ("raw_win", "out_win", "dead_zone", "theta", "cache_name", "single_ambiguous")
    return common


def _read_json(path: Path, default: Any) -> Any:
    return load_json(path) if path.is_file() else default


def _rate(nonzero: int, numel: int, default: float = np.nan) -> float:
    return default if int(numel) == 0 else float(nonzero) / float(numel)


def _summary_dataset_name(dataset: str, train_module: str, summaries: list[dict[str, Any]]) -> str:
    if summaries:
        return str(summaries[0].get("dataset", "scn"))
    if dataset != "auto":
        return dataset.lower()
    return "ucihar" if "ucihar" in train_module.lower() else "sisfall" if "sisfall" in train_module.lower() else "scn"


def _format_energy_line(row: dict[str, Any]) -> str:
    parts = [
        f"E_compute={safe_float(row.get('energy_compute_norm_per_sample')):.3e}",
        f"E_pe_total={safe_float(row.get('energy_pe_total_norm_per_sample')):.3e}",
        f"dense_equiv={safe_float(row.get('dense_equiv_macs_per_sample')):.3e}",
    ]
    for key in ("test_acc", "test_acc_h1", "test_acc_h2"):
        if key in row:
            parts.append(f"{key}={safe_float(row.get(key)):.4f}")
    return "[energy] " + " | ".join(parts)
