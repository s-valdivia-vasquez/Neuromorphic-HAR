#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loss helpers for dual-head HAR models."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def dual_head_head1_loss(
    logits_head1: torch.Tensor,
    y_head1: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss for the global activity head."""
    if logits_head1.dim() != 2:
        raise ValueError(f"logits_head1 must be 2D (B,C). Received {tuple(logits_head1.shape)}")

    y = y_head1.view(-1).long()
    kwargs: Dict[str, Any] = {}

    if class_weight is not None:
        kwargs["weight"] = class_weight.to(device=logits_head1.device, dtype=logits_head1.dtype)
    if float(label_smoothing) > 0.0:
        kwargs["label_smoothing"] = float(label_smoothing)

    return F.cross_entropy(logits_head1, y, **kwargs)


def dual_head_head2_loss(
    logits_head2: torch.Tensor,
    y_head2: torch.Tensor,
    y_head1: Optional[torch.Tensor] = None,
    static_class_idx: int = 2,
    mask_only_static: bool = True,
    ambiguous_value: float = 0.5,
    ignore_ambiguous: bool = True,
    ambiguous_weight: float = 0.0,
    sample_weight: Optional[torch.Tensor] = None,
    label_source: Optional[torch.Tensor] = None,
    gold_weight: float = 1.0,
    silver_weight: float = 0.6,
    pos_weight: Optional[Union[float, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Masked BCEWithLogits loss for the static-refinement head."""
    if logits_head2.dim() == 2 and logits_head2.shape[1] == 1:
        logits_head2 = logits_head2.squeeze(1)
    elif logits_head2.dim() != 1:
        raise ValueError(f"logits_head2 must be (B,) or (B,1). Received {tuple(logits_head2.shape)}")

    batch_size = int(logits_head2.shape[0])
    device = logits_head2.device
    dtype = logits_head2.dtype

    y2 = y_head2.view(-1).to(device=device, dtype=dtype)
    if int(y2.shape[0]) != batch_size:
        raise ValueError("y_head2 and logits_head2 must have the same batch size")

    if mask_only_static:
        if y_head1 is None:
            raise ValueError("mask_only_static=True requires y_head1")
        y1 = y_head1.view(-1).to(device=device)
        if int(y1.shape[0]) != batch_size:
            raise ValueError("y_head1 and logits_head2 must have the same batch size")
        mask_static = y1.long() == int(static_class_idx)
    else:
        mask_static = torch.ones(batch_size, device=device, dtype=torch.bool)

    amb_value = torch.tensor(float(ambiguous_value), device=device, dtype=dtype)
    mask_ambiguous = torch.isclose(y2, amb_value)

    if ignore_ambiguous:
        mask_labeled = ~mask_ambiguous
        y2_eff = y2.clamp(0.0, 1.0)
    else:
        mask_labeled = torch.ones_like(mask_ambiguous, dtype=torch.bool)
        y2_eff = y2.clamp(0.0, 1.0)

    mask_valid = mask_static & mask_labeled
    if not torch.any(mask_valid):
        zero = logits_head2.sum() * 0.0
        return zero, {
            "n_total": batch_size,
            "n_static": int(mask_static.sum().item()),
            "n_ambiguous": int((mask_static & mask_ambiguous).sum().item()),
            "n_valid": 0,
            "n_pos_valid": 0,
            "weight_sum": 0.0,
        }

    pw = None
    if pos_weight is not None:
        if torch.is_tensor(pos_weight):
            pw = pos_weight.to(device=device, dtype=dtype)
        else:
            pw = logits_head2.new_tensor(float(pos_weight))

    loss_vec = F.binary_cross_entropy_with_logits(logits_head2, y2_eff, reduction="none", pos_weight=pw)
    weights = torch.ones(batch_size, device=device, dtype=dtype)

    if sample_weight is not None:
        weights = weights * _batch_vector(sample_weight, batch_size, device, dtype, "sample_weight")
    elif label_source is not None:
        src = _batch_vector(label_source, batch_size, device, None, "label_source")
        is_gold = src.bool()
        weights = weights * torch.where(
            is_gold,
            torch.tensor(float(gold_weight), device=device, dtype=dtype),
            torch.tensor(float(silver_weight), device=device, dtype=dtype),
        )

    if not ignore_ambiguous and float(ambiguous_weight) != 1.0:
        weights = weights * torch.where(
            mask_ambiguous,
            torch.tensor(float(ambiguous_weight), device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )

    valid_weights = weights[mask_valid]
    valid_losses = loss_vec[mask_valid]
    denom = valid_weights.sum().clamp_min(float(eps))
    loss = (valid_losses * valid_weights).sum() / denom

    return loss, {
        "n_total": batch_size,
        "n_static": int(mask_static.sum().item()),
        "n_ambiguous": int((mask_static & mask_ambiguous).sum().item()),
        "n_valid": int(mask_valid.sum().item()),
        "n_pos_valid": int(((y2_eff > 0.5) & mask_valid).sum().item()),
        "weight_sum": float(valid_weights.sum().detach().item()),
    }


def _batch_vector(
    value: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: Optional[torch.dtype],
    name: str,
) -> torch.Tensor:
    vector = value.view(-1).to(device=device)
    if dtype is not None:
        vector = vector.to(dtype=dtype)
    if int(vector.shape[0]) != int(batch_size):
        raise ValueError(f"{name} must have batch size B")
    return vector


def dual_head_losses(
    outputs: Dict[str, torch.Tensor],
    y_head1: torch.Tensor,
    y_head2: Optional[torch.Tensor] = None,
    lambda_head1: float = 1.0,
    lambda_head2: float = 1.0,
    head1_kwargs: Optional[Dict[str, Any]] = None,
    head2_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combined loss helper for dual-head models."""
    if not isinstance(outputs, dict):
        raise TypeError("outputs must be a dict with keys 'head1' and 'head2'")
    if "head1" not in outputs or "head2" not in outputs:
        raise KeyError("outputs must contain keys 'head1' and 'head2'")

    head1_kwargs = {} if head1_kwargs is None else dict(head1_kwargs)
    head2_kwargs = {} if head2_kwargs is None else dict(head2_kwargs)

    loss_h1 = dual_head_head1_loss(outputs["head1"], y_head1, **head1_kwargs)
    if y_head2 is None:
        loss_h2 = outputs["head2"].sum() * 0.0
        info_h2 = {"n_total": int(outputs["head2"].shape[0]), "n_valid": 0}
    else:
        if "y_head1" not in head2_kwargs:
            head2_kwargs["y_head1"] = y_head1
        loss_h2, info_h2 = dual_head_head2_loss(outputs["head2"], y_head2, **head2_kwargs)

    total = float(lambda_head1) * loss_h1 + float(lambda_head2) * loss_h2
    return {
        "loss": total,
        "loss_head1": loss_h1,
        "loss_head2": loss_h2,
        "info_head2": info_h2,
    }

def weighted_cross_entropy_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-entropy loss with optional per-sample weights."""
    
    if class_weight is not None:
        class_weight = class_weight.to(device=logits.device, dtype=logits.dtype)
    loss_vec = F.cross_entropy(logits,y.view(-1).long(),weight=class_weight,label_smoothing=float(label_smoothing),reduction="none")

    if sample_weight is None:
        return loss_vec.mean()

    weights = sample_weight.view(-1).to(device=logits.device, dtype=logits.dtype)
    if int(weights.shape[0]) != int(loss_vec.shape[0]):
        raise ValueError("sample_weight must have batch size B")

    denom = weights.sum().clamp_min(float(eps))
    return (loss_vec * weights).sum() / denom

__all__ = ["dual_head_head1_loss","dual_head_head2_loss","dual_head_losses", "weighted_cross_entropy_loss"]
