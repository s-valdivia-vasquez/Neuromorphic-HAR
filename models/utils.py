#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared layers and utilities for spiking HAR models.

The LIF dynamics, surrogate-gradient spike function, and spiking
convolutional block structure follow the SNN-HAR implementation style from
Li et al. for wearable HAR with spatio-temporal spiking neural networks.

Reference code:
    Intelligent-Computing-Lab-Panda/SNN_HAR
    https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR/tree/main
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Number = Union[int, float]
TripleLike = Union[int, Sequence[int]]


class TriangularSurrogate(torch.autograd.Function):
    """Binary spike with triangular surrogate gradient."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, gamma: float) -> torch.Tensor:
        gamma_value = float(gamma)
        out = (input_tensor > 0).to(input_tensor.dtype)
        gamma_tensor = input_tensor.new_tensor(gamma_value)
        ctx.save_for_backward(input_tensor, gamma_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_tensor, gamma_tensor = ctx.saved_tensors
        gamma = float(gamma_tensor.item())
        grad = (1.0 / (gamma * gamma)) * (gamma - input_tensor.abs()).clamp(min=0.0)
        return grad_output * grad, None


class SmoothSpike(nn.Module):
    """Differentiable spike approximation based on a tanh window."""

    def __init__(self, region: float = 1.0):
        super().__init__()
        self.region = float(region)

    def forward(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        temp = float(temperature)
        clipped = torch.clamp(x, -self.region, self.region)
        denom = 2.0 * torch.tanh(x.new_tensor(self.region * temp))
        soft = torch.tanh(temp * clipped) / denom + 0.5
        hard = (x >= 0).to(x.dtype)
        return (hard - soft).detach() + soft


class LIFSpike(nn.Module):
    """Leaky integrate-and-fire spike layer for tensors shaped as (B, C, T)."""

    def __init__(
        self,
        thresh: float = 0.5,
        tau: float = 0.75,
        gamma: float = 1.0,
        smooth: bool = False,
        soft_reset: bool = True,
    ):
        super().__init__()
        self.thresh = float(thresh)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.smooth = bool(smooth)
        self.soft_reset = bool(soft_reset)
        self._smooth_spike = SmoothSpike(region=1.0) if self.smooth else None

    def _spike(self, membrane_minus_threshold: torch.Tensor) -> torch.Tensor:
        if self._smooth_spike is not None:
            return self._smooth_spike(membrane_minus_threshold, self.gamma)
        return TriangularSurrogate.apply(membrane_minus_threshold, self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"LIFSpike expects input shape (B,C,T). Received {tuple(x.shape)}")

        membrane = torch.zeros_like(x[:, :, 0])
        spikes = []
        for t in range(x.shape[2]):
            membrane = membrane * self.tau + x[:, :, t]
            spike = self._spike(membrane - self.thresh)
            membrane = membrane - spike * self.thresh if self.soft_reset else membrane * (1.0 - spike)
            spikes.append(spike)

        return torch.stack(spikes, dim=2)


def _as_triple(value: TripleLike, name: str) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"{name} must have length 3. Received {value}")
        return tuple(int(v) for v in value)

    v = int(value)
    return (v, v, v)


def _same_like_padding(kernel_sizes: Tuple[int, int, int]) -> Tuple[int, int, int]:
    paddings = []
    for kernel_size in kernel_sizes:
        paddings.append(int(kernel_size) // 2)
    return tuple(paddings)


def _validate_positive(values: Iterable[int], name: str) -> None:
    values = tuple(int(v) for v in values)
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive values. Received {values}")


def _validate_non_negative(values: Iterable[int], name: str) -> None:
    values = tuple(int(v) for v in values)
    if any(v < 0 for v in values):
        raise ValueError(f"{name} cannot contain negative values. Received {values}")


def _to_channels_first_time(x: torch.Tensor, valid_channels: Sequence[int]) -> torch.Tensor:
    """Normalize a 3D tensor to (B, C, T)."""
    valid = {int(c) for c in valid_channels}
    if x.dim() != 3:
        raise ValueError(f"Expected a 3D tensor. Received {tuple(x.shape)}")

    if int(x.shape[1]) in valid:
        return x.contiguous()
    if int(x.shape[2]) in valid:
        return x.permute(0, 2, 1).contiguous()

    raise ValueError(
        f"Could not identify channel dimension. Received shape={tuple(x.shape)}, "
        f"valid channel counts={sorted(valid)}"
    )


class SpikingLinearBlock(nn.Module):
    """Linear block applied at every time step, followed by LIF dynamics."""

    def __init__(
        self,
        in_features: Optional[int],
        out_features: int,
        p_drop: float = 0.0,
        bias: bool = True,
        use_bn: bool = True,
        **lif_kwargs,
    ):
        super().__init__()
        if in_features is None:
            self.fc = nn.LazyLinear(out_features, bias=bias)
        else:
            self.fc = nn.Linear(in_features, out_features, bias=bias)

        self.bn = nn.BatchNorm1d(out_features) if bool(use_bn) else nn.Identity()
        self.lif = LIFSpike(**lif_kwargs)
        self.drop = nn.Dropout(float(p_drop)) if float(p_drop) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SpikingLinearBlock expects (B,C,T). Received {tuple(x.shape)}")

        batch, channels, steps = x.shape
        z = x.permute(0, 2, 1).contiguous().view(batch * steps, channels)
        z = self.fc(z)
        z = self.bn(z)
        z = z.view(batch, steps, -1).permute(0, 2, 1).contiguous()
        return self.drop(self.lif(z))


class ConvLIFMaxPoolBlock(nn.Module):
    """Conv1d -> optional BN -> LIF -> MaxPool1d -> optional Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_kernel_size: int,
        pool_stride: int,
        pool_padding: int,
        p_drop: float = 0.0,
        bias: bool = False,
        use_bn: bool = True,
        **lif_kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            bias=bool(bias),
        )
        self.bn = nn.BatchNorm1d(int(out_channels)) if bool(use_bn) else nn.Identity()
        self.lif = LIFSpike(**lif_kwargs)
        self.pool = nn.MaxPool1d(
            kernel_size=int(pool_kernel_size),
            stride=int(pool_stride),
            padding=int(pool_padding),
        )
        self.drop = nn.Dropout(float(p_drop)) if float(p_drop) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif(x)
        x = self.pool(x)
        return self.drop(x)


class EventInputMixin:
    """Input validation and polarity handling for event-based IMU tensors."""

    n_channels: int
    merge_polarities: bool
    event_scale: float
    len_sw: Optional[int]
    time_steps: Optional[int]

    def _event_to_current(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")

        if x.dim() == 4:
            current = self._polarity_event_to_current(x)
        elif x.dim() == 3:
            possible_channels = (self.n_channels, 2 * self.n_channels)
            current = _to_channels_first_time(x, possible_channels)
        else:
            raise ValueError(
                "Unsupported event input format. Expected (B,T,C,2), (B,C,T,2), "
                f"(B,T,C), or (B,C,T). Received {tuple(x.shape)}"
            )

        if self.len_sw is not None and int(current.shape[2]) != int(self.len_sw):
            raise ValueError(
                f"Input temporal length mismatch: model expects len_sw={self.len_sw}, "
                f"but received T={int(current.shape[2])}"
            )

        current = current.float() * float(self.event_scale)
        return self._temporal_resample(current)

    def _polarity_event_to_current(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 2:
            raise ValueError(f"4D event input must have polarity size 2 in the last axis. Received {tuple(x.shape)}")

        if int(x.shape[2]) == self.n_channels:
            pos = x[..., 0]
            neg = x[..., 1]
            if self.merge_polarities:
                return (pos - neg).permute(0, 2, 1).contiguous()
            return torch.cat([pos, neg], dim=2).permute(0, 2, 1).contiguous()

        if int(x.shape[1]) == self.n_channels:
            pos = x[..., 0]
            neg = x[..., 1]
            if self.merge_polarities:
                return (pos - neg).contiguous()
            return torch.cat([pos, neg], dim=1).contiguous()

        raise ValueError(
            f"Could not identify IMU channel dimension. Received shape={tuple(x.shape)}, "
            f"n_channels={self.n_channels}"
        )

    def _temporal_resample(self, x_bct: torch.Tensor) -> torch.Tensor:
        if self.time_steps is None:
            return x_bct

        target = int(self.time_steps)
        if target <= 0:
            raise ValueError("time_steps must be positive or None")

        batch, channels, steps = x_bct.shape
        if steps == target:
            return x_bct

        if steps > target:
            if steps % target == 0:
                return x_bct.view(batch, channels, target, steps // target).sum(dim=3)
            return F.adaptive_avg_pool1d(x_bct, target) * (float(steps) / float(target))

        return F.interpolate(x_bct, size=target, mode="nearest")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for parameter in model.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        total += parameter.numel()
    return int(total)


__all__ = [
    "Number",
    "TripleLike",
    "TriangularSurrogate",
    "SmoothSpike",
    "LIFSpike",
    "SpikingLinearBlock",
    "ConvLIFMaxPoolBlock",
    "EventInputMixin",
    "count_parameters",
]
