#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spiking neural network models for event-based IMU HAR.

The SCN backbone follows the SNN-HAR implementation style from
Li et al. for wearable HAR with spatio-temporal spiking neural networks.

Reference code:
    Intelligent-Computing-Lab-Panda/SNN_HAR
    https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR/tree/main
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from models.utils import (
    ConvLIFMaxPoolBlock,
    EventInputMixin,
    SpikingLinearBlock,
    TripleLike,
    _as_triple,
    _same_like_padding,
    _to_channels_first_time,
    _validate_non_negative,
    _validate_positive,
)


class DualHeadSCN(nn.Module, EventInputMixin):
    """Dual-head spiking convolutional network for event-based IMU windows."""

    def __init__(
        self,
        n_channels: int = 6,
        n_classes_head1: int = 3,
        conv_channels: TripleLike = (32, 64, 64),
        conv_kernel_sizes: TripleLike = (32, 32, 8),
        conv_strides: TripleLike = (4, 2, 1),
        conv_paddings: Optional[TripleLike] = None,
        pool_kernel_sizes: TripleLike = (2, 2, 2),
        pool_strides: TripleLike = (2, 2, 2),
        pool_paddings: TripleLike = (0, 0, 0),
        backbone: bool = False,
        merge_polarities: bool = False,
        event_scale: float = 1.0,
        len_sw: Optional[int] = None,
        time_steps: Optional[int] = None,
        p_drop: float = 0.35,
        bias: bool = False,
        use_bn: bool = True,
        offset_dim: Optional[int] = None,
        offset_hidden: int = 8,
        offset_scale: float = 1.0,
        offset_drop: float = 0.0,
        spiking_heads: bool = False,
        head_rate_scale: float = 8.0,
        tau: float = 0.75,
        thresh: float = 0.5,
        gamma: float = 1.0,
        smooth: bool = False,
        soft_reset: bool = True,
    ):
        super().__init__()

        self.backbone = bool(backbone)
        self.n_channels = int(n_channels)
        self.n_classes_head1 = int(n_classes_head1)
        self.merge_polarities = bool(merge_polarities)
        self.event_scale = float(event_scale)
        self.len_sw = None if len_sw is None else int(len_sw)
        self.time_steps = None if time_steps is None else int(time_steps)
        self.use_bn = bool(use_bn)
        self.spiking_heads = bool(spiking_heads)
        self.head_rate_scale = float(head_rate_scale)

        self.offset_dim = int(self.n_channels if offset_dim is None else offset_dim)
        self.offset_hidden = int(offset_hidden)
        self.offset_scale = float(offset_scale)
        if self.offset_hidden <= 0:
            raise ValueError("offset_hidden must be > 0")

        self._set_convolution_config(
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            conv_paddings=conv_paddings,
            pool_kernel_sizes=pool_kernel_sizes,
            pool_strides=pool_strides,
            pool_paddings=pool_paddings,
        )

        lif_kwargs = dict(
            tau=float(tau),
            thresh=float(thresh),
            gamma=float(gamma),
            smooth=bool(smooth),
            soft_reset=bool(soft_reset),
        )
        self._build_backbone(float(p_drop), bool(bias), lif_kwargs)
        self._build_offset_branch(float(offset_drop), lif_kwargs)
        self._build_heads(lif_kwargs)

    def _set_convolution_config(
        self,
        conv_channels: TripleLike,
        conv_kernel_sizes: TripleLike,
        conv_strides: TripleLike,
        conv_paddings: Optional[TripleLike],
        pool_kernel_sizes: TripleLike,
        pool_strides: TripleLike,
        pool_paddings: TripleLike,
    ) -> None:
        self.conv_channels = _as_triple(conv_channels, "conv_channels")
        self.conv_kernel_sizes = _as_triple(conv_kernel_sizes, "conv_kernel_sizes")
        self.conv_strides = _as_triple(conv_strides, "conv_strides")
        self.conv_paddings = _same_like_padding(self.conv_kernel_sizes) if conv_paddings is None else _as_triple(conv_paddings, "conv_paddings")
        self.pool_kernel_sizes = _as_triple(pool_kernel_sizes, "pool_kernel_sizes")
        self.pool_strides = _as_triple(pool_strides, "pool_strides")
        self.pool_paddings = _as_triple(pool_paddings, "pool_paddings")

        _validate_positive(self.conv_channels, "conv_channels")
        _validate_positive(self.conv_kernel_sizes, "conv_kernel_sizes")
        _validate_positive(self.conv_strides, "conv_strides")
        _validate_positive(self.pool_kernel_sizes, "pool_kernel_sizes")
        _validate_positive(self.pool_strides, "pool_strides")
        _validate_non_negative(self.conv_paddings, "conv_paddings")
        _validate_non_negative(self.pool_paddings, "pool_paddings")

    def _build_backbone(self, p_drop: float, bias: bool, lif_kwargs: Dict[str, Any]) -> None:
        input_channels = self.n_channels if self.merge_polarities else (2 * self.n_channels)
        c1, c2, c3 = self.conv_channels
        self.out_dim = int(c3)

        self.conv_block1 = self._make_conv_block(input_channels, c1, 0, p_drop, bias, lif_kwargs)
        self.conv_block2 = self._make_conv_block(c1, c2, 1, 0.0, bias, lif_kwargs)
        self.conv_block3 = self._make_conv_block(c2, c3, 2, 0.0, bias, lif_kwargs)

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        index: int,
        p_drop: float,
        bias: bool,
        lif_kwargs: Dict[str, Any],
    ) -> ConvLIFMaxPoolBlock:
        return ConvLIFMaxPoolBlock(
            in_channels,
            out_channels,
            kernel_size=self.conv_kernel_sizes[index],
            stride=self.conv_strides[index],
            padding=self.conv_paddings[index],
            pool_kernel_size=self.pool_kernel_sizes[index],
            pool_stride=self.pool_strides[index],
            pool_padding=self.pool_paddings[index],
            p_drop=float(p_drop),
            bias=bool(bias),
            use_bn=self.use_bn,
            **lif_kwargs,
        )

    def _build_offset_branch(self, offset_drop: float, lif_kwargs: Dict[str, Any]) -> None:
        self.offset_proj_spk = SpikingLinearBlock(
            in_features=self.offset_dim,
            out_features=self.offset_hidden,
            p_drop=float(offset_drop),
            bias=True,
            use_bn=self.use_bn,
            **lif_kwargs,
        )
        self.fused_dim = int(self.out_dim + self.offset_hidden)

    def _build_heads(self, lif_kwargs: Dict[str, Any]) -> None:
        if self.backbone:
            return

        if self.spiking_heads:
            self.head1_spk = SpikingLinearBlock(
                in_features=self.fused_dim,
                out_features=self.n_classes_head1,
                p_drop=0.0,
                bias=True,
                use_bn=self.use_bn,
                **lif_kwargs,
            )
            self.head2_spk = SpikingLinearBlock(
                in_features=self.fused_dim,
                out_features=1,
                p_drop=0.0,
                bias=True,
                use_bn=self.use_bn,
                **lif_kwargs,
            )
            return

        self.head1 = nn.Linear(self.fused_dim, self.n_classes_head1, bias=True)
        self.head2 = nn.Linear(self.fused_dim, 1, bias=True)

    def _forward_event_spikes(self, x: torch.Tensor) -> torch.Tensor:
        h = self._event_to_current(x)
        h = self.conv_block1(h)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        return h

    def _check_offset(
        self,
        offset: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if offset is None:
            return torch.zeros(batch_size, self.offset_dim, device=device, dtype=dtype)

        if not torch.is_tensor(offset):
            offset = torch.as_tensor(offset, device=device, dtype=dtype)
        else:
            offset = offset.to(device=device, dtype=dtype)

        if offset.dim() == 1:
            offset = offset.view(1, -1)
        elif offset.dim() == 3:
            offset = self._squeeze_offset(offset)
        elif offset.dim() != 2:
            raise ValueError(f"offset must be 1D, 2D, or 3D. Received {tuple(offset.shape)}")

        if int(offset.shape[1]) != self.offset_dim:
            raise ValueError(f"offset_dim mismatch: expected {self.offset_dim}, received {int(offset.shape[1])}")
        if int(offset.shape[0]) != batch_size:
            raise ValueError(f"offset batch mismatch: x batch={batch_size}, offset batch={int(offset.shape[0])}")

        return offset * self.offset_scale

    def _squeeze_offset(self, offset: torch.Tensor) -> torch.Tensor:
        if offset.shape[1] == 1 and offset.shape[2] == self.offset_dim:
            return offset[:, 0, :]
        if offset.shape[1] == self.offset_dim and offset.shape[2] == 1:
            return offset[:, :, 0]

        raise ValueError(
            "Unsupported 3D offset. Expected (B,1,C) or (B,C,1). "
            f"Received {tuple(offset.shape)}"
        )

    def _offset_as_constant_current(
        self,
        offset: Optional[torch.Tensor],
        batch_size: int,
        time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        off = self._check_offset(offset, batch_size=batch_size, device=device, dtype=dtype)
        return off.unsqueeze(2).expand(-1, -1, int(time_steps)).contiguous()

    def _dense_heads_forward(self, rep_fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "head1": self.head1(rep_fused),
            "head2": self.head2(rep_fused).squeeze(1),
        }

    def _spiking_heads_forward(self, h_fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        h1_spk = self.head1_spk(h_fused)
        h2_spk = self.head2_spk(h_fused)
        rate_h1 = h1_spk.mean(dim=2)
        rate_h2 = h2_spk.mean(dim=2).squeeze(1)
        return {
            "head1": self.head_rate_scale * rate_h1,
            "head2": self.head_rate_scale * (rate_h2 - 0.5),
        }

    def _fused_representation(
        self,
        x: torch.Tensor,
        offset: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        h_evt = self._forward_event_spikes(x)
        rep_evt = h_evt.mean(dim=2)

        offset_current = self._offset_as_constant_current(
            offset=offset,
            batch_size=int(h_evt.shape[0]),
            time_steps=int(h_evt.shape[2]),
            device=h_evt.device,
            dtype=h_evt.dtype,
        )
        h_off = self.offset_proj_spk(offset_current)
        rep_off = h_off.mean(dim=2)

        if self.spiking_heads:
            h_fused = torch.cat([h_evt, h_off], dim=1)
            return h_fused, h_fused.mean(dim=2)

        rep_fused = torch.cat([rep_evt, rep_off], dim=1)
        return None, rep_fused

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):
        h_fused, rep_fused = self._fused_representation(x, offset)
        if self.backbone:
            return None, rep_fused

        outputs = self._spiking_heads_forward(h_fused) if self.spiking_heads else self._dense_heads_forward(rep_fused)
        return outputs, rep_fused


class SingleHeadSCN(DualHeadSCN):
    """Single-head SCN variant for datasets with one activity classifier."""

    def __init__(
        self,
        *args,
        n_classes: int = 6,
        **kwargs,
    ):
        self.n_classes = int(n_classes)
        kwargs["n_classes_head1"] = self.n_classes
        super().__init__(*args, **kwargs)

        if hasattr(self, "head2"):
            del self.head2
        if hasattr(self, "head2_spk"):
            del self.head2_spk

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):
        h_fused, rep_fused = self._fused_representation(x, offset)
        if self.backbone:
            return None, rep_fused

        if self.spiking_heads:
            h1_spk = self.head1_spk(h_fused)
            logits = self.head_rate_scale * h1_spk.mean(dim=2)
        else:
            logits = self.head1(rep_fused)
        return logits, rep_fused


class FullSNN(nn.Module, EventInputMixin): #Not implemented in main paper
    """Fully spiking MLP baseline for dense IMU or event tensors."""

    def __init__(
        self,
        n_channels: int = 6,
        n_classes: int = 6,
        len_sw: Optional[int] = None,
        hidden_sizes: Sequence[int] = (2048, 2048, 1024, 1024, 512, 512),
        time_steps: Optional[int] = 32,
        backbone: bool = False,
        merge_polarities: bool = True,
        event_scale: float = 1.0,
        p_drop: float = 0.1,
        bias: bool = True,
        use_bn: bool = True,
        tau: float = 0.75,
        thresh: float = 0.5,
        gamma: float = 1.0,
        smooth: bool = False,
        soft_reset: bool = True,
    ):
        super().__init__()
        self.backbone = bool(backbone)
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.len_sw = None if len_sw is None else int(len_sw)
        self.time_steps = None if time_steps is None else int(time_steps)
        self.merge_polarities = bool(merge_polarities)
        self.event_scale = float(event_scale)

        hidden_sizes = tuple(int(h) for h in hidden_sizes)
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer")
        self.out_dim = int(hidden_sizes[-1])

        lif_kwargs = dict(
            tau=float(tau),
            thresh=float(thresh),
            gamma=float(gamma),
            smooth=bool(smooth),
            soft_reset=bool(soft_reset),
        )
        self.layers = self._make_layers(hidden_sizes, p_drop, bias, use_bn, lif_kwargs)

        if not self.backbone:
            self.classifier = SpikingLinearBlock(
                in_features=hidden_sizes[-1],
                out_features=self.n_classes,
                p_drop=0.0,
                bias=bool(bias),
                use_bn=bool(use_bn),
                **lif_kwargs,
            )

    def _make_layers(
        self,
        hidden_sizes: Sequence[int],
        p_drop: float,
        bias: bool,
        use_bn: bool,
        lif_kwargs: Dict[str, Any],
    ) -> nn.ModuleList:
        in_features = None if self.len_sw is None else self.n_channels * self.len_sw
        layers = []
        prev = in_features

        for i, hidden in enumerate(hidden_sizes):
            drop = float(p_drop) if i < len(hidden_sizes) - 1 else min(float(p_drop), 0.05)
            layers.append(
                SpikingLinearBlock(
                    in_features=prev,
                    out_features=hidden,
                    p_drop=drop,
                    bias=bool(bias),
                    use_bn=bool(use_bn),
                    **lif_kwargs,
                )
            )
            prev = hidden

        return nn.ModuleList(layers)

    def _dense_to_current(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
        if x.dim() != 3:
            raise ValueError(f"Dense input must be (B,T,C), (B,C,T), or (B,T,1,C). Received {tuple(x.shape)}")

        x_bct = _to_channels_first_time(x, (self.n_channels,))
        if self.len_sw is not None and int(x_bct.shape[2]) != int(self.len_sw):
            raise ValueError(
                f"Input temporal length mismatch: model expects len_sw={self.len_sw}, "
                f"but received T={int(x_bct.shape[2])}"
            )

        flat = x_bct.contiguous().view(x_bct.shape[0], -1)
        steps = 1 if self.time_steps is None else int(self.time_steps)
        if steps <= 0:
            raise ValueError("time_steps must be positive or None")
        return flat.unsqueeze(2).expand(-1, -1, steps).contiguous()

    def _input_to_current(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 2:
            return self._event_to_current(x)
        return self._dense_to_current(x.float())

    def forward(self, x: torch.Tensor):
        h = self._input_to_current(x)
        for layer in self.layers:
            h = layer(h)

        rep = h.mean(dim=2)
        if self.backbone:
            return None, rep

        out_spk = self.classifier(h)
        logits = out_spk.mean(dim=2)
        return logits, rep


__all__ = ["DualHeadSCN", "SingleHeadSCN", "FullSNN"]
