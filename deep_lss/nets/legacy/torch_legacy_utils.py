"""PyTorch compatibility helpers for legacy network layer lists."""

from __future__ import annotations

import torch
from torch import nn

from deep_lss.nets.torch_utils import ChannelsLastConv1d, Flatten, LazyLayerNorm, get_activation


class AxisLayerNorm(LazyLayerNorm):
    """LayerNorm with Keras-style axis support and lazy shape inference."""


class Dense(nn.Module):
    def __init__(self, out_features, activation=None):
        super().__init__()
        self.linear = nn.LazyLinear(out_features)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.linear(x))


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class MixingDense(nn.Module):
    """PyTorch stand-in for the legacy Kids1000 disconnected-component mixing layer."""

    def __init__(self, activation="relu"):
        super().__init__()
        self.activation = get_activation(activation)
        self.linear = nn.LazyLinear(1)
        self._out_features = None

    def forward(self, x):
        # Lazily preserve the incoming feature width on first use.
        if self._out_features is None:
            self._out_features = x.shape[-1]
            self.linear = nn.LazyLinear(self._out_features).to(device=x.device, dtype=x.dtype)
        return self.activation(self.linear(x))


__all__ = ["AxisLayerNorm", "ChannelsLastConv1d", "Dense", "Flatten", "Lambda", "MixingDense"]
