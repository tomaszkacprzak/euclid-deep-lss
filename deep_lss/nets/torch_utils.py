"""Small PyTorch layer helpers used by network builders."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def get_activation(activation):
    if activation is None:
        return lambda x: x
    if callable(activation):
        return activation
    name = str(activation).lower()
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name in {"gelu", "swish", "silu"}:
        return getattr(F, "silu" if name == "swish" else name)
    raise ValueError(f"Unsupported activation: {activation}")


class Activation(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(x)


class LinearActivation(nn.Module):
    def __init__(self, out_features: int, activation="relu", in_features: int | None = None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features) if in_features is not None else nn.LazyLinear(out_features)
        self.activation = get_activation(activation)
        self.out_features = out_features

    def forward(self, x):
        return self.activation(self.linear(x))


class ChannelsLastConv1d(nn.Module):
    """Conv1d wrapper accepting and returning Keras-style (B, L, C) tensors."""

    def __init__(self, out_channels, kernel_size, stride=1, padding="same", activation=None, in_channels=None, **kwargs):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2
        conv_cls = nn.Conv1d if in_channels is not None else nn.LazyConv1d
        conv_kwargs = dict(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, **kwargs)
        if in_channels is not None:
            conv_kwargs["in_channels"] = in_channels
        self.conv = conv_cls(**conv_kwargs)
        self.activation = get_activation(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.activation(x)


class LazyLayerNorm(nn.Module):
    """LayerNorm that infers the normalized shape from a chosen axis on first use."""

    def __init__(self, axis=-1, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.norm = None

    def forward(self, x):
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        normalized_shape = tuple(x.shape[axis:])
        if self.norm is None or tuple(self.norm.normalized_shape) != normalized_shape:
            self.norm = nn.LayerNorm(normalized_shape, eps=self.eps, elementwise_affine=self.elementwise_affine).to(
                device=x.device, dtype=x.dtype
            )
        return self.norm(x)


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class MeanLayer(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        return torch.mean(inputs, dim=self.axis)


class CallTrainingSequential(nn.Sequential):
    def forward(self, input, training: bool | None = None):
        for module in self:
            try:
                input = module(input, training=training)
            except TypeError:
                input = module(input)
        return input
