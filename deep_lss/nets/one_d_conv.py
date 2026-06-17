# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

import torch
from torch import nn
import torch.nn.functional as F
from deep_lss.nets.deepsphere_torch import healpy_layers
from deep_lss.nets.regression_head import get_regression_head
from deep_lss.nets.torch_utils import ChannelsLastConv1d, get_activation


class OneDResidualBlock(nn.Module):
    """Residual block for channels-last (B, L, C) tensors using PyTorch Conv1d internally."""

    def __init__(self, filters, kernel_size, norm_kwargs={}, norm_type="layer_norm", activation="relu", name=""):
        super().__init__()
        self.activation = get_activation(activation)
        self.conv1 = ChannelsLastConv1d(filters, kernel_size, padding="same", activation=None)
        self.conv2 = ChannelsLastConv1d(filters, kernel_size, padding="same", activation=None)
        if norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(filters, **{k:v for k,v in norm_kwargs.items() if k != "axis"})
            self.norm2 = nn.LayerNorm(filters, **{k:v for k,v in norm_kwargs.items() if k != "axis"})
        elif norm_type == "batch_norm":
            self.norm1 = nn.LazyBatchNorm1d(); self.norm2 = nn.LazyBatchNorm1d()
        else:
            raise NotImplementedError

    def forward(self, input_tensor, training: bool | None = None):
        x = self.activation(self.norm1(self.conv1(input_tensor)))
        x = self.activation(self.norm2(self.conv2(x)))
        return self.activation(x + input_tensor)


class OneDConvLayers:
    def __init__(self, out_features, base_channels=8, downsampling_layers=5, residual_layers=5, kernel_size=9,
                 second_to_last_features=128, dropout_rate=None, activation="relu", smoothing_kwargs=None) -> None:
        self.layers = []
        if smoothing_kwargs is not None:
            self.layers.append(healpy_layers.HealpySmoothing(**smoothing_kwargs))
        n_channels = base_channels
        for i in range(downsampling_layers):
            self.layers.append(ChannelsLastConv1d(n_channels, kernel_size, padding="same", activation=activation))
            self.layers.append(nn.LayerNorm(n_channels))
            n_channels *= 2
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation, name=f"pseudo_conv_{i}"))
        for i in range(residual_layers):
            self.layers.append(OneDResidualBlock(filters=n_channels, kernel_size=kernel_size, activation=activation, name=f"residual_block_{i}"))
        self.layers.extend(get_regression_head(out_features=out_features, head_type="dense",
                                               second_to_last_features=second_to_last_features,
                                               activation=activation, dropout_rate=dropout_rate))

    def get_layers(self):
        return self.layers
