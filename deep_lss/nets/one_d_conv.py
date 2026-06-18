"""PyTorch 1D-convolutional network builder."""

from __future__ import annotations

import torch
from torch import nn

from deep_lss.nets import deepsphere_torch as dst
from deep_lss.nets.regression_head import get_regression_head


class OneDResidualBlock(nn.Module):
    def __init__(self, filters, kernel_size, norm_kwargs=None, norm_type="layer_norm", activation="relu", name=""):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        self.activation = dst.torch_activation(activation) or (lambda x: x)
        self.conv1 = nn.Conv1d(filters, filters, kernel_size, stride=1, padding="same")
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, stride=1, padding="same")
        if norm_type == "layer_norm":
            self.norm1 = dst.LazyLayerNorm(**norm_kwargs)
            self.norm2 = dst.LazyLayerNorm(**norm_kwargs)
        elif norm_type == "batch_norm":
            self.norm1 = nn.BatchNorm1d(filters, **norm_kwargs)
            self.norm2 = nn.BatchNorm1d(filters, **norm_kwargs)
        else:
            raise NotImplementedError

    def _norm(self, norm, x):
        if isinstance(norm, nn.BatchNorm1d):
            return torch.transpose(norm(torch.transpose(x, 1, 2)), 1, 2)
        return norm(x)

    def forward(self, input_tensor):
        x = torch.transpose(input_tensor, 1, 2)
        x = torch.transpose(self.conv1(x), 1, 2)
        x = self.activation(self._norm(self.norm1, x))
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(self.conv2(x), 1, 2)
        x = self.activation(self._norm(self.norm2, x))
        return self.activation(x + input_tensor)


class Conv1dNodes(nn.Module):
    def __init__(self, out_channels, kernel_size, activation="relu"):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = dst.torch_activation(activation)
        self.conv = None

    def forward(self, x):
        if self.conv is None:
            self.conv = nn.Conv1d(x.shape[-1], self.out_channels, self.kernel_size, padding="same").to(
                x.device, x.dtype
            )
        y = torch.transpose(self.conv(torch.transpose(x, 1, 2)), 1, 2)
        return self.activation(y) if self.activation is not None else y


class OneDConvLayers:
    def __init__(
        self,
        out_features,
        base_channels=8,
        downsampling_layers=5,
        residual_layers=5,
        kernel_size=9,
        second_to_last_features=128,
        dropout_rate=None,
        activation="relu",
        smoothing_kwargs=None,
    ) -> None:
        self.layers = []
        if smoothing_kwargs is not None:
            self.layers.append(dst.healpy_smoothing(**smoothing_kwargs))
        n_channels = base_channels
        for _ in range(downsampling_layers):
            self.layers.append(Conv1dNodes(n_channels, kernel_size, activation=activation))
            self.layers.append(dst.LazyLayerNorm())
            n_channels *= 2
            self.layers.append(dst.healpy_pseudo_conv(p=1, Fout=n_channels, activation=activation))
        for _ in range(residual_layers):
            self.layers.append(OneDResidualBlock(filters=n_channels, kernel_size=kernel_size, activation=activation))
        self.layers.extend(
            get_regression_head(
                out_features=out_features,
                head_type="dense",
                second_to_last_features=second_to_last_features,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        )

    def get_layers(self):
        return self.layers
