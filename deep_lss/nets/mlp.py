"""PyTorch MLP components."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import logging

try:
    from msfm.utils import logger
except ImportError:
    logger = None
from deep_lss.nets import deepsphere_torch as dst

LOGGER = logger.get_logger(__file__) if logger is not None else logging.getLogger(__name__)


class PCAWhiteningLayer(nn.Module):
    def __init__(self, n_components, whiten=True, eps=1e-8, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.eps = eps
        self.register_buffer("mean_", torch.empty(0))
        self.register_buffer("components_", torch.empty(0))

    def build(self, input_shape):
        n_in = int(input_shape[-1])
        n_out = min(self.n_components, n_in)
        self.mean_ = torch.zeros(n_in)
        self.components_ = torch.zeros(n_in, n_out)
        return self

    def fit(self, x, max_samples=200_000):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if self.mean_.numel() == 0:
            self.build((None, x.shape[-1]))
        rng = np.random.default_rng(0)
        if x.shape[0] > max_samples:
            x = x[rng.choice(x.shape[0], size=max_samples, replace=False)]
        x = x.astype(np.float64)
        mean = x.mean(axis=0)
        cov = np.cov(x.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        idx = order[: self.n_components]
        components = eigvecs[:, idx] / np.sqrt(eigvals[idx] + self.eps) if self.whiten else eigvecs[:, idx]
        explained = eigvals[order][: self.n_components].sum() / eigvals.sum()
        LOGGER.info(
            f"PCAWhiteningLayer: kept {self.n_components}/{x.shape[1]} components, explained variance = {explained:.3f}"
        )
        device = self.mean_.device
        self.mean_ = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.components_ = torch.as_tensor(components, dtype=torch.float32, device=device)

    def forward(self, inputs):
        if self.mean_.numel() == 0:
            self.build((None, inputs.shape[-1]))
            self.mean_ = self.mean_.to(inputs.device, inputs.dtype)
            self.components_ = self.components_.to(inputs.device, inputs.dtype)
        return (inputs - self.mean_.to(inputs.device, inputs.dtype)) @ self.components_.to(inputs.device, inputs.dtype)


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        output_size,
        num_hidden_units,
        num_layers,
        num_penultimate=None,
        dropout_rate=0.0,
        normalization="layer",
        activation="relu",
        whitening=None,
    ):
        super().__init__()
        self.whitening = whitening
        skip_norm = whitening is not None and whitening.whiten
        if skip_norm:
            self.norm_layer = None
        elif normalization == "layer":
            self.norm_layer = dst.LazyLayerNorm()
        elif normalization == "batch":
            self.norm_layer = nn.LazyBatchNorm1d()
        else:
            raise ValueError(f"Unknown normalization type: {normalization}")
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LazyLinear(num_hidden_units))
            act = dst.torch_activation(activation)
            if act is not None:
                layers.append(dst.LambdaLayer(act))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        if num_penultimate is not None:
            LOGGER.info("Including a penultimate layer in the MLP")
            layers.append(nn.LazyLinear(num_penultimate))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.LazyLinear(output_size)

    def forward(self, inputs, training=None):
        if training is not None:
            self.train(bool(training))
        x = self.whitening(inputs) if self.whitening is not None else inputs
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
