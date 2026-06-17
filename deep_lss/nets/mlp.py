# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

import numpy as np
import torch
from torch import nn
import logging
try:
    from msfm.utils import logger
    LOGGER = logger.get_logger(__file__)
except ModuleNotFoundError:
    LOGGER = logging.getLogger(__file__)
from deep_lss.nets.torch_utils import LazyLayerNorm, LinearActivation


class PCAWhiteningLayer(nn.Module):
    """Offline PCA whitening stored as non-trainable PyTorch buffers."""

    def __init__(self, n_components, whiten=True, eps=1e-8, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.eps = eps
        self.register_buffer("mean_", torch.empty(0), persistent=True)
        self.register_buffer("components_", torch.empty(0, 0), persistent=True)

    def build(self, input_shape):
        n_in = input_shape[-1]
        n_out = min(self.n_components, n_in)
        self.mean_ = torch.zeros(n_in)
        self.components_ = torch.zeros(n_in, n_out)

    @property
    def built(self):
        return self.mean_.numel() > 0

    def fit(self, x, max_samples=200_000):
        if not self.built:
            self.build((None, x.shape[-1]))
        rng = np.random.default_rng(0)
        if x.shape[0] > max_samples:
            x = x[rng.choice(x.shape[0], size=max_samples, replace=False)]
        x = x.astype(np.float64)
        mean = x.mean(axis=0)
        cov = np.cov(x.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][: self.n_components]
        components = eigvecs[:, idx] / np.sqrt(eigvals[idx] + self.eps) if self.whiten else eigvecs[:, idx]
        explained = eigvals[np.argsort(eigvals)[::-1]][: self.n_components].sum() / eigvals.sum()
        LOGGER.info(f"PCAWhiteningLayer: kept {self.n_components}/{x.shape[1]} components, explained variance = {explained:.3f}")
        self.mean_ = torch.as_tensor(mean.astype(np.float32))
        self.components_ = torch.as_tensor(components.astype(np.float32))

    def forward(self, inputs):
        if not self.built:
            raise RuntimeError("PCAWhiteningLayer.fit() or build() must be called before forward().")
        return (inputs - self.mean_.to(inputs.device, inputs.dtype)) @ self.components_.to(inputs.device, inputs.dtype)

    def get_config(self):
        return {"n_components": self.n_components, "whiten": self.whiten, "eps": self.eps}


class MultiLayerPerceptron(nn.Module):
    def __init__(self, output_size, num_hidden_units, num_layers, num_penultimate=None, dropout_rate=0.0,
                 normalization="layer", activation="relu", whitening=None):
        super().__init__()
        self.whitening = whitening
        skip_norm = whitening is not None and whitening.whiten
        if skip_norm:
            self.norm_layer = None
        elif normalization == "layer":
            self.norm_layer = LazyLayerNorm(axis=-1)
        elif normalization == "batch":
            self.norm_layer = nn.LazyBatchNorm1d()
        else:
            raise ValueError(f"Unknown normalization type: {normalization}")
        layers = []
        for _ in range(num_layers):
            layers.append(LinearActivation(num_hidden_units, activation=activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        if num_penultimate is not None:
            LOGGER.info("Including a penultimate layer in the MLP")
            layers.append(nn.LazyLinear(num_penultimate))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.LazyLinear(output_size)

    def forward(self, inputs, training: bool | None = None):
        x = self.whitening(inputs) if self.whitening is not None else inputs
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
