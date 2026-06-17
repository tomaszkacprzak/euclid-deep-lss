# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

import numpy as np
import torch
from torch import nn

from deep_lss.nets.deepsphere_torch import MapGCNN
from deep_lss.nets.torch_utils import LazyLayerNorm
import logging
try:
    from msfm.utils import logger
    LOGGER = logger.get_logger(__file__)
except ModuleNotFoundError:
    LOGGER = logging.getLogger(__file__)


class ClsBinningAndTransformLayer(nn.Module):
    """Non-trainable PyTorch layer that bins raw per-ell Cls with per-pair scale cuts."""

    def __init__(self, n_ell, n_bins, l_min_per_pair, l_max_per_pair, **kwargs):
        super().__init__()
        try:
            from msfm.utils.power_spectra import get_cl_bins
        except ModuleNotFoundError:
            get_cl_bins = lambda lmin, lmax, n: np.linspace(lmin, lmax, n)
        n_z_cross = len(l_min_per_pair)
        assert len(l_max_per_pair) == n_z_cross
        ells = np.arange(n_ell, dtype=np.float64)
        W = np.zeros((n_ell, n_bins, n_z_cross), dtype=np.float32)
        for c, (lmin_c, lmax_c) in enumerate(zip(l_min_per_pair, l_max_per_pair)):
            bin_edges_c = get_cl_bins(lmin_c, lmax_c, n_bins + 1)
            for k in range(n_bins):
                in_bin = (ells >= bin_edges_c[k]) & (ells < bin_edges_c[k + 1])
                if in_bin.sum() > 0:
                    W[in_bin, k, c] = 1.0 / in_bin.sum()
        self.register_buffer("bin_weight", torch.as_tensor(W, dtype=torch.float32), persistent=True)
        self.n_cls_flat = n_bins * n_z_cross
        LOGGER.warning(f"ClsBinningAndTransformLayer: n_bins={n_bins}, n_z_cross={n_z_cross}, output_dim={self.n_cls_flat}")

    def forward(self, cls, training: bool | None = None):
        cls_binned = torch.einsum("blc,lkc->bkc", cls.float(), self.bin_weight.to(cls.device))
        cls_flat = cls_binned.reshape(cls_binned.shape[0], -1)
        return torch.sign(cls_flat) * torch.log(torch.abs(cls_flat) + 1e-10)


class MapsPlusCLSNetwork(nn.Module):
    """PyTorch maps + Cls combined network."""

    def __init__(self, conv_layers, cls_embedding_layers, regression_head_layers, n_side, tfr_n_side, indices,
                 n_neighbors, max_batch_size, initial_Fin, n_cls_bins, l_min_per_pair, l_max_per_pair):
        super().__init__()
        self.gcnn = MapGCNN(nside=n_side, indices=indices, layers=conv_layers, n_neighbors=n_neighbors,
                            max_batch_size=max_batch_size, initial_Fin=initial_Fin)
        self.cls_layer = ClsBinningAndTransformLayer(3 * tfr_n_side, n_cls_bins, l_min_per_pair, l_max_per_pair)
        self.map_norm = LazyLayerNorm(axis=-1)  # normalizes flattened map vectors with lazy feature inference
        self.cls_norm = nn.LayerNorm(self.cls_layer.n_cls_flat)
        self.cls_embedding_layers = nn.ModuleList(cls_embedding_layers)
        self.regression_head_layers = nn.ModuleList(regression_head_layers)

    def forward(self, inputs, training: bool | None = None):
        maps, cls = inputs
        x = self.gcnn(maps)
        x_flat = x.reshape(x.shape[0], -1)
        x_flat = self.map_norm(x_flat)
        cls_flat = self.cls_norm(self.cls_layer(cls))
        for layer in self.cls_embedding_layers:
            cls_flat = layer(cls_flat)
        x = torch.cat([x_flat, cls_flat], dim=-1)
        for layer in self.regression_head_layers:
            x = layer(x)
        return x
