"""PyTorch maps-plus-Cls combined network."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from deep_lss.nets import deepsphere_torch as dst
import logging

try:
    from msfm.utils import logger
except ImportError:
    logger = None

LOGGER = logger.get_logger(__file__) if logger is not None else logging.getLogger(__name__)


class ClsBinningAndTransformLayer(nn.Module):
    def __init__(self, n_ell, n_bins, l_min_per_pair, l_max_per_pair, **kwargs):
        super().__init__()
        from msfm.utils.power_spectra import get_cl_bins

        n_z_cross = len(l_min_per_pair)
        ells = np.arange(n_ell, dtype=np.float64)
        W = np.zeros((n_ell, n_bins, n_z_cross), dtype=np.float32)
        for c, (lmin_c, lmax_c) in enumerate(zip(l_min_per_pair, l_max_per_pair)):
            bin_edges_c = get_cl_bins(lmin_c, lmax_c, n_bins + 1)
            for k in range(n_bins):
                in_bin = (ells >= bin_edges_c[k]) & (ells < bin_edges_c[k + 1])
                if in_bin.sum() > 0:
                    W[in_bin, k, c] = 1.0 / in_bin.sum()
        self.register_buffer("bin_weight", torch.as_tensor(W))
        self.n_cls_flat = n_bins * n_z_cross

    def forward(self, cls, training=None):
        cls_binned = torch.einsum("blc,lkc->bkc", cls.float(), self.bin_weight.to(cls.device, cls.dtype))
        cls_flat = cls_binned.reshape(cls_binned.shape[0], -1)
        return torch.sign(cls_flat) * torch.log(torch.abs(cls_flat) + 1e-10)


class MapsPlusCLSNetwork(nn.Module):
    def __init__(
        self,
        conv_layers,
        cls_embedding_layers,
        regression_head_layers,
        n_side,
        tfr_n_side,
        indices,
        n_neighbors,
        max_batch_size,
        initial_Fin,
        n_cls_bins,
        l_min_per_pair,
        l_max_per_pair,
    ):
        super().__init__()
        self.gcnn = dst.healpy_gcnn(
            nside=n_side,
            indices=indices,
            layers=conv_layers,
            n_neighbors=n_neighbors,
            max_batch_size=max_batch_size,
            initial_Fin=initial_Fin,
        )
        self.cls_layer = ClsBinningAndTransformLayer(3 * tfr_n_side, n_cls_bins, l_min_per_pair, l_max_per_pair)
        self.map_norm = dst.LazyLayerNorm()
        self.cls_norm = dst.LazyLayerNorm()
        self.cls_embedding_layers = nn.ModuleList(cls_embedding_layers)
        self.regression_head_layers = nn.ModuleList(regression_head_layers)

    def forward(self, inputs, training=None):
        if training is not None:
            self.train(bool(training))
        maps, cls = inputs
        x = self.gcnn(maps)
        x_flat = self.map_norm(torch.flatten(x, start_dim=1))
        cls_flat = self.cls_norm(self.cls_layer(cls))
        for layer in self.cls_embedding_layers:
            cls_flat = layer(cls_flat)
        x = torch.cat([x_flat, cls_flat], dim=-1)
        for layer in self.regression_head_layers:
            x = layer(x)
        return x
