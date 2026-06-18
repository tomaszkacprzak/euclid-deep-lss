# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics
"""PyTorch ResNet-style DeepSphere layer builder."""

from __future__ import annotations

from deep_lss.nets import deepsphere_torch as dst
from deep_lss.nets.regression_head import get_regression_head
import logging

try:
    from msfm.utils import logger
except ImportError:
    logger = None

LOGGER = logger.get_logger(__file__) if logger is not None else logging.getLogger(__name__)


class ResNetLayers:
    def __init__(
        self,
        out_features,
        # convolutions
        n_base_channels=32,
        n_downsampling=3,
        n_cheby=2,
        n_residuals=6,
        # regression head
        head_type="dense",
        dense_layers=None,
        dropout_rate=None,
        poly_degree=5,
        norm_kwargs=None,
        activation="relu",
        smoothing_kwargs=None,
    ) -> None:
        norm_kwargs = norm_kwargs or {}
        self.layers = []

        if smoothing_kwargs is not None:
            self.layers.append(dst.healpy_smoothing(**smoothing_kwargs))
        else:
            LOGGER.warning("No smoothing layer is included in the network")

        n_channels = base_channels
        for _ in range(downsampling_layers):
            self.layers.append(dst.healpy_pseudo_conv(p=1, Fout=n_channels, activation=activation))
            n_channels *= 2

        for _ in range(cheby_layers):
            self.layers.append(dst.healpy_chebyshev(K=poly_degree, Fout=n_channels, activation=activation))
            self.layers.append(dst.LazyLayerNorm(**norm_kwargs))
            self.layers.append(dst.healpy_pseudo_conv(p=1, Fout=n_channels, activation=activation))

        for _ in range(residual_layers):
            self.layers.append(
                dst.healpy_residual(
                    "CHEBY",
                    layer_kwargs={"K": poly_degree, "activation": activation, "use_bias": True},
                    use_bn=True,
                    bn_kwargs=norm_kwargs,
                    norm_type="layer_norm",
                )
            )

        self._conv_layers = list(self.layers)
        regression_head_layers = get_regression_head(
            out_features=out_features,
            head_type=head_type,
            dense_layers=dense_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            poly_degree=poly_degree,
            norm_kwargs=norm_kwargs,
        )
        self._head_layers_no_flatten = regression_head_layers[1:] if head_type == "dense" else regression_head_layers
        self.layers.extend(regression_head_layers)

    def get_layers(self):
        return self.layers

    def get_conv_layers(self):
        return self._conv_layers

    def get_head_layers_no_flatten(self):
        return self._head_layers_no_flatten
