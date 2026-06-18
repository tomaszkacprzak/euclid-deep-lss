# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics
"""PyTorch regression-head builders used by Deep LSS networks."""

from __future__ import annotations

from torch import nn

from deep_lss.nets import deepsphere_torch as dst
import logging

try:
    from msfm.utils import logger
except ImportError:
    logger = None

LOGGER = logger.get_logger(__file__) if logger is not None else logging.getLogger(__name__)
MeanLayer = dst.MeanLayer


def get_regression_head(
    out_features,
    head_type="dense",
    dense_layers=None,
    second_to_last_features=None,
    activation="relu",
    dropout_rate=None,
    poly_degree=5,
    norm_kwargs=None,
):
    """Build a regression head as a list of ``torch.nn.Module`` layers.

    Existing YAML keys are preserved: ``second_to_last_features`` is treated as
    a one-layer ``dense_layers`` specification when ``dense_layers`` is absent.
    """
    norm_kwargs = norm_kwargs or {}
    layers: list[nn.Module] = []

    if head_type == "dense":
        LOGGER.info("Using a dense regression head")
        if dense_layers is None and second_to_last_features is not None:
            dense_layers = [second_to_last_features]

        layers.append(dst.Flatten())
        layers.append(dst.LazyLayerNorm(**norm_kwargs))

        if dense_layers is not None:
            LOGGER.warning(f"Using dense_layers={dense_layers} in the regression head")
            for h in dense_layers:
                layers.append(dst.dense(h, activation=activation))
                layers.append(dst.LazyLayerNorm(**norm_kwargs))

        if dropout_rate is not None:
            assert not dense_layers, "Dropout and hidden dense layers should not be used together"
            LOGGER.warning(f"Using dropout with probability {dropout_rate} in the regression head")
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.LazyLinear(out_features))

    elif head_type == "conv":
        assert dropout_rate is None, "Dropout not supported for convolutional head"
        LOGGER.info("Using a convolutional + averaging regression head")
        layers.append(dst.LazyLayerNorm(**norm_kwargs))
        layers.append(dst.healpy_chebyshev(K=poly_degree, Fout=out_features, activation=None))
        layers.append(dst.MeanLayer(axis=-2))
    else:
        raise ValueError(f"Unknown regression head type: {head_type}")

    return layers


def get_cls_embedding_layers(hidden_layers, dropout_rate=None, activation="relu"):
    """Build a PyTorch MLP for binned Cl embeddings."""
    if not hidden_layers:
        return []
    layers: list[nn.Module] = []
    for h in hidden_layers:
        layers.append(dst.dense(h, activation=activation))
        layers.append(dst.LazyLayerNorm())
    if dropout_rate is not None:
        layers.append(nn.Dropout(dropout_rate))
    return layers
