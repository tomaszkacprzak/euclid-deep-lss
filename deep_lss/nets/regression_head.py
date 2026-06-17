# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

from torch import nn
from deep_lss.nets.deepsphere_torch import healpy_layers
from deep_lss.nets.torch_utils import Flatten, LazyLayerNorm, LinearActivation, MeanLayer
import logging
try:
    from msfm.utils import logger
    LOGGER = logger.get_logger(__file__)
except ModuleNotFoundError:
    LOGGER = logging.getLogger(__file__)


def get_regression_head(out_features, head_type="dense", dense_layers=None, activation="relu", dropout_rate=None,
                        poly_degree=5, norm_kwargs={}, second_to_last_features=None):
    layers = []
    if dense_layers is None and second_to_last_features is not None:
        dense_layers = [second_to_last_features]
    if head_type == "dense":
        LOGGER.info("Using a dense regression head")
        layers.append(Flatten())
        layers.append(nn.LayerNorm(norm_kwargs["normalized_shape"]) if "normalized_shape" in norm_kwargs else LazyLayerNorm(axis=-1))
        if dense_layers is not None:
            LOGGER.warning(f"Using dense_layers={dense_layers} in the regression head")
            for h in dense_layers:
                layers.append(LinearActivation(h, activation=activation))
                layers.append(nn.LayerNorm(h))
        if dropout_rate is not None:
            assert not dense_layers, "Dropout and hidden dense layers should not be used together"
            LOGGER.warning(f"Using dropout with probability {dropout_rate} in the regression head")
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.LazyLinear(out_features))
    elif head_type == "conv":
        assert dropout_rate is None, "Dropout not supported for convolutional head"
        LOGGER.info("Using a convolutional + averaging regression head")
        layers.append(nn.Identity())
        layers.append(healpy_layers.HealpyChebyshev(K=poly_degree, Fout=out_features, activation=None))
        layers.append(MeanLayer(axis=-2))
    else:
        raise ValueError(f"Unknown regression head type: {head_type}")
    return layers


def get_cls_embedding_layers(hidden_layers, dropout_rate=None, activation="relu"):
    if not hidden_layers:
        return []
    layers = []
    for h in hidden_layers:
        layers.append(LinearActivation(h, activation=activation))
        layers.append(nn.LayerNorm(h))
    if dropout_rate is not None:
        layers.append(nn.Dropout(dropout_rate))
    return layers
