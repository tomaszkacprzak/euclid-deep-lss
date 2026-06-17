from deep_lss.nets.deepsphere_torch import healpy_layers
import torch
from torch import nn

from deep_lss.nets.legacy.torch_legacy_utils import AxisLayerNorm, ChannelsLastConv1d, Dense, Flatten, Lambda, MixingDense

"""
This file contains the specifications for the training, e.g. the network layers
"""

# Define if this network is intended for baryons or not
#######################################################

with_bary = True

# get the number of params according to bary
n_params = 4

# tags for training
param_ind = [0, 1, 5, 6]

# Define the layers
###################

bn_kwargs = dict()
layers = [healpy_layers.HealpyPseudoConv(p=1, Fout=32, activation="relu"),
          healpy_layers.HealpyPseudoConv(p=1, Fout=64, activation="relu"),
          healpy_layers.HealpyPseudoConv(p=1, Fout=128, activation="relu"),
          healpy_layers.HealpyChebyshev(K=5, Fout=256, activation="relu"),
          AxisLayerNorm(axis=-1),
          healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation="relu"),
          healpy_layers.HealpyChebyshev(K=5, Fout=256, activation="relu"),
          AxisLayerNorm(axis=-1),
          healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation="relu"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          # mix the disconnected parts
          MixingDense(activation="relu"),
          AxisLayerNorm(axis=1),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          # mix the disconnected parts
          MixingDense(activation="relu"),
          AxisLayerNorm(axis=1),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          # mix the disconnected parts
          MixingDense(activation="relu"),
          AxisLayerNorm(axis=1),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          # mix the disconnected parts
          MixingDense(activation="relu"),
          AxisLayerNorm(axis=1),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          # mix the disconnected parts
          MixingDense(activation="relu"),
          AxisLayerNorm(axis=1),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": "relu",
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          Flatten(),
          AxisLayerNorm(axis=-1),
          Dense(n_params)]
