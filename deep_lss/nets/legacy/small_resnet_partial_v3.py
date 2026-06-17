from deep_lss.nets.deepsphere_torch import healpy_layers
import torch
from torch import nn

from deep_lss.nets.legacy.torch_legacy_utils import AxisLayerNorm, ChannelsLastConv1d, Dense, Flatten, Lambda, MixingDense

"""
This file contains the specifications for the training, e.g. the network layers
"""

# quick and dirty layer

class ResnetIdentityBlock(nn.Module):
    def __init__(self, kernel_size, filters):
        super().__init__()

        self.conv2a = ChannelsLastConv1d(filters, kernel_size, stride=1, padding="same", activation="relu")
        self.bn2a = AxisLayerNorm(axis=-1)

        self.conv2b = ChannelsLastConv1d(filters, kernel_size, stride=1, padding="same", activation="relu")
        self.bn2b = AxisLayerNorm(axis=-1)

    
    def forward(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x)
        x = torch.relu(x)
 
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = torch.relu(x)

        x += input_tensor
        return torch.relu(x)


# Define if this network is intended for baryons or not
#######################################################

with_bary = False

# get the number of params according to bary
if with_bary:
    n_params = 6
else:
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
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          Flatten(),
          AxisLayerNorm(axis=-1),
          Dense(n_params)]
