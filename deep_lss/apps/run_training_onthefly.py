# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Train the DeepSphere graph neural networks at the fiducial cosmology and its perturbations using the information
maximizing loss to find an informative summary statistic.

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
"""

import os, sys, threading, warnings
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)

import argparse, yaml, wandb, shutil

from datetime import datetime
from time import time
from contextlib import nullcontext

from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear
from msfm.utils import logger, input_output, files, parameters

from deep_lss.utils import distribute, configuration, evaluation, optimization
from deep_lss.models.grid_model import GridLossModel
from deep_lss.nets import NETWORKS
from deep_lss.nets.regression_head import get_cls_embedding_layers

LOGGER = logger.get_logger(__file__)

BATCH_SIZE = 24
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50000
# TODO: add more constant variables if needed

def setup():
    description = "Train the specified network at the fiducial cosmology."
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument("-v", "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )

    # TODO: add arguments for webdataset pattern
  
    args, _ = parser.parse_known_args()

    # TODO: setup

    return args

def main():

    # TODO: implement
    # TODO: use OntheflyPhysicsModelLinear
    # TODO: use HealpyGCNN resnet
    # TODO: use data distributed training
    # TODO: use pytorch only (no tensorflow anywhere!)
    # TODO: support checkpoints, re-starting training from a checkpoint
    # TODO: support wandb
    # TODO: use variational mutual information maximization loss function, only
    # TODO: support validation on held-out part of the data, every 1000 steps
    # TODO: monitor validation loss and training loss in wandb
    # TODO: training arguments, such as learning rate, batch size, number of epochs, should be hard-coded at the top of the file
    # TODO: make the script as simple as possible, no calls to external functions, classes, modules in this repository, only standard library and external packages that are installed
    

    pass

if __name__ == "__main__":

    args = setup()

    main()

