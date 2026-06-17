# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""Deprecated TensorFlow distribution entry points.

Distributed support has migrated to :mod:`deep_lss.utils.distribute.general`, which uses native
``torch.distributed`` and ``DistributedDataParallel``. This module remains only to produce a clear migration error
for stale imports.
"""


def setup_tf_distribute_mirrored_strategy():
    raise RuntimeError("TensorFlow distribution support was removed; use distribute.get_strategy('ddp') instead.")


def setup_tf_distribute_multi_worker_mirrored_strategy():
    raise RuntimeError("TensorFlow distribution support was removed; launch with torchrun/Slurm and use ddp instead.")
