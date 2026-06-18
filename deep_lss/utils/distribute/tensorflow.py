# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""Deprecated TensorFlow distribution helpers.

The active distribution implementation lives in :mod:`deep_lss.utils.distribute.torch`.
This module is kept only to provide a clear migration error for legacy imports.
"""


def setup_tf_distribute_mirrored_strategy():
    raise RuntimeError(
        "TensorFlow distribution support is deprecated. Use --dist_strategy single_gpu or ddp "
        "and deep_lss.utils.distribute.torch.setup_torch_distributed instead."
    )


def setup_tf_distribute_multi_worker_mirrored_strategy():
    raise RuntimeError(
        "TensorFlow MultiWorkerMirroredStrategy support is deprecated. Use PyTorch DDP "
        "via --dist_strategy ddp and torch.distributed.init_process_group instead."
    )
