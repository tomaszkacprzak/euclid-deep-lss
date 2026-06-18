# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""Deprecated Horovod distribution helpers.

The project now uses PyTorch distributed training. Import
``deep_lss.utils.distribute.torch`` for ``torch.distributed`` setup,
``DistributedDataParallel`` wrapping, and dataset sharding helpers.
"""


class HorovodStrategy:
    """Deprecated placeholder for legacy type checks."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "HorovodStrategy is deprecated. Use --dist_strategy ddp and "
            "deep_lss.utils.distribute.torch.TorchDistributedContext instead."
        )


def setup_horovod():
    raise RuntimeError(
        "Horovod distribution support is deprecated. Use --dist_strategy ddp and "
        "torch.distributed.init_process_group instead."
    )
