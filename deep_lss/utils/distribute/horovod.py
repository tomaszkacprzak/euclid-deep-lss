# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""Horovod compatibility aliases backed by native PyTorch distributed utilities."""

from msfm.utils import logger

from .general import TorchDistributedStrategy, init_process_group

LOGGER = logger.get_logger(__file__)


class HorovodStrategy(TorchDistributedStrategy):
    """Deprecated alias for the native PyTorch distributed strategy."""


def setup_horovod() -> TorchDistributedStrategy:
    LOGGER.warning("Horovod TensorFlow support was removed; using native torch.distributed/DDP instead")
    return init_process_group()
