# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""Utils for distributed training with PyTorch."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import wandb

from deep_lss.utils.distribute.torch import TorchDistributedContext, setup_torch_distributed
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def get_strategy(strategy_name=None):
    """Return a PyTorch distribution context.

    Args:
        strategy_name (str, optional): One of ``None``/``"none"``, ``"single_gpu"``, or ``"ddp"``.

    Returns:
        TorchDistributedContext: Context with rank/world-size metadata plus helpers for DDP.
    """
    try:
        n_tasks = int(os.environ["SLURM_NTASKS"])
        LOGGER.info(f"Running on {n_tasks} tasks in total")
    except KeyError:
        LOGGER.info("Running locally as SLURM_NTASKS is not set")

    if strategy_name in (None, "none"):
        return setup_torch_distributed("none")
    if strategy_name in ("single_gpu", "ddp"):
        return setup_torch_distributed(strategy_name)
    raise ValueError(f"Unknown distribution strategy {strategy_name}")


def check_devices():
    """Logs the number of discovered CPUs and GPUs using PyTorch."""
    try:
        n_cpus = len(os.sched_getaffinity(0))
        if n_cpus != os.cpu_count():
            LOGGER.debug(
                f"len(os.sched_getaffinity(0)) = {len(os.sched_getaffinity(0))} and",
                f" os.cpu_count() = {os.cpu_count()} disagree",
            )
    except AttributeError:
        n_cpus = os.cpu_count()
    LOGGER.info(f"Running on {n_cpus} CPU cores")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus == 0:
        LOGGER.warning("No GPU discovered by PyTorch, running on CPUs only")
    else:
        LOGGER.info(f"Individual task(s) running on {n_gpus} GPU(s)")

    try:
        n_gpus_cuda = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        assert n_gpus == n_gpus_cuda, f"The number of GPUs in PyTorch {n_gpus} and CUDA {n_gpus_cuda} should be equal"
    except KeyError:
        if n_gpus == 0:
            LOGGER.warning("No CUDA enabled GPUs found")

    return n_cpus, n_gpus


def get_world_size(strategy=None):
    if strategy is not None and hasattr(strategy, "world_size"):
        return int(strategy.world_size)
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank(strategy=None):
    if strategy is not None and hasattr(strategy, "rank"):
        return int(strategy.rank)
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_batch_size(strategy, global_batch_size):
    """Calculate per-rank batch size from global batch size and PyTorch world size."""
    world_size = get_world_size(strategy)
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"The global batch size {global_batch_size} has to be divisible by the PyTorch world size {world_size}"
        )
    local_batch_size = global_batch_size // world_size
    LOGGER.info(f"Using the local batch size {local_batch_size} on rank {get_rank(strategy)}")
    return local_batch_size


def get_global_batch_size(strategy, local_batch_size):
    """Calculate global batch size from per-rank batch size and PyTorch world size."""
    world_size = get_world_size(strategy)
    global_batch_size = int(local_batch_size * world_size)
    LOGGER.info(f"Using the global batch size {global_batch_size}")
    return global_batch_size


def get_wandb_group_name(strategy):
    """Generate a W&B group shared by all ranks in a distributed run."""
    if get_world_size(strategy) <= 1:
        return None
    group_name = wandb.util.generate_id() if get_rank(strategy) == 0 else None
    if isinstance(strategy, TorchDistributedContext):
        group_name = strategy.broadcast_object(group_name, root_rank=0)
    LOGGER.info(f"group = {group_name}")
    return group_name
