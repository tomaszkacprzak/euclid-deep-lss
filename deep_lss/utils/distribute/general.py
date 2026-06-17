# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

PyTorch distributed training utilities.
"""

from __future__ import annotations

import os
import socket
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.nn.parallel import DistributedDataParallel
import wandb

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _expand_slurm_nodelist(nodelist: str) -> str:
    """Return the first host in a Slurm nodelist."""
    if not nodelist:
        return "127.0.0.1"
    try:
        output = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True)
        return output.splitlines()[0]
    except Exception as exc:  # pragma: no cover - depends on Slurm utilities
        LOGGER.warning(f"Could not expand SLURM_NODELIST={nodelist!r} with scontrol ({exc}); using hostname")
        return socket.gethostname()


def detect_rank_env() -> dict[str, int]:
    """Detect distributed rank information from torchrun or Slurm environment variables."""
    rank = _env_int("RANK", _env_int("SLURM_PROCID", 0))
    local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))
    world_size = _env_int("WORLD_SIZE", _env_int("SLURM_NTASKS", 1))
    local_world_size = _env_int("LOCAL_WORLD_SIZE", _env_int("SLURM_NTASKS_PER_NODE", 1))
    return {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "world_size": int(world_size),
        "local_world_size": int(local_world_size),
    }


def configure_slurm_environment(default_port: int = 29500) -> None:
    """Populate torch.distributed env:// variables when launched directly under Slurm."""
    if "SLURM_PROCID" not in os.environ:
        return

    rank_env = detect_rank_env()
    os.environ.setdefault("RANK", str(rank_env["rank"]))
    os.environ.setdefault("LOCAL_RANK", str(rank_env["local_rank"]))
    os.environ.setdefault("WORLD_SIZE", str(rank_env["world_size"]))
    os.environ.setdefault("LOCAL_WORLD_SIZE", str(rank_env["local_world_size"]))
    os.environ.setdefault("MASTER_PORT", str(default_port))
    os.environ.setdefault("MASTER_ADDR", _expand_slurm_nodelist(os.environ.get("SLURM_NODELIST", "")))


class TorchInputContext:
    """Small input context compatible with the previous dataset-factory API."""

    def __init__(self, num_input_pipelines: int, input_pipeline_id: int):
        self.num_input_pipelines = num_input_pipelines
        self.input_pipeline_id = input_pipeline_id


@dataclass
class TorchClusterResolver:
    task_id: int
    task_type: str = "worker"
    cluster_spec: Any = None


class TorchDistributedStrategy:
    """Compatibility wrapper backed by native ``torch.distributed`` collectives."""

    def __init__(self, distributed: bool = False):
        self.distributed = distributed and torch.distributed.is_available() and torch.distributed.is_initialized()
        self.num_replicas_in_sync = torch.distributed.get_world_size() if self.distributed else 1
        self.replica_id = torch.distributed.get_rank() if self.distributed else 0
        self.local_rank = detect_rank_env()["local_rank"]
        self.cluster_resolver = TorchClusterResolver(task_id=self.replica_id)

    def distribute_datasets_from_function(self, dataset_fn: Callable[[TorchInputContext], Any]) -> Any:
        return dataset_fn(TorchInputContext(self.num_replicas_in_sync, self.replica_id))

    def scope(self):
        return nullcontext()

    def run(self, fn: Callable[..., Any], args: tuple = (), kwargs: dict | None = None) -> Any:
        return fn(*args, **(kwargs or {}))

    def gather(self, tensor: torch.Tensor, axis: int = 0) -> torch.Tensor:
        if not self.distributed:
            return tensor
        if axis != 0:
            tensor = tensor.transpose(0, axis).contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(self.num_replicas_in_sync)]
        torch.distributed.all_gather(gathered, tensor)
        result = torch.cat(gathered, dim=0)
        return result if axis == 0 else result.transpose(0, axis).contiguous()

    def reduce(self, reduce_op: Any, value: torch.Tensor | float | int, axis: int | tuple | None = None) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if not self.distributed:
            return value
        result = value.clone()
        op_name = getattr(reduce_op, "name", str(reduce_op)).upper()
        if "SUM" in op_name:
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        elif "MEAN" in op_name or "AVG" in op_name:
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
            result = result / self.num_replicas_in_sync
        else:
            raise NotImplementedError(f"TorchDistributedStrategy.reduce only implements SUM and MEAN, got {reduce_op}")
        return result

    def broadcast(self, tensor: torch.Tensor, root_rank: int = 0) -> torch.Tensor:
        if self.distributed:
            torch.distributed.broadcast(tensor, src=root_rank)
        return tensor

    def broadcast_object(self, obj: Any, root_rank: int = 0) -> Any:
        if not self.distributed:
            return obj
        objects = [obj if self.replica_id == root_rank else None]
        torch.distributed.broadcast_object_list(objects, src=root_rank)
        return objects[0]

    def barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier()

    def wrap_model(self, module: torch.nn.Module, **ddp_kwargs: Any) -> DistributedDataParallel:
        if not self.distributed:
            raise RuntimeError("wrap_model requires an initialized torch.distributed process group")
        if torch.cuda.is_available():
            ddp_kwargs.setdefault("device_ids", [self.local_rank])
            ddp_kwargs.setdefault("output_device", self.local_rank)
        return DistributedDataParallel(module, **ddp_kwargs)


def init_process_group(backend: str | None = None, init_method: str = "env://", set_cuda_device: bool = True) -> TorchDistributedStrategy:
    configure_slurm_environment()
    rank_env = detect_rank_env()
    if set_cuda_device and torch.cuda.is_available():
        torch.cuda.set_device(rank_env["local_rank"])
    if rank_env["world_size"] > 1 and not torch.distributed.is_initialized():
        backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        torch.distributed.init_process_group(backend=backend, init_method=init_method)
        LOGGER.warning(
            f"Training is distributed with torch.distributed backend={backend}, "
            f"rank={torch.distributed.get_rank()}/{torch.distributed.get_world_size()}, local_rank={rank_env['local_rank']}"
        )
    else:
        LOGGER.warning("Training is not distributed, using a single PyTorch process")
    return TorchDistributedStrategy(distributed=rank_env["world_size"] > 1)


def get_strategy(strategy_name: str | bool | None = None) -> TorchDistributedStrategy:
    """Return a native PyTorch distributed strategy wrapper.

    Accepted distributed aliases are ``torch``, ``torchrun``, ``ddp``, ``distributed`` and ``slurm``.
    """
    if strategy_name in (None, False, "none", "single"):
        return TorchDistributedStrategy(distributed=False)
    if strategy_name in (True, "torch", "torchrun", "ddp", "distributed", "slurm"):
        return init_process_group()
    raise ValueError(f"Unknown distribution strategy {strategy_name}")


def check_devices() -> tuple[int | None, int]:
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpus = os.cpu_count()
    LOGGER.info(f"Running on {n_cpus} CPU cores")
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        LOGGER.warning("No CUDA GPUs discovered by PyTorch, running on CPUs only")
    else:
        LOGGER.info(f"Individual task(s) running on {n_gpus} GPU(s)")
    return n_cpus, n_gpus


def get_local_batch_size(strategy: TorchDistributedStrategy, global_batch_size: int) -> int:
    n_replicas = strategy.num_replicas_in_sync
    if global_batch_size % n_replicas != 0:
        raise ValueError(
            f"The global batch size {global_batch_size} has to be divisible by the number of synced replicas {n_replicas}"
        )
    local_batch_size = global_batch_size // n_replicas
    LOGGER.info(f"Using the local batch size {local_batch_size}")
    return local_batch_size


def get_global_batch_size(strategy: TorchDistributedStrategy, local_batch_size: int) -> int:
    global_batch_size = int(local_batch_size * strategy.num_replicas_in_sync)
    LOGGER.info(f"Using the global batch size {global_batch_size}")
    return global_batch_size


def get_wandb_group_name(strategy: TorchDistributedStrategy) -> str | None:
    if strategy.num_replicas_in_sync == 1:
        return None
    group_name = wandb.util.generate_id() if strategy.replica_id == 0 else None
    group_name = strategy.broadcast_object(group_name, root_rank=0)
    LOGGER.info(f"group = {group_name}")
    return group_name
