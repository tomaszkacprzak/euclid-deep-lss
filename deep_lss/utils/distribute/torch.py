# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""PyTorch distributed training helpers.

This module replaces the legacy TensorFlow/Horovod distribution utilities with
small wrappers around :mod:`torch.distributed`,
:class:`torch.nn.parallel.DistributedDataParallel`, and
:class:`torch.utils.data.distributed.DistributedSampler`.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TorchInputContext:
    """Input-pipeline context compatible with existing dataset factories."""

    num_input_pipelines: int
    input_pipeline_id: int


class TorchDistributedContext:
    """Runtime context for PyTorch single-process and DDP training."""

    def __init__(self, backend: str = "none", init_method: Optional[str] = None, timeout: Any = None):
        if backend not in {"none", "single_gpu", "ddp"}:
            raise ValueError(f"Unknown PyTorch distribution backend {backend!r}")

        self.backend = backend
        self.distributed = backend == "ddp"
        self.rank = _env_int("RANK", _env_int("SLURM_PROCID", 0))
        self.local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))
        self.world_size = _env_int("WORLD_SIZE", _env_int("SLURM_NTASKS", 1)) if self.distributed else 1

        if backend in {"single_gpu", "ddp"} and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank % torch.cuda.device_count())
            self.device = torch.device("cuda", self.local_rank % torch.cuda.device_count())
        else:
            self.device = torch.device("cpu")

        if self.distributed:
            if not dist.is_available():
                raise RuntimeError("torch.distributed is not available in this PyTorch build")
            if not dist.is_initialized():
                selected_backend = "nccl" if torch.cuda.is_available() else "gloo"
                kwargs = {"backend": selected_backend, "init_method": init_method or "env://"}
                if timeout is not None:
                    kwargs["timeout"] = timeout
                dist.init_process_group(**kwargs)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            LOGGER.warning(
                f"Training is distributed with PyTorch DDP ({self.world_size} ranks, rank {self.rank}, "
                f"local rank {self.local_rank}, device {self.device})"
            )
        elif backend == "single_gpu":
            LOGGER.warning(f"Training on a single PyTorch device: {self.device}")
        else:
            LOGGER.warning("Training is not distributed")

        self.num_replicas_in_sync = self.world_size
        self.replica_id = self.rank

    def scope(self):
        """No-op context manager for compatibility with older training setup code."""

        return contextlib.nullcontext()

    def input_context(self) -> TorchInputContext:
        return TorchInputContext(num_input_pipelines=self.world_size, input_pipeline_id=self.rank)

    def distribute_datasets_from_function(self, dataset_fn: Callable[[TorchInputContext], Any]) -> Any:
        """Call an existing dataset factory with rank/world-size sharding metadata."""

        return dataset_fn(self.input_context())

    def run(self, fn: Callable[..., Any], args: tuple = (), kwargs: Optional[dict] = None) -> Any:
        return fn(*args, **(kwargs or {}))

    def reduce(self, reduce_op: Any, value: Any, axis: Any = None) -> Any:
        """Reduce a scalar/tensor across all ranks for simple metric aggregation."""

        if not self.distributed:
            return value
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=self.device)
        tensor = tensor.clone().to(self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if str(reduce_op).upper().endswith("MEAN") or str(reduce_op).upper() == "MEAN":
            tensor /= self.world_size
        return tensor

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def broadcast_object(self, obj: Any, root_rank: int = 0) -> Any:
        if not self.distributed:
            return obj
        values = [obj if self.rank == root_rank else None]
        dist.broadcast_object_list(values, src=root_rank)
        return values[0]

    def wrap_model(self, model: torch.nn.Module, **ddp_kwargs: Any) -> torch.nn.Module:
        """Move ``model`` to this context's device and wrap it in DDP when needed."""

        model = model.to(self.device)
        if not self.distributed:
            return model
        if self.device.type == "cuda" and "device_ids" not in ddp_kwargs:
            ddp_kwargs["device_ids"] = [self.device.index]
            ddp_kwargs.setdefault("output_device", self.device.index)
        return DistributedDataParallel(model, **ddp_kwargs)

    def create_sampler(self, dataset: Dataset, shuffle: bool = True, drop_last: bool = False) -> Optional[DistributedSampler]:
        """Return a ``DistributedSampler`` for map-style datasets under DDP."""

        if not self.distributed:
            return None
        if isinstance(dataset, IterableDataset):
            raise TypeError("Iterable datasets must be explicitly sharded with shard_iterable_dataset()")
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def shard_iterable_dataset(self, dataset: Iterable[Any]) -> Iterator[Any]:
        """Shard an iterable/TFRecord-style dataset by yielding items for this rank only."""

        for index, item in enumerate(dataset):
            if index % self.world_size == self.rank:
                yield item


    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        """Create a DataLoader using DistributedSampler for map-style DDP datasets.

        Iterable/TFRecord-style datasets are not implicitly sampled; callers should
        apply shard_iterable_dataset() or pass rank/world-size to their TFRecord
        reader so records are explicitly partitioned across workers.
        """

        if isinstance(dataset, IterableDataset):
            if self.distributed:
                dataset = self.shard_iterable_dataset(dataset)
            return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, **dataloader_kwargs)

        sampler = self.create_sampler(dataset, shuffle=shuffle, drop_last=drop_last)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=drop_last,
            **dataloader_kwargs,
        )

    def shutdown(self) -> None:
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def setup_torch_distributed(backend: Optional[str] = None, **kwargs: Any) -> TorchDistributedContext:
    """Create a PyTorch distribution context for ``none``, ``single_gpu``, or ``ddp``."""

    return TorchDistributedContext(backend=backend or "none", **kwargs)
