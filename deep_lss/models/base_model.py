# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""PyTorch training utilities for deep_lss models.

This module intentionally contains no TensorFlow or Horovod dependencies.  The
:class:`BaseModel` class is a small trainer around a ``torch.nn.Module`` with
checkpointing, mixed precision, gradient clipping, TensorBoard/W&B logging, and
``DataLoader``-oriented train/evaluation loops.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)
Batch = Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]
LossFunction = Callable[..., torch.Tensor]


class _Logger:
    """Single logging abstraction for TensorBoard and optional W&B."""

    def __init__(self, summary_dir: Optional[Union[str, os.PathLike[str]]] = None, wandb_run: Any = None):
        self.writer = SummaryWriter(str(summary_dir)) if summary_dir is not None else None
        self.wandb_run = wandb_run

    def scalar(self, name: str, value: Any, step: int) -> None:
        value = _to_python_number(value)
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
        if self.wandb_run is not None:
            self.wandb_run.log({name: value}, step=step)

    def histogram(self, name: str, value: torch.Tensor, step: int) -> None:
        if self.writer is not None:
            self.writer.add_histogram(name, value.detach().cpu(), step)
        if self.wandb_run is not None:
            self.wandb_run.log({name: value.detach().cpu()}, step=step)

    def image(self, name: str, value: torch.Tensor, step: int) -> None:
        if self.writer is not None:
            self.writer.add_image(name, value.detach().cpu(), step)
        if self.wandb_run is not None:
            self.wandb_run.log({name: value.detach().cpu()}, step=step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _to_python_number(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        return value.item() if value.numel() == 1 else value.cpu()
    return value


def _move_to_device(batch: Batch, device: torch.device) -> Batch:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_to_device(value, device) for value in batch]
    return batch


def _split_batch(batch: Batch) -> Tuple[Any, Any, Tuple[Any, ...], Dict[str, Any]]:
    """Extract ``(inputs, labels, extras, kwargs)`` from common DataLoader batches."""
    if isinstance(batch, Mapping):
        inputs = batch.get("x", batch.get("inputs", batch.get("input")))
        labels = batch.get("y", batch.get("labels", batch.get("label", batch.get("theta"))))
        if inputs is None:
            raise ValueError("Mapping batches must contain one of: x, inputs, input")
        reserved = {"x", "inputs", "input", "y", "labels", "label", "theta"}
        kwargs = {k: v for k, v in batch.items() if k not in reserved}
        return inputs, labels, (), kwargs
    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Empty batches are not supported")
        inputs = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        return inputs, labels, tuple(batch[2:]), {}
    return batch, None, (), {}


class BaseModel(object):
    """PyTorch-oriented base trainer for ``torch.nn.Module`` networks."""

    def __init__(
        self,
        network: nn.Module,
        input_shape: Optional[Tuple[int, ...]] = None,
        optimizer: Optional[Union[str, Optimizer]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Any = None,
        summary_dir: Optional[Union[str, os.PathLike[str]]] = None,
        checkpoint_dir: Optional[Union[str, os.PathLike[str]]] = None,
        restore_checkpoint: bool = False,
        max_checkpoints: int = 3,
        init_step: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        mixed_precision: bool = False,
        summary_every: int = 1,
        wandb_run: Any = None,
        **_legacy_kwargs: Any,
    ):
        if not isinstance(network, nn.Module):
            raise TypeError(f"network must be a torch.nn.Module, got {type(network).__name__}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network = network.to(self.device)
        self.input_shape = input_shape
        self.optimizer = self._build_optimizer(optimizer, optimizer_kwargs or {})
        self.scheduler = scheduler
        self.summary_dir = Path(summary_dir) if summary_dir is not None else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.max_checkpoints = max_checkpoints
        self.global_step = int(init_step)
        self.epoch = 0
        self.summary_every = max(int(summary_every), 1)
        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.logger = _Logger(self.summary_dir, wandb_run=wandb_run)
        self.summary_writer = self.logger.writer
        self.metadata: Dict[str, Any] = {}

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if restore_checkpoint:
            self.restore_model()
        LOGGER.info("Initialized PyTorch BaseModel on device %s", self.device)

    def _build_optimizer(self, optimizer: Optional[Union[str, Optimizer]], kwargs: Dict[str, Any]) -> Optimizer:
        if isinstance(optimizer, Optimizer):
            return optimizer
        name = "adam" if optimizer is None else str(optimizer).lower()
        if name == "adam":
            return torch.optim.Adam(self.network.parameters(), **kwargs)
        if name == "adamw":
            return torch.optim.AdamW(self.network.parameters(), **kwargs)
        if name == "sgd":
            return torch.optim.SGD(self.network.parameters(), **kwargs)
        raise NotImplementedError(f"Optimizer {optimizer} is not implemented")

    def increment_step(self) -> None:
        self.global_step += 1

    def change_step(self, delta: int) -> None:
        self.global_step += int(delta)

    def set_step(self, step: int) -> None:
        self.global_step = int(step)

    def get_step(self) -> int:
        return int(self.global_step)

    def checkpoint_state(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": int(self.epoch),
            "step": int(self.global_step),
            "metadata": {**self.metadata, **(metadata or {})},
        }

    def save_model(self, metadata: Optional[Dict[str, Any]] = None, filename: Optional[str] = None) -> Path:
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory was declared during model initialization")
        filename = filename or f"ckpt-step-{self.global_step}.pt"
        path = self.checkpoint_dir / filename
        torch.save(self.checkpoint_state(metadata), path)
        self._prune_checkpoints()
        LOGGER.info("Saved PyTorch checkpoint to %s", path)
        return path

    def _prune_checkpoints(self) -> None:
        if self.checkpoint_dir is None or self.max_checkpoints is None or self.max_checkpoints <= 0:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("ckpt-step-*.pt"), key=lambda p: p.stat().st_mtime)
        for old_checkpoint in checkpoints[:-self.max_checkpoints]:
            old_checkpoint.unlink(missing_ok=True)

    def _latest_checkpoint(self) -> Path:
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory was declared during model initialization")
        checkpoints = sorted(self.checkpoint_dir.glob("ckpt-step-*.pt"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            raise ValueError(f"No PyTorch checkpoints found in {self.checkpoint_dir}")
        return checkpoints[-1]

    def restore_model(self) -> Dict[str, Any]:
        return self.restore_model_from_checkpoint_path(self._latest_checkpoint())

    def restore_model_from_checkpoint_path(self, checkpoint_path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("step", 0))
        self.metadata = dict(checkpoint.get("metadata", {}))
        LOGGER.info("Restored PyTorch checkpoint from %s", checkpoint_path)
        return checkpoint

    def write_summary(self, label: str, value: Any, summary_type: str = "scalar", skip: bool = False) -> None:
        if skip or self.global_step % self.summary_every != 0:
            return
        if summary_type == "scalar":
            self.logger.scalar(label, value, self.global_step)
        elif summary_type == "histogram":
            self.logger.histogram(label, value, self.global_step)
        elif summary_type == "image":
            self.logger.image(label, value, self.global_step)
        else:
            raise ValueError(f"Invalid summary type {summary_type}")

    def base_train_step(
        self,
        input_tensor: Any,
        loss_function: LossFunction,
        input_labels: Any = None,
        clip_by_value: Optional[Tuple[float, float]] = None,
        clip_by_norm: Optional[float] = None,
        clip_by_global_norm: Optional[float] = None,
        l2_norm_weight: Optional[float] = None,
        **loss_kwargs: Any,
    ) -> torch.Tensor:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            predictions = self.network(input_tensor)
            if input_labels is not None:
                loss = loss_function(predictions, input_labels, **loss_kwargs)
            else:
                loss = loss_function(predictions, **loss_kwargs)
            if l2_norm_weight is not None:
                l2_terms = [p.norm(2) for p in self.network.parameters() if p.requires_grad]
                l2_loss = torch.linalg.vector_norm(torch.stack(l2_terms))
                self.write_summary("loss/l2_reg", l2_loss)
                loss = loss + l2_norm_weight * l2_loss
        self.write_summary("loss/total", loss)

        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        parameters = [p for p in self.network.parameters() if p.grad is not None]
        for parameter in parameters:
            parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if clip_by_value is not None:
            clip_value = max(abs(float(clip_by_value[0])), abs(float(clip_by_value[1])))
            clip_grad_value_(parameters, clip_value=clip_value)
        if clip_by_norm is not None:
            for parameter in parameters:
                clip_grad_norm_([parameter], max_norm=float(clip_by_norm))
        grad_norm = (
            clip_grad_norm_(parameters, max_norm=float("inf"))
            if parameters
            else torch.tensor(0.0, device=self.device)
        )
        self.write_summary("global_grad_norm", grad_norm)
        if clip_by_global_norm is not None:
            clip_grad_norm_(parameters, max_norm=float(clip_by_global_norm))

        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.increment_step()
        self.write_summary("learning_rate", self.optimizer.param_groups[0].get("lr", 0.0))
        return loss.detach()

    def train_one_epoch(self, dataloader: DataLoader, loss_function: LossFunction, **step_kwargs: Any) -> float:
        if not isinstance(dataloader, DataLoader):
            raise TypeError("train_one_epoch expects a torch.utils.data.DataLoader")
        losses = []
        for batch in dataloader:
            inputs, labels, extras, kwargs = _split_batch(_move_to_device(batch, self.device))
            loss = self.base_train_step(inputs, loss_function, labels, extra_inputs=extras, **kwargs, **step_kwargs)
            losses.append(float(loss.cpu()))
        self.epoch += 1
        if self.scheduler is not None:
            self.scheduler.step()
        mean_loss = sum(losses) / max(len(losses), 1)
        self.write_summary("epoch/loss", mean_loss)
        return mean_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, loss_function: Optional[LossFunction] = None) -> Dict[str, float]:
        self.network.eval()
        losses = []
        for batch in dataloader:
            inputs, labels, extras, kwargs = _split_batch(_move_to_device(batch, self.device))
            predictions = self.network(inputs)
            if loss_function is not None:
                if labels is not None:
                    loss = loss_function(predictions, labels, extra_inputs=extras, **kwargs)
                else:
                    loss = loss_function(predictions, extra_inputs=extras, **kwargs)
                losses.append(float(loss.detach().cpu()))
        return {"loss": sum(losses) / max(len(losses), 1)} if losses else {}

    def build_network(self, input_shape: Tuple[int, ...]) -> None:
        dummy = torch.zeros(input_shape, device=self.device)
        self.network(dummy)

    def print_summary(self, **_kwargs: Any) -> None:
        LOGGER.info("%s", self.network)

    def delete_temp_summaries(self) -> None:
        if self.summary_dir is not None and self.summary_dir.exists():
            shutil.rmtree(self.summary_dir)

    def __call__(
        self, input_tensor: Any, training: bool = False, numpy: bool = False, *args: Any, **kwargs: Any
    ) -> Any:
        self.network.train(mode=training)
        with torch.set_grad_enabled(training):
            output = self.network(_move_to_device(input_tensor, self.device), *args, **kwargs)
        if numpy:
            return output.detach().cpu().numpy()
        return output
