# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen
"""

from pathlib import Path

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is an optional runtime dependency
    SummaryWriter = None


class TensorBoardLogger:
    """Small TensorBoard logging abstraction used by models and trainers.

    Keeping TensorBoard-specific calls behind this class lets model code call a
    stable ``write`` API while future logging backends can be swapped in here.
    """

    def __init__(self, log_dir, enabled=True, global_step=0):
        self.writer = None
        self.global_step = int(global_step)
        if log_dir is not None and enabled:
            if SummaryWriter is None:
                raise ImportError("torch.utils.tensorboard.SummaryWriter requires tensorboard to be installed")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))

    def set_step(self, step):
        self.global_step = int(step)

    def increment_step(self, delta=1):
        self.global_step += int(delta)

    def write(self, label, value, summary_type="scalar", step=None, print_scalar=False):
        if self.writer is None:
            return
        step = self.global_step if step is None else int(step)
        value = _to_loggable_value(value)
        if summary_type == "scalar":
            self.writer.add_scalar(label, value, step)
            if print_scalar:
                print(f"{label}: {value}")
        elif summary_type == "histogram":
            self.writer.add_histogram(label, value, step)
        elif summary_type == "image":
            if torch.is_tensor(value) and value.ndim == 4:
                self.writer.add_images(label, value, step)
            else:
                self.writer.add_image(label, value, step)
        else:
            raise ValueError(f"Invalid summary type {summary_type} was passed")

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()


def _to_loggable_value(value):
    """Detach tensors and move them to CPU before TensorBoard logging."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
    return value


def write_summary(label, value, summary_writer, training=True, summary_type="scalar", print_scalar=False, step=None):
    """Handle different kinds of summaries to TensorBoard.

    Args:
        label (str): The name of the summary.
        value: The value to log. Torch tensors are detached and moved to CPU.
        summary_writer: A :class:`TensorBoardLogger` or ``torch.utils.tensorboard.SummaryWriter``.
        training (bool, optional): Only log during training. Defaults to True.
        summary_type (str, optional): One of ``scalar``, ``histogram`` or ``image``. Defaults to ``scalar``.
        print_scalar (bool, optional): Print scalar values to the console. Defaults to False.
        step (int, optional): Integer global step. Defaults to the logger's current step if available.

    Raises:
        ValueError: If an invalid summary_type is passed.
    """
    if summary_writer is None or not training:
        return
    if hasattr(summary_writer, "write"):
        summary_writer.write(label, value, summary_type=summary_type, step=step, print_scalar=print_scalar)
        return

    # Compatibility for callers that pass a raw PyTorch SummaryWriter.
    step = int(0 if step is None else step)
    value = _to_loggable_value(value)
    if summary_type == "scalar":
        summary_writer.add_scalar(label, value, step)
        if print_scalar:
            print(f"{label}: {value}")
    elif summary_type == "histogram":
        summary_writer.add_histogram(label, value, step)
    elif summary_type == "image":
        if torch.is_tensor(value) and value.ndim == 4:
            summary_writer.add_images(label, value, step)
        else:
            summary_writer.add_image(label, value, step)
    else:
        raise ValueError(f"Invalid summary type {summary_type} was passed")
