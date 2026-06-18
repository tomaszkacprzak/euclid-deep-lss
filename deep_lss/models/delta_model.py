# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""PyTorch delta-loss trainer wrappers."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from msfm.utils import logger
from deep_lss.models.base_model import BaseModel

LOGGER = logger.get_logger(__file__)


class DeltaLossModel(BaseModel):
    """BaseModel specialization for PyTorch delta-loss training."""

    def setup_delta_loss_step(
        self,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        clip_by_value=None,
        clip_by_norm=None,
        clip_by_global_norm=5.0,
        l2_norm_weight=None,
        **loss_kwargs: Any,
    ):
        """Create ``delta_train_step`` using standard PyTorch autograd.

        The original TensorFlow delta objective must be provided as a PyTorch
        callable via ``loss_fn``.  The callable receives the network predictions
        and any keyword arguments supplied to this setup method.
        """
        if loss_fn is None:
            raise NotImplementedError("Pass a PyTorch delta loss callable via loss_fn")

        self.vali_loss_fn = lambda preds: loss_fn(preds, training=False, **loss_kwargs)

        def delta_train_step(input_batch):
            if isinstance(input_batch, torch.Tensor):
                input_batch = input_batch.to(self.device, non_blocking=True)
            return self.base_train_step(
                input_tensor=input_batch,
                loss_function=loss_fn,
                input_labels=None,
                clip_by_value=clip_by_value,
                clip_by_norm=clip_by_norm,
                clip_by_global_norm=clip_by_global_norm,
                l2_norm_weight=l2_norm_weight,
                **loss_kwargs,
            )

        self.delta_train_step = delta_train_step
        LOGGER.info("Set up PyTorch delta training step")
        return delta_train_step
