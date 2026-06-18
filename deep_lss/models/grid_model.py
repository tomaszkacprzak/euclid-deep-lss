# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""PyTorch grid-loss trainer wrappers."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.nn import functional as F

from msfm.utils import logger
from deep_lss.models.base_model import BaseModel

LOGGER = logger.get_logger(__file__)


class GridLossModel(BaseModel):
    """BaseModel specialization for supervised grid-summary training in PyTorch."""

    def setup_grid_loss_step(
        self,
        loss: Any = "mse",
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        clip_by_value=None,
        clip_by_norm=None,
        clip_by_global_norm=10.0,
        l2_norm_weight=None,
        **loss_kwargs: Any,
    ):
        """Create ``grid_train_step`` using standard PyTorch autograd.

        ``loss`` may be ``"mse"``/``"mean_squared_error"`` or any callable
        accepting ``(predictions, labels, **kwargs)``.  A callable can also be
        supplied via ``loss_fn`` for compatibility with custom likelihood or
        mutual-information objectives implemented in PyTorch.
        """
        objective = loss_fn or loss
        if isinstance(objective, str):
            name = objective.lower()
            if name in {"mse", "mean_squared_error"}:
                objective = lambda preds, labels, **_: F.mse_loss(preds, labels)
            else:
                raise NotImplementedError(
                    f"Grid loss '{loss}' must be ported as a PyTorch callable and passed via loss_fn"
                )
        if not callable(objective):
            raise TypeError("Grid loss must be a string identifier or a callable")

        self.vali_loss_fn = lambda preds, theta: objective(preds, theta, training=False, **loss_kwargs)

        def grid_train_step(x, theta, *extra_inputs):
            x = x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
            theta = theta.to(self.device, non_blocking=True) if isinstance(theta, torch.Tensor) else theta
            return self.base_train_step(
                input_tensor=x,
                loss_function=objective,
                input_labels=theta,
                clip_by_value=clip_by_value,
                clip_by_norm=clip_by_norm,
                clip_by_global_norm=clip_by_global_norm,
                l2_norm_weight=l2_norm_weight,
                extra_inputs=extra_inputs,
                **loss_kwargs,
            )

        self.grid_train_step = grid_train_step
        self.grid_train_step_uses_pair_ids = False
        LOGGER.info("Set up PyTorch grid training step")
        return grid_train_step
