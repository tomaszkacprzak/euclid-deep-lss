# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2024
Author: Arne Thomsen

PyTorch optimizer and learning-rate scheduler helpers.
"""

import math

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, _LRScheduler

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def _torch_optimizer_kwargs(config):
    """Translate legacy Keras optimizer kwargs to torch.optim kwargs."""
    kwargs = dict(config or {})
    beta_1 = kwargs.pop("beta_1", None)
    beta_2 = kwargs.pop("beta_2", None)
    if beta_1 is not None or beta_2 is not None:
        kwargs["betas"] = (0.9 if beta_1 is None else float(beta_1), 0.999 if beta_2 is None else float(beta_2))
    return kwargs


class LinearWarmupCosineDecaySchedule(_LRScheduler):
    """Linear warmup followed by cosine decay.

    The constructor intentionally keeps the field names used by the previous
    Keras schedule configuration: ``initial_learning_rate`` is the first warmup
    LR, ``warmup_target`` is the peak/base LR, ``decay_steps`` is the number of
    post-warmup scheduler steps, and ``alpha`` is the final LR as a fraction of
    ``warmup_target``.
    """

    def __init__(self, optimizer, initial_learning_rate, warmup_steps, warmup_target, decay_steps, alpha, last_epoch=-1):
        self.warmup_init_learning_rate = float(initial_learning_rate)
        self.warmup_steps = int(warmup_steps)
        self.learning_rate = float(warmup_target)
        self.decay_steps = max(1, int(decay_steps))
        self.decay_alpha = float(alpha)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = step / float(self.warmup_steps)
            lr = self.warmup_init_learning_rate + progress * (self.learning_rate - self.warmup_init_learning_rate)
        else:
            decay_step = min(step - self.warmup_steps, self.decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / self.decay_steps))
            decayed = (1.0 - self.decay_alpha) * cosine + self.decay_alpha
            lr = self.learning_rate * decayed
        return [lr for _ in self.optimizer.param_groups]


def get_scheduler(optimizer, net_conf, loss_function="delta"):
    """Return a torch LR scheduler and cadence (``"step"`` or ``"epoch"``)."""
    assert loss_function in ["delta", "likelihood", "mutual_info"]
    loss_key = loss_function + "_loss"
    opt_conf = net_conf["optimization"][loss_key]
    scheduler = opt_conf.get("scheduler")
    learning_rate = float(opt_conf["learning_rate"])

    if scheduler is None:
        LOGGER.info(f"Using constant learning rate {learning_rate}")
        return None, None
    if scheduler == "cosine":
        warmup_steps = int(opt_conf["warmup_steps"])
        decay_steps = int(net_conf["training"]["n_steps"]) - warmup_steps
        LOGGER.info("Using cosine learning rate schedule with warmup")
        return (
            LinearWarmupCosineDecaySchedule(
                optimizer=optimizer,
                initial_learning_rate=float(opt_conf["warmup_init_learning_rate"]),
                warmup_steps=warmup_steps,
                warmup_target=learning_rate,
                decay_steps=decay_steps,
                alpha=float(opt_conf["decay_alpha"]),
            ),
            "step",
        )
    if scheduler == "warmup":
        start_factor = float(opt_conf["warmup_init_learning_rate"]) / learning_rate
        LOGGER.info("Using linear warmup learning rate schedule")
        return (
            LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=int(opt_conf["warmup_steps"]),
            ),
            "step",
        )
    raise NotImplementedError(f"Scheduler {scheduler} not implemented yet")


def get_optimizer(net_conf, loss_function="delta", restore_checkpoint=False, parameters=None):
    """Create a PyTorch optimizer plus LR scheduler.

    Returns:
        tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None, str | None]:
        optimizer, scheduler, and scheduler cadence. Step-cadence schedulers
        should be advanced once after every optimizer update; epoch-cadence
        schedulers should be advanced after each epoch.
    """
    del restore_checkpoint  # Optimizer state is restored by torch checkpoints, not constructed here.
    if parameters is None:
        raise ValueError("PyTorch optimizers require model parameters; pass parameters=model.parameters().")
    assert loss_function in ["delta", "likelihood", "mutual_info"]
    loss_key = loss_function + "_loss"
    opt_conf = net_conf["optimization"][loss_key]
    learning_rate = float(opt_conf["learning_rate"])
    optimizer_name = net_conf["optimization"]["optimizer"].lower()
    kwargs = _torch_optimizer_kwargs(opt_conf.get("optimizer_kwargs", {}))

    if optimizer_name == "adam":
        optimizer = Adam(parameters, lr=learning_rate, **kwargs)
        LOGGER.info("Using torch.optim.Adam optimizer")
    elif optimizer_name == "adamw":
        optimizer = AdamW(parameters, lr=learning_rate, **kwargs)
        LOGGER.info("Using torch.optim.AdamW optimizer")
    elif optimizer_name == "sgd":
        optimizer = SGD(parameters, lr=learning_rate, **kwargs)
        LOGGER.info("Using torch.optim.SGD optimizer")
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    scheduler, cadence = get_scheduler(optimizer, net_conf, loss_function)
    return optimizer, scheduler, cadence
