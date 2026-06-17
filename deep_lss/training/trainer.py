"""Shared PyTorch training runtime helpers for app entry points.

This module centralizes device selection, native ``torch.distributed`` setup,
autocast/GradScaler mixed precision, checkpoint IO, scalar summaries, and small
schedule helpers so command-line apps can migrate from legacy framework-specific
boilerplate incrementally while keeping their YAML/CLI contracts stable.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from deep_lss.utils import distribute
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

from deep_lss.utils.summary import TensorBoardLogger


@dataclass
class TrainerRuntime:
    strategy: distribute.TorchDistributedStrategy
    device: torch.device
    mixed_precision: bool = False
    mixed_precision_dtype: torch.dtype = torch.float16
    grad_accum_steps: int = 1

    def __post_init__(self) -> None:
        self.grad_accum_steps = max(1, int(self.grad_accum_steps))
        self.scaler = make_grad_scaler(self.device, self.mixed_precision, self.mixed_precision_dtype)

    @property
    def is_chief(self) -> bool:
        return self.strategy.replica_id == 0

    def autocast(self):
        return autocast_context(self.device, self.mixed_precision, self.mixed_precision_dtype)

    def backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, step_index: int) -> bool:
        """Backpropagate and step ``optimizer`` when accumulation is complete."""
        scaled = loss / self.grad_accum_steps
        if self.scaler.is_enabled():
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()
        should_step = (step_index + 1) % self.grad_accum_steps == 0
        if should_step:
            if self.scaler.is_enabled():
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return should_step


def dtype_from_name(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    mapping = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    try:
        return mapping[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported mixed precision dtype {name!r}") from exc


def select_device(strategy: distribute.TorchDistributedStrategy | None = None) -> torch.device:
    if torch.cuda.is_available():
        index = getattr(strategy, "local_rank", 0) if strategy is not None else torch.cuda.current_device()
        return torch.device("cuda", index)
    return torch.device("cpu")


def create_runtime(dist_strategy=None, mixed_precision=False, mixed_precision_dtype="float16", grad_accum_steps=1) -> TrainerRuntime:
    _, _ = distribute.check_devices()
    strategy = distribute.get_strategy(dist_strategy)
    device = select_device(strategy)
    LOGGER.info(f"PyTorch version {torch.__version__}")
    LOGGER.info(f"Using PyTorch device {device}")
    if mixed_precision:
        LOGGER.warning(f"Using PyTorch autocast mixed precision dtype={mixed_precision_dtype}")
    return TrainerRuntime(strategy, device, bool(mixed_precision), dtype_from_name(mixed_precision_dtype), grad_accum_steps)


def make_grad_scaler(device: torch.device, enabled: bool, dtype: torch.dtype):
    enabled = bool(enabled and device.type == "cuda" and dtype == torch.float16)
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@contextmanager
def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        yield
        return
    try:
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True):
            yield
    except (AttributeError, TypeError):
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype, enabled=True):
                yield
        else:
            with nullcontext():
                yield


def load_checkpoint(path: str | Path, model=None, optimizer=None, scheduler=None, scaler=None, map_location=None, strict=True) -> dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    if model is not None:
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(state.get("model", state), strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state


def save_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, **metadata) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    state = {"model": module.state_dict(), **metadata}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    return str(path)


class ScalarWriter:
    def __init__(self, log_dir: str | Path | None, enabled: bool = True):
        self.logger = TensorBoardLogger(log_dir, enabled=enabled) if log_dir is not None else None

    @property
    def writer(self):
        return None if self.logger is None else self.logger.writer

    def scalar(self, name: str, value: float, step: int) -> None:
        if self.logger is not None:
            self.logger.write(name, value, summary_type="scalar", step=int(step))

    def flush(self) -> None:
        if self.logger is not None:
            self.logger.flush()

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()


class CosineWithWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0, alpha: float = 0.0, last_epoch: int = -1):
        import math
        total_steps = max(1, int(total_steps))
        warmup_steps = max(0, int(warmup_steps))
        def fn(step):
            if warmup_steps and step < warmup_steps:
                return max(1e-12, float(step + 1) / float(warmup_steps))
            progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
            return alpha + (1.0 - alpha) * 0.5 * (1.0 + math.cos(math.pi * progress))
        super().__init__(optimizer, fn, last_epoch=last_epoch)

# Compatibility shim used only while app-level YAML/CLI migrations are staged.
class _TorchMean:
    def __init__(self): self.values=[]
    def update_state(self, v): self.values.append(torch.as_tensor(v).detach())
    def result(self): return torch.stack(self.values).mean() if self.values else torch.tensor(float('nan'))

class _TorchMSE:
    def __call__(self, y_pred, y_true): return torch.mean((y_pred - y_true) ** 2)

class _Schedules:
    class PolynomialDecay:
        def __init__(self, initial_learning_rate, decay_steps, end_learning_rate=0.0, power=1.0, **kw):
            self.initial_learning_rate=initial_learning_rate; self.decay_steps=max(1,decay_steps); self.end_learning_rate=end_learning_rate; self.power=power
        def __call__(self, step):
            step=min(float(step), self.decay_steps); frac=(1-step/self.decay_steps)**self.power
            return self.end_learning_rate + (self.initial_learning_rate-self.end_learning_rate)*frac
    class CosineDecay:
        def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warmup_steps=0, **kw):
            import math; self.math=math; self.lr=initial_learning_rate; self.decay_steps=max(1,decay_steps); self.alpha=alpha; self.warmup_steps=warmup_steps
        def __call__(self, step):
            if self.warmup_steps and step < self.warmup_steps: return self.lr*float(step+1)/self.warmup_steps
            p=min(1.0, max(0.0, (float(step)-self.warmup_steps)/max(1,self.decay_steps-self.warmup_steps)))
            return self.lr*(self.alpha+(1-self.alpha)*0.5*(1+self.math.cos(self.math.pi*p)))

class _Summary:
    @staticmethod
    @contextmanager
    def record_if(_): yield

class _ProfilerExperimental:
    @staticmethod
    def Trace(*a, **k): return nullcontext()
    @staticmethod
    def start(*a, **k): return None
    @staticmethod
    def stop(*a, **k): return None

class _TorchCompat:
    float32=torch.float32
    keras=type('keras', (), {'optimizers': type('optimizers', (), {'schedules': _Schedules}), 'metrics': type('metrics', (), {'Mean': _TorchMean, 'MeanSquaredError': _TorchMSE})})
    math=type('math', (), {'is_nan': staticmethod(torch.isnan)})
    summary=_Summary
    profiler=type('profiler', (), {'experimental': _ProfilerExperimental})
    @staticmethod
    def function(fn=None, **_): return (lambda f: f) if fn is None else fn
    @staticmethod
    def Variable(value, trainable=False, dtype=None): return torch.as_tensor(value, dtype=dtype)
    @staticmethod
    def zeros(shape, dtype=torch.float32): return torch.zeros(shape, dtype=dtype)
    @staticmethod
    def constant(value, dtype=None): return torch.as_tensor(value, dtype=dtype)
    @staticmethod
    def reshape(x, shape): return torch.reshape(torch.as_tensor(x), tuple(shape))
    @staticmethod
    def slice(x, begin, size):
        sl=tuple(slice(b, None if sz == -1 else b+sz) for b,sz in zip(begin,size)); return x[sl]
    sqrt=staticmethod(torch.sqrt); reduce_mean=staticmethod(torch.mean); square=staticmethod(torch.square)
    @staticmethod
    def cast(x, dtype): return torch.as_tensor(x, dtype=dtype, device=x.device if torch.is_tensor(x) else None)

torch_compat = _TorchCompat()
