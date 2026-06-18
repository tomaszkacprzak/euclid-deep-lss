"""Project adapters for the PyTorch ``deepsphere-cosmo-pytorch`` API.

The upstream package still exposes the historical ``deepsphere`` import path,
but its public classes are now ``torch.nn.Module`` objects.  This module keeps
all project-specific construction in one place so network builders can preserve
our existing YAML keys while returning native PyTorch modules.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F

try:
    from deepsphere import HealpyGCNN as _HealpyGCNN
    from deepsphere import healpy_layers
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise ImportError(
        "deepsphere-cosmo-pytorch is required for DeepSphere networks. Install the project dependencies first."
    ) from exc


def torch_activation(activation: Optional[Any]) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Translate legacy Keras-style activation names/callables to PyTorch callables."""
    if activation is None:
        return None
    if activation in ("linear", "identity"):
        return None
    if callable(activation):
        return activation
    mapping = {
        "relu": F.relu,
        "elu": F.elu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "softplus": F.softplus,
    }
    if activation not in mapping:
        raise ValueError(f"Unsupported activation for PyTorch DeepSphere adapter: {activation!r}")
    return mapping[activation]


class LambdaLayer(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class MeanLayer(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=self.axis)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=1)


class SequentialWithTraining(nn.Sequential):
    """Sequential that accepts and ignores Keras-style ``training=...``."""

    def forward(self, input, training=None):  # noqa: A002 - keep compatibility with Keras call sites
        if training is not None:
            self.train(bool(training))
        return super().forward(input)


class LazyLayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5, elementwise_affine: bool = True, **kwargs: Any):
        super().__init__()
        kwargs.pop("axis", None)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.norm: Optional[nn.LayerNorm] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            self.norm = nn.LayerNorm(x.shape[-1], eps=self.eps, elementwise_affine=self.elementwise_affine).to(
                device=x.device, dtype=x.dtype
            )
        return self.norm(x)


def dense(out_features: int, activation: Optional[Any] = None, **kwargs: Any) -> nn.Module:
    modules: list[nn.Module] = [nn.LazyLinear(out_features)]
    act = torch_activation(activation)
    if act is not None:
        modules.append(LambdaLayer(act))
    return nn.Sequential(*modules)


def healpy_smoothing(**kwargs: Any) -> nn.Module:
    return healpy_layers.HealpySmoothing(**kwargs)


def healpy_pseudo_conv(p: int, Fout: int, activation: Optional[Any] = None, **kwargs: Any) -> nn.Module:
    return healpy_layers.HealpyPseudoConv(p=p, Fout=Fout, activation=torch_activation(activation), **kwargs)


def healpy_chebyshev(K: int, Fout: Optional[int] = None, activation: Optional[Any] = None, **kwargs: Any) -> Any:
    return healpy_layers.HealpyChebyshev(K=K, Fout=Fout, activation=torch_activation(activation), **kwargs)


def healpy_residual(
    layer_type: str, layer_kwargs: dict[str, Any], activation: Optional[Any] = None, **kwargs: Any
) -> Any:
    layer_kwargs = dict(layer_kwargs)
    layer_kwargs["activation"] = torch_activation(layer_kwargs.get("activation"))
    return healpy_layers.Healpy_ResidualLayer(
        layer_type, layer_kwargs=layer_kwargs, activation=torch_activation(activation), **kwargs
    )


def healpy_vit(**kwargs: Any) -> nn.Module:
    kwargs = dict(kwargs)
    kwargs["activation"] = torch_activation(kwargs.get("activation"))
    return healpy_layers.Healpy_ViT(**kwargs)


def healpy_transformer(**kwargs: Any) -> Any:
    kwargs = dict(kwargs)
    kwargs["activation"] = torch_activation(kwargs.get("activation"))
    return healpy_layers.Healpy_Transformer(**kwargs)


def healpy_gcnn(
    *, nside, indices, layers: Iterable[Any], n_neighbors=8, max_batch_size=None, initial_Fin=None
) -> nn.Module:
    return _HealpyGCNN(
        nside=nside,
        indices=indices,
        layers=list(layers),
        n_neighbors=n_neighbors,
        max_batch_size=max_batch_size,
        initial_Fin=initial_Fin,
    )
