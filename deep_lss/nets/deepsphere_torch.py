"""Project adapter for the PyTorch DeepSphere package.

The runtime standardizes on the upstream PyTorch package installed as::

    deepsphere @ git+https://github.com/deepsphere/deepsphere-pytorch.git

That package exposes modules such as ``deepsphere.models.spherical_unet``,
``deepsphere.layers.chebyshev``, ``deepsphere.layers.samplings`` and
``deepsphere.utils.laplacian_funcs``.  The older TensorFlow cosmology package
exposed ``deepsphere.HealpyGCNN`` and ``deepsphere.healpy_layers``; import
DeepSphere objects through this adapter instead of depending on either upstream
layout directly.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Iterable


def require_deepsphere():
    """Raise a helpful error if the standardized PyTorch DeepSphere package is unavailable."""
    try:
        return importlib.import_module("deepsphere")
    except ImportError as exc:
        raise ImportError(
            "PyTorch DeepSphere is required. Install the project dependency "
            "`deepsphere @ git+https://github.com/deepsphere/deepsphere-pytorch.git`."
        ) from exc


def _load(path: str, name: str):
    require_deepsphere()
    return getattr(importlib.import_module(path), name)


def ChebConv(*args, **kwargs):
    return _load("deepsphere.layers.chebyshev", "ChebConv")(*args, **kwargs)


def SphericalChebConv(*args, **kwargs):
    return _load("deepsphere.layers.chebyshev", "SphericalChebConv")(*args, **kwargs)


def cheb_conv(*args, **kwargs):
    return _load("deepsphere.layers.chebyshev", "cheb_conv")(*args, **kwargs)


def Healpix(*args, **kwargs):
    return _load("deepsphere.layers.samplings.healpix_pool_unpool", "Healpix")(*args, **kwargs)


def HealpixAvgPool(*args, **kwargs):
    return _load("deepsphere.layers.samplings.healpix_pool_unpool", "HealpixAvgPool")(*args, **kwargs)


def HealpixMaxPool(*args, **kwargs):
    return _load("deepsphere.layers.samplings.healpix_pool_unpool", "HealpixMaxPool")(*args, **kwargs)


def SphericalUNet(*args, **kwargs):
    return _load("deepsphere.models.spherical_unet", "SphericalUNet")(*args, **kwargs)


def SphericalUNetTemporalConv(*args, **kwargs):
    return _load("deepsphere.models.spherical_unet", "SphericalUNetTemporalConv")(*args, **kwargs)


def SphericalUNetTemporalLSTM(*args, **kwargs):
    return _load("deepsphere.models.spherical_unet", "SphericalUNetTemporalLSTM")(*args, **kwargs)


def prepare_laplacian(*args, **kwargs):
    return _load("deepsphere.utils.laplacian_funcs", "prepare_laplacian")(*args, **kwargs)


def scipy_csr_to_sparse_tensor(*args, **kwargs):
    return _load("deepsphere.utils.laplacian_funcs", "scipy_csr_to_sparse_tensor")(*args, **kwargs)


def get_healpix_laplacians(*args, **kwargs):
    return _load("deepsphere.utils.laplacian_funcs", "get_healpix_laplacians")(*args, **kwargs)


build_healpix_laplacians = get_healpix_laplacians


def MapGCNN(*, layers: Iterable, nside=None, indices=None, n_neighbors=20, **kwargs):
    """Build the project-level PyTorch map branch from graph-aware modules.

    PyTorch DeepSphere does not provide the old TensorFlow ``HealpyGCNN``
    constructor.  Runtime model construction should instantiate graph-aware
    PyTorch modules directly and pass them here as a regular module sequence.
    Graph metadata is retained as attributes for checkpoint/debug provenance.
    """
    require_deepsphere()
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    module = nn.Sequential(*layers)
    module.nside = nside
    module.indices = indices
    module.n_neighbors = n_neighbors
    return module


MapBranch = MapGCNN
HealpyGCNN = MapGCNN
HealpyPooling = Healpix
HealpyAvgPool = HealpixAvgPool
HealpyDownsampling = HealpixAvgPool
HealpyChebyshev = SphericalChebConv


class _UnavailableDeepSphereLayer:
    """Deferred-error module for DeepSphere layers not present in the PyTorch API."""

    def __init__(self, *args, **kwargs):
        nn = importlib.import_module("torch.nn")

        class _Layer(nn.Module):
            def __init__(self, args, kwargs):
                super().__init__()
                self.args = args
                self.kwargs = kwargs

            def forward(self, *inputs, **forward_kwargs):
                raise NotImplementedError(
                    "This TensorFlow DeepSphere layer is not available in the standardized PyTorch DeepSphere API. "
                    "Construct an equivalent torch.nn.Module and import PyTorch DeepSphere primitives from "
                    "deep_lss.nets.deepsphere_torch."
                )

        self.module = _Layer(args, kwargs)

    def __call__(self):
        return self.module


def HealpySmoothing(*args, **kwargs):
    return _UnavailableDeepSphereLayer(*args, **kwargs)()


def _legacy_layer(*args, **kwargs):
    return _UnavailableDeepSphereLayer(*args, **kwargs)()


healpy_layers = SimpleNamespace(
    HealpySmoothing=HealpySmoothing,
    HealpyChebyshev=HealpyChebyshev,
    HealpyPseudoConv=_legacy_layer,
    Healpy_ResidualLayer=_legacy_layer,
    Healpy_ViT=_legacy_layer,
    Healpy_Transformer=_legacy_layer,
)

__all__ = [
    "MapGCNN", "MapBranch", "HealpyGCNN", "HealpySmoothing", "HealpyChebyshev",
    "HealpyPooling", "HealpyAvgPool", "HealpyDownsampling", "Healpix", "HealpixAvgPool",
    "HealpixMaxPool", "SphericalUNet", "SphericalUNetTemporalConv", "SphericalUNetTemporalLSTM",
    "ChebConv", "SphericalChebConv", "cheb_conv", "build_healpix_laplacians",
    "get_healpix_laplacians", "prepare_laplacian", "scipy_csr_to_sparse_tensor",
    "healpy_layers", "require_deepsphere",
]
