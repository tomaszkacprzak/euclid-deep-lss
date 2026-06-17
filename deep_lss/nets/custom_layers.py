import numpy as np
import torch
from torch import nn

try:
    from msfm.utils import scales
except ModuleNotFoundError:
    class _ScalesFallback:
        @staticmethod
        def gaussian_high_pass_factor_alm(l, l_min):
            return np.ones_like(l, dtype=np.float32) if l_min is None else (l >= l_min).astype(np.float32)
        @staticmethod
        def gaussian_low_pass_factor_alm(l, l_max, theta_fwhm=None, arcmin=None):
            return np.ones_like(l, dtype=np.float32) if l_max is None else (l <= l_max).astype(np.float32)
    scales = _ScalesFallback()


class MeanBinningLayer(nn.Module):
    """Bin a (batch, dim1, dim2) tensor along axis 1 and return per-bin means."""

    def __init__(self, bin_edges, **kwargs):
        super().__init__()
        self.register_buffer("bin_edges", torch.as_tensor(bin_edges, dtype=torch.float32), persistent=False)
        self.num_bins = len(bin_edges) - 1

    def forward(self, inputs):
        dim1 = inputs.shape[1]
        indices = torch.arange(dim1, dtype=torch.float32, device=inputs.device)
        bin_indices = torch.bucketize(indices, self.bin_edges[1:-1].to(inputs.device))
        bin_indices = torch.where(indices < self.bin_edges[0].to(inputs.device), 0, bin_indices)
        bin_indices = torch.where(indices >= self.bin_edges[-1].to(inputs.device), self.num_bins - 1, bin_indices)
        means = []
        for i in range(self.num_bins):
            mask = (bin_indices == i).to(dtype=inputs.dtype).view(1, dim1, 1)
            means.append((inputs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8))
        return torch.stack(means, dim=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_bins, input_shape[2])


class PowerSpectrumSmoothingLayer(nn.Module):
    def __init__(self, n_cls, l_min=None, l_max=None, theta_fwhm=None, arcmin=None):
        super().__init__()
        l = np.arange(n_cls)
        band_pass_fac = (
            scales.gaussian_high_pass_factor_alm(l, l_min) ** 2
            * scales.gaussian_low_pass_factor_alm(l, l_max, theta_fwhm, arcmin) ** 2
        )
        self.register_buffer("band_pass_fac", torch.as_tensor(band_pass_fac, dtype=torch.float32))

    def forward(self, inputs):
        if inputs.ndim == 1:
            return inputs * self.band_pass_fac.to(inputs.device, inputs.dtype)
        if inputs.ndim in [2, 3]:
            return inputs * self.band_pass_fac.to(inputs.device, inputs.dtype).view(1, -1, *([1] if inputs.ndim == 3 else []))
        raise ValueError("Invalid input shape. Expected 1D, 2D, or 3D tensor.")
