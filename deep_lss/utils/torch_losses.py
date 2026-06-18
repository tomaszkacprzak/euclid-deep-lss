"""PyTorch loss helpers for feature regularization."""

import torch


def compute_vicreg_var_cov_loss(z, eps=1e-4):
    batch_size = z.shape[0]
    feature_dim = z.shape[1]
    z_centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(torch.mean(torch.square(z_centered), dim=0) + eps)
    var_loss = torch.mean(torch.square(std - 1.0))
    cov_matrix = torch.matmul(z_centered.T, z_centered) / (batch_size - 1)
    cov_loss = torch.sum(torch.square(cov_matrix)) - torch.sum(torch.square(torch.diagonal(cov_matrix)))
    cov_loss = cov_loss / (feature_dim**2 - feature_dim)
    return var_loss, cov_loss


def compute_vicreg_invariance_loss(z, pair_ids, return_diagnostics=False):
    match = torch.all(pair_ids[:, None, :] == pair_ids[None, :, :], dim=-1)
    mask = match & ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    mask_f = mask.to(dtype=z.dtype)
    n_pair_entries = mask_f.sum()
    diff = z[:, None, :] - z[None, :, :]
    pairwise_mse = torch.mean(torch.square(diff), dim=-1)
    loss = torch.nan_to_num(torch.sum(pairwise_mse * mask_f) / n_pair_entries.clamp_min(1.0), nan=0.0)
    if return_diagnostics:
        return loss, {
            "n_positive_pairs": n_pair_entries / 2.0,
            "n_anchored_samples": torch.any(mask, dim=1).to(z.dtype).sum(),
        }
    return loss


def _rbf_kernel(x, y, bandwidths=None):
    feature_dim = x.shape[1]
    dim_scale = torch.sqrt(torch.as_tensor(feature_dim, dtype=x.dtype, device=x.device))
    if bandwidths is None:
        bandwidths = [0.1 * dim_scale, 1.0 * dim_scale, 10.0 * dim_scale]
    xx = torch.sum(torch.square(x), dim=1, keepdim=True)
    yy = torch.sum(torch.square(y), dim=1, keepdim=True)
    distances = torch.relu(xx - 2 * torch.matmul(x, y.T) + yy.T)
    kernel = torch.zeros_like(distances)
    for bandwidth in bandwidths:
        kernel = kernel + torch.exp(-distances / (2 * bandwidth**2))
    return kernel / len(bandwidths)


def compute_mmd_loss(z, interpretable=False, z_gaussian=None):
    if z_gaussian is None:
        z_gaussian = torch.randn_like(z)
    batch_size = z.shape[0]
    norm = float(batch_size * batch_size)
    mmd_loss = torch.sum(_rbf_kernel(z, z)) / norm - 2 * torch.sum(_rbf_kernel(z, z_gaussian)) / norm
    if interpretable:
        mmd_loss = mmd_loss + torch.sum(_rbf_kernel(z_gaussian, z_gaussian)) / norm
    return mmd_loss


def compute_sw_loss(z, num_projections=None, method="analytical", z_gaussian=None):
    feature_dim = z.shape[1]
    if num_projections is None:
        num_projections = max(512, feature_dim)
    projections = torch.randn(feature_dim, num_projections, dtype=z.dtype, device=z.device)
    projections = torch.nn.functional.normalize(projections, dim=0)
    sorted_z = torch.sort(torch.matmul(z, projections), dim=0).values
    if method == "analytical":
        probs = (torch.arange(z.shape[0], dtype=z.dtype, device=z.device) + 0.5) / z.shape[0]
        sorted_gaussian = torch.distributions.Normal(0.0, 1.0).icdf(probs).unsqueeze(-1)
    elif method == "sample":
        if z_gaussian is None:
            z_gaussian = torch.randn_like(z)
        sorted_gaussian = torch.sort(torch.matmul(z_gaussian, projections), dim=0).values
    else:
        raise ValueError(f"Invalid method {method}. Must be 'sample' or 'analytical'.")
    return torch.mean(torch.square(sorted_z - sorted_gaussian))


class ZMemoryBank:
    def __init__(self, size=None):
        self.size = size
        self.bank = None
        self.index = 0

    def update_and_get(self, z):
        if self.size is None:
            return z, z.new_tensor(1.0)
        if self.bank is None:
            self.bank = torch.randn(self.size, z.shape[-1], dtype=z.dtype, device=z.device)
        batch_size = z.shape[0]
        indices = (torch.arange(batch_size, device=z.device) + self.index) % self.size
        self.bank[indices] = z.detach()
        self.index = int((self.index + batch_size) % self.size)
        return torch.cat([z, self.bank], dim=0), z.new_tensor((batch_size + self.size) / batch_size)
