# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics
"""PyTorch negative-likelihood losses."""

import torch


def fill_triangular(x, upper=True):
    """PyTorch equivalent of ``tfp.math.fill_triangular`` for batched vectors."""
    n_tril = x.shape[-1]
    n = int((int((8 * n_tril + 1) ** 0.5) - 1) // 2)
    if n * (n + 1) // 2 != n_tril:
        raise ValueError(f"Last dimension {n_tril} does not describe a triangular matrix")
    out = x.new_zeros(*x.shape[:-1], n, n)
    idx = torch.triu_indices(n, n, device=x.device) if upper else torch.tril_indices(n, n, device=x.device)
    out[..., idx[0], idx[1]] = x
    return out


def neg_likelihood_loss(
    predictions,
    theta_true,
    n_theta,
    lambda_tikhonov=None,
    eps=1e-30,
    training=False,
    summary_writer=None,
    summary_suffix="",
    img_summary=False,
    xla=False,
):
    """Negative likelihood loss using a predicted precision Cholesky factor.

    The first ``n_theta`` outputs are the predicted mean; the remaining outputs parameterize an upper-triangular
    matrix ``Lᵀ``.  The loss matches the legacy formula ``||Lᵀ(theta_pred-theta_true)||² - logdet(LᵀL)``.
    Summary-writing arguments are accepted for config compatibility and ignored.
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.as_tensor(predictions, dtype=torch.float32)
    theta_true = torch.as_tensor(theta_true, dtype=predictions.dtype, device=predictions.device)
    n_triang_with_diag = n_theta * (n_theta + 1) // 2
    theta_pred, cov_pred = torch.split(predictions, [n_theta, n_triang_with_diag], dim=1)
    residual = theta_pred - theta_true
    upper_triangular = fill_triangular(cov_pred, upper=True)
    diag = torch.diagonal(upper_triangular, dim1=-2, dim2=-1) + eps
    mean_log_det = -torch.mean(torch.sum(torch.log(torch.square(diag)), dim=1))
    lt_residual = torch.matmul(upper_triangular, residual.unsqueeze(-1)).squeeze(-1)
    mean_lt_residual_norm = torch.mean(torch.sum(torch.square(lt_residual), dim=1))
    loss = mean_lt_residual_norm + mean_log_det
    if lambda_tikhonov is not None:
        loss = loss - lambda_tikhonov * torch.linalg.matrix_norm(upper_triangular, ord="fro", dim=(-2, -1)).mean()
    return loss
