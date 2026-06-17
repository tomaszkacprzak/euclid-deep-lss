# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""PyTorch likelihood losses."""

import torch

from deep_lss.utils import summary
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def _fill_upper_triangular(values, n):
    upper = values.new_zeros((*values.shape[:-1], n, n))
    idx = torch.triu_indices(n, n, device=values.device)
    upper[..., idx[0], idx[1]] = values
    return upper


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
    """Calculate the negative likelihood loss as a scalar PyTorch tensor."""

    LOGGER.warning("Tracing neg_likelihood_loss")

    if not torch.is_tensor(predictions):
        predictions = torch.as_tensor(predictions)
    if not torch.is_tensor(theta_true):
        theta_true = torch.as_tensor(theta_true, dtype=predictions.dtype, device=predictions.device)
    else:
        theta_true = theta_true.to(device=predictions.device, dtype=predictions.dtype)

    n_triang_with_diag = n_theta * (n_theta + 1) // 2
    theta_pred, cov_pred = torch.split(predictions, [n_theta, n_triang_with_diag], dim=1)
    residual = theta_pred - theta_true

    upper_triangular = _fill_upper_triangular(cov_pred, n_theta)

    if img_summary:
        mean_upper_triangular = torch.mean(upper_triangular, dim=0, keepdim=True)
        upper_triangular_img = mean_upper_triangular.unsqueeze(-1)
        mean_cov = torch.matmul(mean_upper_triangular, mean_upper_triangular.transpose(-1, -2))
        cov_img = mean_cov.unsqueeze(-1)
        if not xla:
            summary.write_summary("likelihood_tri_img" + summary_suffix, upper_triangular_img, summary_writer, training, summary_type="image")
            summary.write_summary("likelihood_cov_img" + summary_suffix, cov_img, summary_writer, training, summary_type="image")

    diag = torch.diagonal(upper_triangular, dim1=-2, dim2=-1)
    diag = diag + diag.new_tensor(eps)
    log_det = torch.sum(torch.log(torch.square(diag)), dim=1)
    mean_log_det = -torch.mean(log_det)

    Lt_residual = torch.sum(upper_triangular * residual.unsqueeze(1), dim=-1)
    Lt_residual_norm = torch.sum(torch.square(Lt_residual), dim=1)
    mean_Lt_residual_norm = torch.mean(Lt_residual_norm)

    if not xla:
        summary.write_summary("loss/likelihood_log_det" + summary_suffix, mean_log_det, summary_writer, training)
        summary.write_summary("loss/likelihood_residual" + summary_suffix, mean_Lt_residual_norm, summary_writer, training)

    loss = mean_Lt_residual_norm + mean_log_det

    if lambda_tikhonov is not None:
        lambda_tikhonov = torch.as_tensor(lambda_tikhonov, dtype=predictions.dtype, device=predictions.device)
        frob_norm = torch.linalg.norm(upper_triangular, ord="fro", dim=(-2, -1))
        tikhonov_loss = -lambda_tikhonov * torch.mean(frob_norm)
        if not xla:
            summary.write_summary("loss/likelihood_tikhonov" + summary_suffix, tikhonov_loss, summary_writer, training)
        loss = loss + tikhonov_loss

    return loss.mean()
