# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics
"""PyTorch implementation of the delta loss used for information-maximising summaries."""

import numpy as np
import torch


def _as_tensor(x, *, dtype=None, device=None):
    if isinstance(x, torch.Tensor):
        if dtype is not None or device is not None:
            return x.to(dtype=dtype or x.dtype, device=device or x.device)
        return x
    return torch.as_tensor(x, dtype=dtype, device=device)


def _dtype_from_prediction(predictions):
    return (
        predictions.dtype
        if isinstance(predictions, torch.Tensor) and predictions.is_floating_point()
        else torch.float32
    )


def torch_matrix_condition(m):
    """Return the matrix condition number over the last two dimensions."""
    s = torch.linalg.svdvals(m)
    return s[..., 0] / s[..., -1]


# Backwards-compatible name used by some configs/imports.
tf_matrix_condition = torch_matrix_condition


def get_jac_and_cov_matrix(
    predictions, n_params, n_same, off_sets, n_output=None, summary_writer=None, training=False, strategy=None
):
    """Calculate fiducial sample covariances and finite-difference Jacobians.

    The input ordering is ``fiducial, param0_minus, param0_plus, param1_minus, ...`` along the first axis.  All
    operations are PyTorch/autograd-compatible.  Distributed TensorFlow/Horovod strategies from the legacy
    implementation are intentionally unsupported in this PyTorch utility.
    """
    if strategy is not None:
        raise NotImplementedError("Distributed TensorFlow/Horovod strategies are not supported by the PyTorch loss")

    dtype = _dtype_from_prediction(predictions)
    predictions = _as_tensor(predictions, dtype=dtype)
    if n_output is None:
        n_output = int(predictions.shape[-1])

    splits = [
        split.reshape(-1, n_same, n_output) for split in torch.tensor_split(predictions, 2 * n_params + 1, dim=0)
    ]

    fiducial = splits[0]
    centered = fiducial - fiducial.mean(dim=1, keepdim=True)
    cov = torch.matmul(centered.transpose(-2, -1), centered) / (float(n_same) - 1.0)

    off_sets = _as_tensor(off_sets, dtype=predictions.dtype, device=predictions.device)
    derivatives = []
    for i in range(n_params):
        mean_minus = splits[2 * (i + 1) - 1].mean(dim=1)
        mean_plus = splits[2 * (i + 1)].mean(dim=1)
        derivatives.append((mean_plus - mean_minus) / (2.0 * off_sets[i]))
    jacobian = torch.stack(derivatives, dim=-1)
    return cov, jacobian


def _log_abs_det(m, eps):
    sign, logabsdet = torch.linalg.slogdet(m)
    fallback = torch.log(torch.abs(torch.linalg.det(m)) + eps)
    return torch.where(sign != 0, logabsdet, fallback)


def delta_loss(
    predictions,
    n_params,
    n_same,
    off_sets,
    force_params_value=0.0,
    force_params_weight=1.0,
    jac_weight=100.0,
    cov_loss=False,
    jac_cond_weight=None,
    n_output=None,
    n_partial=None,
    weights=None,
    no_correlations=False,
    use_log_det=True,
    tikhonov_regu=False,
    eps=1e-32,
    summary_writer=None,
    training=True,
    img_summary=False,
    print_scalar=False,
    summary_suffix="",
    strategy=None,
):
    """PyTorch/autograd-compatible delta loss.

    Summary-writing arguments are accepted for config compatibility and ignored.
    """
    cov, jacobian = get_jac_and_cov_matrix(
        predictions, n_params, n_same, off_sets, n_output=n_output, strategy=strategy
    )
    predictions = _as_tensor(predictions, dtype=cov.dtype, device=cov.device)
    if n_output is None:
        n_output = int(predictions.shape[-1])

    if no_correlations and n_output != n_params:
        raise ValueError("Independent summaries (no_correlations) is only possible if n_output == n_params")
    if no_correlations and n_partial is not None:
        raise ValueError("Independent summaries (no_correlations) is only possible if n_partial is None")

    if use_log_det:
        if no_correlations:
            cov_det_loss = torch.log(torch.diagonal(cov, dim1=-2, dim2=-1) + eps) - torch.log(
                torch.square(torch.diagonal(jacobian, dim1=-2, dim2=-1)) + eps
            )
            cov_det_loss = cov_det_loss.mean(dim=-1)
        elif n_partial is None:
            if tikhonov_regu:
                eye = torch.eye(n_params, dtype=cov.dtype, device=cov.device).unsqueeze(0) * eps
                jac_log_det = torch.linalg.slogdet(torch.matmul(jacobian.transpose(-2, -1), jacobian) + eye).logabsdet
                cov_log_det = torch.linalg.slogdet(cov + eye).logabsdet
                cov_det_loss = cov_log_det - jac_log_det
            else:
                cov_det_loss = _log_abs_det(cov, eps) - 2.0 * _log_abs_det(jacobian, eps)
        else:
            j_part = jacobian[:, :, :n_partial]
            jt_cov_j = torch.matmul(j_part.transpose(-2, -1), torch.matmul(cov, j_part))
            jt_j = torch.matmul(j_part.transpose(-2, -1), j_part)
            if tikhonov_regu:
                eye = torch.eye(min(n_params, n_partial), dtype=cov.dtype, device=cov.device).unsqueeze(0) * eps
                cov_log_det = torch.linalg.slogdet(jt_cov_j + eye).logabsdet
                jac_log_det = torch.linalg.slogdet(jt_j + eye).logabsdet
            else:
                cov_log_det = _log_abs_det(jt_cov_j, eps)
                jac_log_det = _log_abs_det(jt_j, eps)
            cov_det_loss = cov_log_det - 2.0 * jac_log_det
    else:
        cov_det_loss = torch.linalg.det(cov)

    if weights is not None:
        weights = _as_tensor(weights, dtype=cov.dtype, device=cov.device)
        cov_det_loss = cov_det_loss * (weights / weights.sum())
    loss = cov_det_loss.mean()

    if jac_weight is not None:
        if cov_loss:
            eye = torch.eye(n_output, dtype=cov.dtype, device=cov.device).unsqueeze(0)
            jac_loss = 0.5 * (
                ((cov - eye) ** 2).mean(dim=(-2, -1)) + ((torch.linalg.inv(cov) - eye) ** 2).mean(dim=(-2, -1))
            )
        else:
            eye = torch.eye(n_output, n_params, dtype=cov.dtype, device=cov.device).unsqueeze(0)
            diff = jacobian - eye
            if n_partial is not None:
                diff = diff[:, :, :n_partial]
            jac_loss = (diff**2).mean(dim=(-2, -1))
        if weights is not None:
            jac_loss = jac_loss * (weights / weights.sum())
        loss = loss + float(jac_weight) * jac_loss.mean()

    if jac_cond_weight is not None:
        c = torch_matrix_condition(jacobian[..., :n_partial] if n_partial is not None else jacobian)
        if weights is not None:
            c = c * (weights / weights.sum())
        loss = loss + float(jac_cond_weight) * c.mean()

    if force_params_value is not None and force_params_weight is not None:
        mid_params = torch.tensor_split(predictions, 2 * n_params + 1, dim=0)[0].reshape(-1, n_same, n_output)
        target = _as_tensor(force_params_value, dtype=predictions.dtype, device=predictions.device)
        diff_loss = (mid_params - target).mean(dim=1) ** 2
        if weights is not None:
            diff_loss = diff_loss.mean(dim=1) * (weights / weights.sum())
        loss = loss + float(force_params_weight) * diff_loss.mean()

    return loss
