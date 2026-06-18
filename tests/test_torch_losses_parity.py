import importlib.util
import math

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")


def _np_delta_reference(predictions, n_params, n_same, offsets, eps=1e-32):
    n_output = predictions.shape[-1]
    splits = np.split(predictions, 2 * n_params + 1, axis=0)
    splits = [s.reshape(-1, n_same, n_output) for s in splits]
    centered = splits[0] - splits[0].mean(axis=1, keepdims=True)
    cov = np.einsum("hjk,hjl->hkl", centered, centered) / (n_same - 1.0)
    jac = []
    for i in range(n_params):
        jac.append((splits[2 * (i + 1)].mean(axis=1) - splits[2 * (i + 1) - 1].mean(axis=1)) / (2.0 * offsets[i]))
    jac = np.stack(jac, axis=-1)
    loss = np.mean(np.log(np.abs(np.linalg.det(cov)) + eps) - 2.0 * np.log(np.abs(np.linalg.det(jac)) + eps))
    loss += 100.0 * np.mean((jac - np.eye(n_output, n_params)[None, :, :]) ** 2)
    loss += np.mean(splits[0].mean(axis=1) ** 2)
    return cov, jac, loss


def test_delta_loss_matches_numpy_reference():
    import torch
    from deep_lss.utils.delta_loss import delta_loss, get_jac_and_cov_matrix

    rng = np.random.default_rng(4)
    n_params = n_output = 2
    n_same = 5
    predictions = rng.normal(size=(3 * (2 * n_params + 1) * n_same, n_output)) + np.eye(2)[0]
    offsets = np.array([0.2, 0.4])
    ref_cov, ref_jac, ref_loss = _np_delta_reference(predictions, n_params, n_same, offsets)

    pred_t = torch.tensor(predictions, dtype=torch.float64, requires_grad=True)
    cov, jac = get_jac_and_cov_matrix(pred_t, n_params, n_same, offsets)
    loss = delta_loss(pred_t, n_params, n_same, offsets)
    loss.backward()

    np.testing.assert_allclose(cov.detach().numpy(), ref_cov, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jac.detach().numpy(), ref_jac, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(loss.detach().numpy(), ref_loss, rtol=1e-10, atol=1e-10)
    assert pred_t.grad is not None


def test_likelihood_loss_matches_numpy_reference():
    import torch
    from deep_lss.utils.likelihood_loss import neg_likelihood_loss

    predictions = np.array([[0.1, -0.2, 1.0, 0.3, 1.4], [0.3, 0.4, 0.9, -0.1, 1.2]], dtype=np.float64)
    theta = np.array([[0.0, 0.0], [0.5, 0.1]], dtype=np.float64)
    uppers = np.array([[[1.0, 0.3], [0.0, 1.4]], [[0.9, -0.1], [0.0, 1.2]]])
    residual = predictions[:, :2] - theta
    log_det = -np.mean(np.sum(np.log(np.diagonal(uppers, axis1=1, axis2=2) ** 2), axis=1))
    quad = np.mean(np.sum((uppers @ residual[..., None]).squeeze(-1) ** 2, axis=1))

    pred_t = torch.tensor(predictions, dtype=torch.float64, requires_grad=True)
    loss = neg_likelihood_loss(pred_t, torch.tensor(theta, dtype=torch.float64), 2)
    loss.backward()

    np.testing.assert_allclose(loss.detach().numpy(), quad + log_det, rtol=1e-10, atol=1e-10)
    assert pred_t.grad is not None


def test_mutual_info_distance_correlation_matches_numpy_reference():
    import torch
    from deep_lss.utils.mutual_info_loss import distance_correlation

    summary = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    theta = np.array([[0.1], [0.5], [0.9]], dtype=np.float64)

    def h(a, b):
        d = np.sqrt(np.sum((a[:, None] - b[None, :]) ** 2, axis=-1) + 1e-12)
        return d - d.mean(axis=1, keepdims=True) - d.mean(axis=0, keepdims=True) + d.mean()

    h_s = h(summary, summary)
    h_t = h(theta, theta)
    ref = -(h_s * h_t).mean() / (math.sqrt((h_s**2).mean() + 1e-12) * math.sqrt((h_t**2).mean() + 1e-12) + 1e-12)
    summary_t = torch.tensor(summary, dtype=torch.float64, requires_grad=True)
    loss = distance_correlation(summary_t, torch.tensor(theta, dtype=torch.float64))
    loss.backward()
    np.testing.assert_allclose(loss.detach().numpy(), ref, rtol=1e-10, atol=1e-10)
    assert summary_t.grad is not None


def test_torch_loss_helpers_match_numpy_reference():
    import torch
    from deep_lss.utils.torch_losses import compute_vicreg_invariance_loss, compute_vicreg_var_cov_loss

    z = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [1.5, 2.0]], dtype=np.float64)
    zc = z - z.mean(axis=0, keepdims=True)
    std = np.sqrt(np.mean(zc**2, axis=0) + 1e-4)
    ref_var = np.mean((std - 1.0) ** 2)
    cov = zc.T @ zc / (z.shape[0] - 1)
    ref_cov = (np.sum(cov**2) - np.sum(np.diag(cov) ** 2)) / (z.shape[1] ** 2 - z.shape[1])

    z_t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    var_loss, cov_loss = compute_vicreg_var_cov_loss(z_t)
    np.testing.assert_allclose(var_loss.detach().numpy(), ref_var, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cov_loss.detach().numpy(), ref_cov, rtol=1e-10, atol=1e-10)

    pair_ids = np.array([[1, 1], [1, 1], [2, 1], [3, 1]])
    ref_inv = np.mean((z[0] - z[1]) ** 2)
    inv_loss = compute_vicreg_invariance_loss(z_t, torch.tensor(pair_ids))
    np.testing.assert_allclose(inv_loss.detach().numpy(), ref_inv, rtol=1e-10, atol=1e-10)
