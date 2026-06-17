# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""PyTorch mutual-information losses and critic/density-estimator helpers."""

import torch
from torch import nn
import torch.nn.functional as F

from deep_lss.nets.mlp import MultiLayerPerceptron
from deep_lss.nets.gaussian_mixture import GaussianMixtureModel
from deep_lss.nets.normalizing_flow import NormalizingFlowModel


class JensenShannonCriticFromNet(nn.Module):
    """Critic that combines a summary network, theta encoder, and scalar critic head."""

    def __init__(self, summary_net, dim_theta, dropout_rate=0.0, num_hidden_units=128, num_layers=2):
        super().__init__()
        self.summary_net = summary_net
        self.theta_net = MultiLayerPerceptron(
            output_size=dim_theta, num_hidden_units=num_hidden_units, num_layers=num_layers, dropout_rate=dropout_rate
        )
        self.critic_net = MultiLayerPerceptron(
            output_size=1, num_hidden_units=num_hidden_units, num_layers=num_layers, dropout_rate=dropout_rate
        )

    def forward(self, inputs, training: bool | None = None):
        x, theta = inputs
        out_summary = self.summary_net(x)
        out_theta = self.theta_net(theta)
        return self.critic_net(torch.cat([out_summary, out_theta], dim=-1))


class VariationalModelFromNet(nn.Module):
    """Module returning log p(theta | summary_net(x))."""

    def __init__(self, summary_net, estimator):
        super().__init__()
        self.summary_net = summary_net
        self.estimator = estimator

    def forward(self, inputs, training: bool | None = None):
        x, theta = inputs
        return self.estimator.log_prob(theta, self.summary_net(x))


class VariationalModelFromSummary(nn.Module):
    """Module returning -log p(theta | summary)."""

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def forward(self, inputs, training: bool | None = None):
        summary, theta = inputs
        return -self.estimator.log_prob(theta, summary)


def jensen_shannon_divergence(critic, x, theta, m_inner_loop=16, training=True):
    """Jensen-Shannon-divergence lower-bound loss as a scalar PyTorch tensor."""
    batch_size = x.shape[0]
    term_1 = torch.mean(F.softplus(-critic([x, theta], training=training)))

    term_2 = x.new_zeros(())
    for _ in range(m_inner_loop):
        permuted_indices = torch.randperm(batch_size, device=x.device)
        permuted_theta = theta.index_select(0, permuted_indices)
        term_2 = term_2 + torch.mean(F.softplus(critic([x, permuted_theta], training=training)))

    term_2 = term_2 / x.new_tensor(float(m_inner_loop))
    return term_1 + term_2


def get_jensen_shannon_critic_from_net(
    summary_net, dim_x, dim_theta, dropout_rate=0.0, num_hidden_units=128, num_layers=2
):
    return JensenShannonCriticFromNet(
        summary_net=summary_net,
        dim_theta=dim_theta,
        dropout_rate=dropout_rate,
        num_hidden_units=num_hidden_units,
        num_layers=num_layers,
    )


def get_jensen_shannon_critic_from_pred(dim_summary, dim_theta, dropout_rate=0.0, num_hidden_units=128, num_layers=2):
    raise NotImplementedError("This function can't be implemented. The critic network must be passed as an argument.")


def safe_norm(x, axis=None, eps=1e-12):
    return torch.sqrt(torch.sum(torch.square(x), dim=axis) + x.new_tensor(eps))


def h_tilde(a, b, eps=1e-12):
    diff_ij = safe_norm(a[:, None] - b[None, :], axis=-1, eps=eps)
    term1 = diff_ij
    term2 = torch.mean(diff_ij, dim=1, keepdim=True)
    term3 = torch.mean(diff_ij, dim=0, keepdim=True)
    term4 = torch.mean(diff_ij)
    return term1 - term2 - term3 + term4


def distance_correlation(summary, theta, training=True, eps=1e-12):
    h_theta = h_tilde(theta, theta, eps=eps)
    h_summary = h_tilde(summary, summary, eps=eps)
    numerator = torch.mean(h_theta * h_summary)
    sum_h_theta_squared = torch.mean(torch.square(h_theta))
    sum_h_summary_squared = torch.mean(torch.square(h_summary))
    denominator = torch.sqrt(sum_h_theta_squared + summary.new_tensor(eps)) * torch.sqrt(
        sum_h_summary_squared + summary.new_tensor(eps)
    )
    return -(numerator / denominator.clamp_min(summary.new_tensor(eps)))


def get_variational_model_from_net(
    summary_net,
    dim_x,
    dim_summary,
    dim_theta,
    num_components=4,
    num_hidden_layers=2,
    num_hidden_units=128,
    activation="relu",
):
    estimator = GaussianMixtureModel(
        dim_theta=dim_theta,
        dim_summary=dim_summary,
        num_components=num_components,
        num_hidden_layers=num_hidden_layers,
        num_hidden_units=num_hidden_units,
        activation=activation,
    )
    return VariationalModelFromNet(summary_net, estimator)


def get_variational_model_from_summary(
    dim_summary,
    dim_theta,
    density_estimator="gmm",
    num_components=4,
    full_covariance=True,
    num_hidden_layers=2,
    num_hidden_units=128,
    activation="relu",
    num_layers=4,
    scale_eps=1e-5,
    log_scale_clip=5.0,
):
    if density_estimator == "gmm":
        estimator = GaussianMixtureModel(
            dim_theta=dim_theta,
            dim_summary=dim_summary,
            num_components=num_components,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            activation=activation,
            full_covariance=full_covariance,
        )
    elif density_estimator == "flow":
        estimator = NormalizingFlowModel(
            dim_theta=dim_theta,
            dim_summary=dim_summary,
            num_layers=num_layers,
            num_hidden_units=num_hidden_units,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            scale_eps=scale_eps,
            log_scale_clip=log_scale_clip,
        )
    else:
        raise ValueError(f"Unknown density_estimator '{density_estimator}', choose 'gmm' or 'flow'")

    return VariationalModelFromSummary(estimator)
