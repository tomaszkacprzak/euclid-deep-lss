# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics
"""PyTorch mutual-information lower-bound utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_units=128, num_layers=2, dropout_rate=0.0):
        super().__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_features, num_hidden_units), nn.ReLU()]
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            in_features = num_hidden_units
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class JensenShannonCritic(nn.Module):
    def __init__(self, summary_net, theta_net, critic_net):
        super().__init__()
        self.summary_net = summary_net
        self.theta_net = theta_net
        self.critic_net = critic_net

    def forward(self, inputs):
        x, theta = inputs
        return self.critic_net(torch.cat([self.summary_net(x), self.theta_net(theta)], dim=-1))


def _call_critic(critic, x, theta, training=True):
    if isinstance(critic, nn.Module):
        critic.train(training)
    try:
        return critic([x, theta], training=training)
    except TypeError:
        return critic([x, theta])


def jensen_shannon_divergence(critic, x, theta, m_inner_loop=16, training=True):
    batch_size = x.shape[0]
    term_1 = F.softplus(-_call_critic(critic, x, theta, training=training)).mean()
    term_2 = x.new_tensor(0.0)
    for _ in range(m_inner_loop):
        permuted_theta = theta[torch.randperm(batch_size, device=theta.device)]
        term_2 = term_2 + F.softplus(_call_critic(critic, x, permuted_theta, training=training)).mean()
    return term_1 + term_2 / float(m_inner_loop)


def get_jensen_shannon_critic_from_net(
    summary_net, dim_x, dim_theta, dropout_rate=0.0, num_hidden_units=128, num_layers=2
):
    del dim_x
    theta_net = MultiLayerPerceptron(dim_theta, dim_theta, num_hidden_units, num_layers, dropout_rate)
    critic_net = MultiLayerPerceptron(dim_theta + dim_theta, 1, num_hidden_units, num_layers, dropout_rate)
    return JensenShannonCritic(summary_net, theta_net, critic_net)


def get_jensen_shannon_critic_from_pred(dim_summary, dim_theta, dropout_rate=0.0, num_hidden_units=128, num_layers=2):
    theta_net = MultiLayerPerceptron(dim_theta, dim_theta, num_hidden_units, num_layers, dropout_rate)
    critic_net = MultiLayerPerceptron(dim_summary + dim_theta, 1, num_hidden_units, num_layers, dropout_rate)
    return lambda inputs, training=True: critic_net(torch.cat([inputs[0], theta_net(inputs[1])], dim=-1))


def safe_norm(x, axis=None, eps=1e-12):
    return torch.sqrt(torch.sum(torch.square(x), dim=axis) + eps)


def h_tilde(a, b, eps=1e-12):
    diff_ij = safe_norm(a[:, None] - b[None, :], axis=-1, eps=eps)
    return diff_ij - diff_ij.mean(dim=1, keepdim=True) - diff_ij.mean(dim=0, keepdim=True) + diff_ij.mean()


def distance_correlation(summary, theta, training=True, eps=1e-12):
    del training
    h_theta = h_tilde(theta, theta, eps=eps)
    h_summary = h_tilde(summary, summary, eps=eps)
    numerator = torch.mean(h_theta * h_summary)
    denominator = torch.sqrt(torch.mean(torch.square(h_theta)) + eps) * torch.sqrt(
        torch.mean(torch.square(h_summary)) + eps
    )
    return -(numerator / (denominator + eps))


def get_variational_model_from_net(*args, **kwargs):
    raise NotImplementedError("Variational density-estimator model construction must be provided as a PyTorch module.")


def get_variational_model_from_summary(*args, **kwargs):
    raise NotImplementedError("Variational density-estimator model construction must be provided as a PyTorch module.")
