"""PyTorch conditional Gaussian mixture density model."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal, Normal, Independent
from deep_lss.nets import deepsphere_torch as dst


class GaussianMixtureModel(nn.Module):
    def __init__(
        self,
        dim_theta,
        dim_summary,
        num_components,
        num_hidden_layers=2,
        num_hidden_units=128,
        activation="relu",
        full_covariance=True,
        diagonal_eps=1e-5,
    ):
        super().__init__()
        self.dim_theta = dim_theta
        self.dim_summary = dim_summary
        self.num_components = num_components
        self.full_covariance = full_covariance
        self.diagonal_eps = diagonal_eps
        self.mixture_logits_net = self._build_network(num_components, num_hidden_layers, num_hidden_units, activation)
        self.loc_net = self._build_network(num_components * dim_theta, num_hidden_layers, num_hidden_units, activation)
        out = num_components * ((dim_theta * (dim_theta + 1)) // 2 if full_covariance else dim_theta)
        self.scale_net = self._build_network(out, num_hidden_layers, num_hidden_units, activation)

    def _build_network(self, output_size, n_layers, n_units, activation):
        layers = []
        for _ in range(n_layers):
            layers += [nn.LazyLinear(n_units), dst.LambdaLayer(dst.torch_activation(activation) or (lambda x: x))]
        layers.append(nn.LazyLinear(output_size))
        return nn.Sequential(*layers)

    def log_prob(self, theta, summary):
        logits = self.mixture_logits_net(summary.float())
        loc = self.loc_net(summary.float()).reshape(-1, self.num_components, self.dim_theta)
        raw_scale = self.scale_net(summary.float())
        if self.full_covariance:
            tril = torch.zeros(
                theta.shape[0],
                self.num_components,
                self.dim_theta,
                self.dim_theta,
                device=theta.device,
                dtype=theta.dtype,
            )
            idx = torch.tril_indices(self.dim_theta, self.dim_theta, device=theta.device)
            tril[:, :, idx[0], idx[1]] = raw_scale.reshape(theta.shape[0], self.num_components, -1)
            diag = torch.nn.functional.softplus(torch.diagonal(tril, dim1=-2, dim2=-1)) + self.diagonal_eps
            tril = tril.diagonal_scatter(diag, dim1=-2, dim2=-1)
            comp = MultivariateNormal(loc, scale_tril=tril)
        else:
            scale = (
                torch.nn.functional.softplus(raw_scale.reshape(-1, self.num_components, self.dim_theta))
                + self.diagonal_eps
            )
            comp = Independent(Normal(loc, scale), 1)
        return MixtureSameFamily(Categorical(logits=logits), comp).log_prob(theta.float())

    def mean(self, summary):
        logits = self.mixture_logits_net(summary.float())
        loc = self.loc_net(summary.float()).reshape(-1, self.num_components, self.dim_theta)
        return torch.sum(torch.softmax(logits, dim=-1).unsqueeze(-1) * loc, dim=1)
