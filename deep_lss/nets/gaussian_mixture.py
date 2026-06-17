import torch
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal, Independent, Normal
from deep_lss.nets.torch_utils import LinearActivation


class GaussianMixtureModel(nn.Module):
    def __init__(self, dim_theta, dim_summary, num_components, num_hidden_layers=2, num_hidden_units=128,
                 activation="relu", full_covariance=True, diagonal_eps=1e-5):
        super().__init__()
        self.dim_theta = dim_theta; self.dim_summary = dim_summary; self.num_components = num_components
        self.full_covariance = full_covariance; self.diagonal_eps = diagonal_eps
        self.mixture_logits_net = self._build_network(num_components, num_hidden_layers, num_hidden_units, activation)
        self.loc_net = self._build_network(num_components * dim_theta, num_hidden_layers, num_hidden_units, activation)
        self.tril_size = (dim_theta * (dim_theta + 1)) // 2
        out = num_components * (self.tril_size if full_covariance else dim_theta)
        self.scale_net = self._build_network(out, num_hidden_layers, num_hidden_units, activation)

    def _build_network(self, output_size, num_hidden_layers, num_hidden_units, activation):
        layers = []; last = self.dim_summary
        for _ in range(num_hidden_layers):
            layers.append(LinearActivation(num_hidden_units, activation=activation, in_features=last)); last = num_hidden_units
        layers.append(nn.Linear(last, output_size)); return nn.Sequential(*layers)

    def _params(self, summary):
        summary = summary.float()
        logits = self.mixture_logits_net(summary).float()
        loc = self.loc_net(summary).reshape(-1, self.num_components, self.dim_theta).float()
        if self.full_covariance:
            raw = self.scale_net(summary).reshape(-1, self.num_components, self.tril_size)
            tril = torch.zeros(raw.shape[0], self.num_components, self.dim_theta, self.dim_theta, device=raw.device, dtype=raw.dtype)
            idx = torch.tril_indices(self.dim_theta, self.dim_theta, device=raw.device)
            tril[..., idx[0], idx[1]] = raw
            diag = torch.diagonal(tril, dim1=-2, dim2=-1)
            diag.copy_(torch.nn.functional.softplus(diag) + self.diagonal_eps)
            comp = MultivariateNormal(loc, scale_tril=tril.float())
        else:
            scale = torch.nn.functional.softplus(self.scale_net(summary).reshape(-1, self.num_components, self.dim_theta)) + self.diagonal_eps
            comp = Independent(Normal(loc, scale), 1)
        return logits, loc, comp

    def log_prob(self, theta, summary):
        logits, _, comp = self._params(summary)
        return MixtureSameFamily(Categorical(logits=logits), comp).log_prob(theta.float())

    def mean(self, summary):
        logits, loc, _ = self._params(summary)
        return (torch.softmax(logits, dim=-1)[:, :, None] * loc).sum(dim=1)
