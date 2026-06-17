"""PyTorch conditional RealNVP-style affine coupling flow."""

import math
import torch
from torch import nn
from deep_lss.nets.torch_utils import LinearActivation


class NormalizingFlowModel(nn.Module):
    def __init__(self, dim_theta, dim_summary, num_layers=4, num_hidden_units=64, num_hidden_layers=2,
                 activation="relu", scale_eps=1e-5, log_scale_clip=5.0):
        super().__init__()
        self.dim_theta = dim_theta
        self.dim_summary = dim_summary
        self.num_layers = num_layers
        self.scale_eps = scale_eps
        self.log_scale_clip = log_scale_clip
        self._d = dim_theta // 2
        nets = []
        for i in range(num_layers):
            in_size = (dim_theta - self._d if i % 2 == 0 else self._d) + dim_summary
            out_size = self._d if i % 2 == 0 else dim_theta - self._d
            nets.append(self._build_coupling_net(in_size, out_size, num_hidden_units, num_hidden_layers, activation))
        self.coupling_nets = nn.ModuleList(nets)

    def _build_coupling_net(self, in_size, out_size, num_hidden_units, num_hidden_layers, activation):
        layers = []
        last = in_size
        for _ in range(num_hidden_layers):
            layers.append(LinearActivation(num_hidden_units, activation=activation, in_features=last))
            last = num_hidden_units
        layers.append(nn.Linear(last, 2 * out_size))
        return nn.Sequential(*layers)

    def _shift_log_scale(self, net, context, out_size):
        out = net(context.float())
        return out[:, :out_size], torch.clamp(out[:, out_size:], -self.log_scale_clip, self.log_scale_clip)

    def log_prob(self, theta, summary):
        theta = theta.float(); summary = summary.float()
        z = theta
        log_det_J = torch.zeros(theta.shape[0], dtype=torch.float32, device=theta.device)
        d = self._d
        for i, net in enumerate(self.coupling_nets):
            if i % 2 == 0:
                z_pass, z_transform = z[:, d:], z[:, :d]
                shift, log_scale = self._shift_log_scale(net, torch.cat([z_pass, summary], dim=-1), d)
                scale = torch.exp(log_scale) + self.scale_eps
                z = torch.cat([(z_transform - shift) / scale, z_pass], dim=-1)
            else:
                z_pass, z_transform = z[:, :d], z[:, d:]
                shift, log_scale = self._shift_log_scale(net, torch.cat([z_pass, summary], dim=-1), self.dim_theta - d)
                scale = torch.exp(log_scale) + self.scale_eps
                z = torch.cat([z_pass, (z_transform - shift) / scale], dim=-1)
            log_det_J += -torch.log(scale).sum(dim=-1)
        log_p_base = -0.5 * (self.dim_theta * math.log(2.0 * math.pi) + torch.square(z).sum(dim=-1))
        return log_p_base + log_det_J

    def inverse(self, z, summary):
        z = z.float(); summary = summary.float()
        theta = z
        d = self._d
        for i in reversed(range(self.num_layers)):
            net = self.coupling_nets[i]
            if i % 2 == 0:
                theta_up, z_low = theta[:, d:], theta[:, :d]
                shift, log_scale = self._shift_log_scale(net, torch.cat([theta_up, summary], dim=-1), d)
                theta = torch.cat([z_low * (torch.exp(log_scale) + self.scale_eps) + shift, theta_up], dim=-1)
            else:
                theta_low, z_up = theta[:, :d], theta[:, d:]
                shift, log_scale = self._shift_log_scale(net, torch.cat([theta_low, summary], dim=-1), self.dim_theta - d)
                theta = torch.cat([theta_low, z_up * (torch.exp(log_scale) + self.scale_eps) + shift], dim=-1)
        return theta

    def mean(self, summary, n_samples=256):
        summary = summary.float(); batch_size = summary.shape[0]
        z = torch.randn(batch_size, n_samples, self.dim_theta, dtype=torch.float32, device=summary.device)
        summary_tiled = summary[:, None, :].repeat(1, n_samples, 1)
        theta = self.inverse(z.reshape(-1, self.dim_theta), summary_tiled.reshape(-1, self.dim_summary))
        return theta.reshape(batch_size, n_samples, self.dim_theta).mean(dim=1)
