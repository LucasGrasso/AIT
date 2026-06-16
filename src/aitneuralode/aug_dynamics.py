import torch
from torch import nn


class AugDynamics(nn.Module):
    """dz/dt para z = [x(d) ; A(1) ; x̄(d)]."""

    def __init__(self, f, h, d):
        super().__init__()
        self.f, self.h, self.d = f, h, d

    def forward(self, t, z):
        d = self.d
        x = z[..., :d]
        hv = self.h(x)  # > 0
        return torch.cat([self.f(x), hv, hv * x], dim=-1)  # [dx, dA, d\bar{x}]
