import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .field import _Field


class NeuralODE(nn.Module):
    def __init__(self, f, T=1.0, method="dopri5", atol=1e-6, rtol=1e-6, adjoint=True):
        super().__init__()
        self.dyn = _Field(f)
        self.T = T
        self.method, self.atol, self.rtol = method, atol, rtol
        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, x):
        t = torch.tensor([0.0, self.T], device=x.device, dtype=x.dtype)
        x_T = self.odeint(
            self.dyn, x, t, method=self.method, atol=self.atol, rtol=self.rtol
        )[
            -1
        ]  # (B, d)
        T_star = x.new_full((x.shape[0],), self.T)  # (B,)
        return x_T, T_star
