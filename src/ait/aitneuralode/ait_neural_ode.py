import torch, torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint, odeint_event
from .aug_dynamics import AugDynamics


class AITNeuralODE(nn.Module):
    def __init__(
        self,
        f,
        h,
        t_max=10.0,
        eps=1e-3,
        method="dopri5",
        atol=1e-6,
        rtol=1e-6,
        adjoint=True,
    ):
        super().__init__()
        self.dyn = AugDynamics(f, h)
        self.t_max, self.eps = t_max, eps
        self.kw = dict(method=method, atol=atol, rtol=rtol)
        self.interface = odeint_adjoint if adjoint else odeint

    def _event_fn(self, t, z):
        _, A, _ = z
        return torch.stack(
            [(1.0 - self.eps) - A, self.t_max - t]  # natural halt, Tmax halt
        )

    def forward(self, x):  # x: (B, d)
        B = x.shape[0]
        outs = [self._solve_single(x[i]) for i in range(B)]
        x_hats, t_stars = zip(*outs)

        return torch.stack(x_hats), torch.stack(t_stars)

    def _solve_single(self, x0):
        z0 = (x0, x0.new_zeros(()), torch.zeros_like(x0))  # (x, A=0, x̄=0)
        t0 = x0.new_zeros(())
        t_star, traj = odeint_event(
            self.dyn,
            z0,
            t0,
            event_fn=self._event_fn,
            odeint_interface=self.interface,
            reverse_time=False,
            **self.kw,
        )
        xT, AT, xbarT = (
            traj[0][-1],
            traj[1][-1],
            traj[2][-1],
        )
        return xbarT + (1.0 - AT) * xT, t_star
