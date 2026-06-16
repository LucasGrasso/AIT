import torch, torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint, odeint_event
from aug_dynamics import AugDynamics


class AITNeuralODE(nn.Module):
    def __init__(
        self,
        f,
        h,
        d,
        t_max=10.0,
        eps=1e-3,
        method="dopri5",
        atol=1e-6,
        rtol=1e-6,
        adjoint=True,
    ):
        super().__init__()
        self.dyn = AugDynamics(f, h, d)
        self.d, self.t_max, self.eps = d, t_max, eps
        self.kw = dict(method=method, atol=atol, rtol=rtol)
        self.interface = odeint_adjoint if adjoint else odeint

    def _event_fn(self, t, z):
        A = z[self.d]
        return torch.stack(
            [(1.0 - self.eps) - A, self.t_max - t]  # natural halt, Tmax halt
        )

    def forward(self, x):  # x: (B, d)
        B, d = x.shape
        dev, dt = x.device, x.dtype

        def solve_single(x0):
            z0 = torch.cat([x0, x0.new_zeros(1), x0.new_zeros(d)], dim=-1)
            t0 = torch.zeros((), device=dev, dtype=dt)
            t_star, traj = odeint_event(
                self.dyn,
                z0,
                t0,
                event_fn=self._event_fn,
                odeint_interface=self.interface,
                reverse_time=False,
                **self.kw,
            )
            zT = traj[-1]  # ← último estado de la trayectoria
            xT, AT, xbar = zT[:d], zT[d], zT[d + 1 :]
            x_hat = xbar + (1.0 - AT) * xT
            return x_hat, t_star

        outs = [solve_single(x[i]) for i in range(B)]
        x_hats, t_stars = zip(*outs)

        return torch.stack(x_hats), torch.stack(t_stars)
