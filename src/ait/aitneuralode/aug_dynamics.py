from torch import nn


class AugDynamics(nn.Module):
    """dz/dt para z = [x(d) ; A(1) ; x̄(d)]."""

    def __init__(self, f, h):
        super().__init__()
        self.f, self.h = f, h

    def forward(self, t, state):
        x, _, _ = state
        hx = self.h(x).reshape(())
        return self.f(x), hx, hx * x
