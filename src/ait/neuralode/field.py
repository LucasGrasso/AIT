from torch import nn


class _Field(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, t, x):
        return self.f(x)
