import equinox as eqx

from .odefn import ODEFn


class HaltingUnit(ODEFn):
    h_min: float = eqx.field(static=True, default=1.0)

    def __init__(self, h_min: float = 1.0):
        self.h_min = h_min
