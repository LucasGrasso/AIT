import jax
import jax.numpy as jnp
import equinox as eqx
from ait.odefn.halting_unit import HaltingUnit


class ConvHaltUnit(HaltingUnit):
    conv: eqx.nn.Conv2d
    lin: eqx.nn.Linear
    time_dependent: bool = eqx.field(static=True)

    def __init__(
        self, key, channels=1, hidden=8, initial_bias=1.0, h_min=1.0, time_dependent=True
    ):
        kc, kl = jax.random.split(key, 2)
        in_ch = channels + (1 if time_dependent else 0)
        self.conv = eqx.nn.Conv2d(in_ch, hidden, 3, padding=1, key=kc)
        lin = eqx.nn.Linear(hidden, 1, key=kl)
        # init bias
        lin = eqx.tree_at(lambda l: l.bias, lin, jnp.array([initial_bias]))
        self.lin = lin
        self.time_dependent = time_dependent
        super().__init__(h_min)

    def __call__(self, t, x, args=None):  # x: (C, H, W)
        if self.time_dependent:
            tc = jnp.full((1, *x.shape[1:]), t, dtype=x.dtype)
            x = jnp.concatenate([x, tc], axis=0)
        feat = jax.nn.softplus(self.conv(x))
        pooled = jnp.mean(feat, axis=(1, 2))
        return jax.nn.softplus(self.lin(pooled))[0] + self.h_min
