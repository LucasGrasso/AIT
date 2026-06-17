import jax
import jax.numpy as jnp
import equinox as eqx
from ait.odefn.halting_unit import HaltingUnit


class ConvHaltUnit(HaltingUnit):
    conv: eqx.nn.Conv2d
    lin: eqx.nn.Linear

    def __init__(self, key, channels=1, hidden=8, initial_bias=1.0, h_min=1.0):
        kc, kl = jax.random.split(key, 2)
        self.conv = eqx.nn.Conv2d(channels, hidden, 3, padding=1, key=kc)
        lin = eqx.nn.Linear(hidden, 1, key=kl)
        # init bias
        lin = eqx.tree_at(lambda l: l.bias, lin, jnp.array([initial_bias]))
        self.lin = lin
        super().__init__(h_min)

    def __call__(self, x):
        feat = jax.nn.relu(self.conv(x))
        pooled = jnp.mean(feat, axis=(1, 2))
        return jax.nn.softplus(self.lin(pooled))[0] + self.h_min
