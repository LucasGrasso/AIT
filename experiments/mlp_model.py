import equinox as eqx
import jax
import jax.numpy as jnp

from ait.odefn import ODEFn, HaltingUnit


class MLPField(ODEFn):
    mlp: eqx.nn.MLP
    time_dependent: bool = eqx.field(static=True)

    def __init__(self, key, dim, width=64, depth=2, time_dependent=True):
        in_dim = dim + (1 if time_dependent else 0)
        self.mlp = eqx.nn.MLP(
            in_dim, dim, width, depth, activation=jax.nn.tanh, key=key
        )
        self.time_dependent = time_dependent

    def __call__(self, t, x, args=None):  # x: (dim,)
        if self.time_dependent:
            x = jnp.concatenate([x, jnp.reshape(t, (1,)).astype(x.dtype)])
        return self.mlp(x)


class MLPHaltUnit(HaltingUnit):
    mlp: eqx.nn.MLP
    time_dependent: bool = eqx.field(static=True)

    def __init__(self, key, dim, width=64, depth=1, h_min=1.0, time_dependent=True):
        in_dim = dim + (1 if time_dependent else 0)
        self.mlp = eqx.nn.MLP(in_dim, 1, width, depth, activation=jax.nn.tanh, key=key)
        self.time_dependent = time_dependent
        super().__init__(h_min)

    def __call__(self, t, x, args=None):  # x: (dim,)
        if self.time_dependent:
            x = jnp.concatenate([x, jnp.reshape(t, (1,)).astype(x.dtype)])
        return jax.nn.softplus(self.mlp(x))[0] + self.h_min
