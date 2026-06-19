import jax
import jax.numpy as jnp
import equinox as eqx

from ait import AITNeuralODE, NeuralODE
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

    def __call__(self, t, x):  # x: (dim,)
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

    def __call__(self, t, x):  # x: (dim,)
        if self.time_dependent:
            x = jnp.concatenate([x, jnp.reshape(t, (1,)).astype(x.dtype)])
        return jax.nn.softplus(self.mlp(x))[0] + self.h_min


class VecODEModel(eqx.Module):
    ode: AITNeuralODE | NeuralODE
    head: eqx.nn.Linear

    def __init__(
        self,
        key,
        dim,
        model="ait",
        width=64,
        t_max=1.0,
        time_dependent=True,
        dense=False,
        save_interval=0.1,
    ):
        k = jax.random.split(key, 3)
        f = MLPField(k[0], dim, width, time_dependent=time_dependent)
        if model == "ait":
            self.ode = AITNeuralODE(
                f,
                MLPHaltUnit(k[1], dim, width, time_dependent=time_dependent),
                t_max=t_max,
                dense=dense,
                save_interval=save_interval,
            )
        else:
            self.ode = NeuralODE(
                f, T=t_max, dense=dense, save_interval=save_interval
            )
        self.head = eqx.nn.Linear(dim, 1, key=k[2])

    def __call__(self, x):  # x: (B, dim)
        x_out, T, steps = self.ode(x)
        out = jax.vmap(self.head)(x_out)  # (B, 1)
        return out, T, steps
