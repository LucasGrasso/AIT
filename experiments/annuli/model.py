import jax
import jax.numpy as jnp
import equinox as eqx

from ait import AITNeuralODE, NeuralODE
from ..mlp_model import MLPField, MLPHaltUnit


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
            self.ode = NeuralODE(f, T=t_max, dense=dense, save_interval=save_interval)
        self.head = eqx.nn.Linear(dim, 1, key=k[2])

    def __call__(self, x):  # x: (B, dim)
        x_out, T, steps = self.ode(x)
        out = jax.vmap(self.head)(x_out)  # (B, 1)
        return out, T, steps
