import jax
import equinox as eqx

from ait import NeuralODE, AITNeuralODE
from .conv_field import ConvField
from .conv_halting_unit import ConvHaltUnit


class ImgODEClassifier(eqx.Module):
    ode: AITNeuralODE | NeuralODE
    head: eqx.nn.Linear

    def __init__(self, key, model="ait", channels=1, nf=64, hw=28, t_max=1.0, tol=1e-3, time_dependent=True):
        k = jax.random.split(key, 3)
        f = ConvField(k[0], channels, nf, time_dependent=time_dependent)
        if model == "ait":
            self.ode = AITNeuralODE(
                f,
                ConvHaltUnit(k[1], channels, time_dependent=time_dependent),
                t_max=t_max,
                tol=tol,
            )
        else:
            self.ode = NeuralODE(f, T=t_max, tol=tol)
        self.head = eqx.nn.Linear(channels * hw * hw, 10, key=k[2])

    def __call__(self, imgs):  # (B,1,28,28)
        x_out, T, steps = self.ode(imgs)
        logits = jax.vmap(lambda z: self.head(z.reshape(-1)))(x_out)
        return logits, T, steps
