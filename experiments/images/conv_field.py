import jax
import equinox as eqx

from ait.odefn import ODEFn


class ConvField(ODEFn):
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    c3: eqx.nn.Conv2d

    def __init__(self, key, channels=1, nf=64):
        k = jax.random.split(key, 3)
        self.c1 = eqx.nn.Conv2d(channels, nf, 1, key=k[0])
        self.c2 = eqx.nn.Conv2d(nf, nf, 3, padding=1, key=k[1])
        self.c3 = eqx.nn.Conv2d(nf, channels, 1, key=k[2])

    def __call__(self, x):
        x = jax.nn.relu(self.c1(x))
        x = jax.nn.relu(self.c2(x))
        return self.c3(x)
