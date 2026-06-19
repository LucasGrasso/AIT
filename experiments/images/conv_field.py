import jax
import jax.numpy as jnp
import equinox as eqx

from ait.odefn import ODEFn


class ConvField(ODEFn):
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    c3: eqx.nn.Conv2d
    time_dependent: bool = eqx.field(static=True)

    def __init__(self, key, channels=1, nf=64, time_dependent=True):
        k = jax.random.split(key, 3)
        in_ch = channels + (1 if time_dependent else 0)
        self.c1 = eqx.nn.Conv2d(in_ch, nf, 1, key=k[0])
        self.c2 = eqx.nn.Conv2d(nf, nf, 3, padding=1, key=k[1])
        self.c3 = eqx.nn.Conv2d(nf, channels, 1, key=k[2])
        self.time_dependent = time_dependent

    def __call__(self, t, x):  # x: (C, H, W)
        if self.time_dependent:
            tc = jnp.full((1, *x.shape[1:]), t, dtype=x.dtype)
            x = jnp.concatenate([x, tc], axis=0)
        x = jax.nn.softplus(self.c1(x))
        x = jax.nn.softplus(self.c2(x))
        return self.c3(x)
