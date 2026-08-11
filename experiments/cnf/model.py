import math
from typing import Any, NamedTuple


import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ait import AITNeuralODE, NeuralODE, ODEFn, HaltingUnit, Readout
from ..mlp_model import MLPField, MLPHaltUnit


class CNFArgs(NamedTuple):
    """What the CNF stack passes through diffrax's untyped `args` channel.

    A NamedTuple rather than a bare eps array for two reasons: it is a pytree,
    so `vmap` splits a batched `eps` of shape (B, dim) into a per-sample (dim,)
    for each solve; and `AITNeuralODE` hands the same `args` to the field *and*
    the halting unit, so a named slot says "index into me" instead of letting
    the halting unit mistake an epsilon for its own state.
    """

    eps: Array
    rest: Any = None


def make_args(key, x):
    """eps <- sample_unit_variance(x.shape) for a whole batch x: (B, dim).

    Sampled by the caller, not inside the model: a model that held a key would
    have to carry mutable RNG state through an eqx.Module, and sampling inside
    `CNFField.__call__` would redraw eps at every solver stage -- "outside the
    integral" in Algorithm 1 means diffrax pins it for the whole trajectory.
    """
    return CNFArgs(eps=jax.random.normal(key, x.shape))


class CNFField(ODEFn):
    """f_aug([z, delta_logp], t) from Algorithm 1 of arxiv 1810.01367.

    State is [z, delta_logp] in R^(dim+1), integrated from [x, 0]:

        dz/dt          = f(z(t), t)
        d(delta_logp)  = -Tr(df/dz)

    `f` only sees the data coords, so it stays an ordinary MLPField(dim).

    The Hutchinson branch is the single-sample MC estimate of eq. (7)'s
    Tr(A) = E_p(eps)[eps^T A eps]. One draw is enough because log p is *linear*
    in the integrated trace, so the estimate is unbiased and so is its gradient;
    the variance is averaged down by the B independent draws in the minibatch
    and by resampling every step. Averaging K>1 draws is pure loss at dim=2 --
    K=2 VJPs already costs what jacfwd costs, with nonzero variance.

    With hutchinson=False the trace is exact via jacfwd: 2 JVPs at dim=2 versus
    1 VJP for the estimator, so ~2x the cost for zero variance.
    """

    f: ODEFn
    dim: int = eqx.field(static=True)
    hutchinson: bool = eqx.field(static=True)

    def __init__(self, f, dim, hutchinson=True):
        self.f, self.dim, self.hutchinson = f, dim, hutchinson

    def __call__(self, t, state, args: CNFArgs | None = None):  # state: (dim+1,)
        z = state[: self.dim]
        if self.hutchinson:
            if args is None or not isinstance(args, CNFArgs):
                raise ValueError("CNFField(hutchinson=True) needs `CNFArgs`")
            eps = args[0]
            f_t, vjp = jax.vjp(lambda y: self.f(t, y), z)
            (g,) = vjp(eps)  # g <- eps^T df/dz
            tr = jnp.dot(g, eps)  # Tr~ = matrix_multiply(g, eps)
        else:
            f_t = self.f(t, z)
            tr = jnp.trace(jax.jacfwd(lambda y: self.f(t, y))(z))
        return jnp.concatenate([f_t, jnp.reshape(-tr, (1,))])  # [f_t, -Tr~]


class CNFHaltUnit(HaltingUnit):
    """
    Halts on the data coords only; the log-density coord is bookkeeping, and
    feeding it to the halt MLP would break its input size the same way.
    """

    h: HaltingUnit
    dim: int = eqx.field(static=True)

    def __init__(self, h, dim):
        self.h, self.dim = h, dim
        super().__init__(h.h_min)

    def __call__(self, t, state, args: CNFArgs | None = None):
        return self.h(t, state[: self.dim])


def standard_normal_logpdf(z):  # z: (B, dim) -> (B,)
    return -0.5 * (z.shape[-1] * math.log(2 * math.pi) + jnp.sum(z**2, axis=-1))


class CNFModel(eqx.Module):
    ode: AITNeuralODE | NeuralODE
    dim: int = eqx.field(static=True)
    hutchinson: bool = eqx.field(static=True)

    def __init__(
        self,
        key,
        dim,
        model="ait",
        width=64,
        t_max=1.0,
        hutchinson=True,
        dense=False,
        save_interval=0.1,
    ):
        k1, k2 = jax.random.split(key, 2)
        time_dependent = model == "node"
        f = CNFField(
            MLPField(k1, dim, width, 3, time_dependent=time_dependent),
            dim,
            hutchinson=hutchinson,
        )
        if model == "ait":
            self.ode = AITNeuralODE(
                f,
                CNFHaltUnit(
                    MLPHaltUnit(k2, dim, width, time_dependent=time_dependent), dim
                ),
                t_max=t_max,
                readout=Readout.ENDPOINT,
                dense=dense,
                save_interval=save_interval,
            )
        else:
            self.ode = NeuralODE(f, T=t_max, dense=dense, save_interval=save_interval)
        self.dim = dim
        self.hutchinson = hutchinson

    def __call__(self, x, args: CNFArgs | None = None):  # x: (B, dim)
        state0 = jnp.concatenate([x, jnp.zeros((x.shape[0], 1), x.dtype)], axis=1)
        state_T, T, steps = self.ode(state0, args)
        z, delta_logp = state_T[:, : self.dim], state_T[:, self.dim]
        logp = standard_normal_logpdf(z) - delta_logp  # log p_z0(z) - delta_logp
        if isinstance(self.ode, AITNeuralODE):
            h = lambda a: self.ode.h(0.0, a)
            h0, h1 = jax.vmap(h)(x), jax.vmap(h)(z)
            logp = logp + jnp.log(h0) - jnp.log(h1)
        return logp[:, None], T, steps
