"""
uv run python -m experiments.mnist.mnist --model ait  --epochs 15 --lam 0.0001 --runs 3
uv run python -m experiments.mnist.mnist --model node --epochs 15 --lam 0.0    --runs 3
"""

import os

import jax
import jax.numpy as jnp
import equinox as eqx

from ait import AITNeuralODE, NeuralODE
from ..training import (
    train_sweep,
    save_csv,
    base_parser,
    config_from_args,
    ce_loss,
    accuracy,
)
from ..logger import get_logger

from .data import get_loaders


class ConvField(eqx.Module):
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


class HaltUnit(eqx.Module):
    conv: eqx.nn.Conv2d
    lin: eqx.nn.Linear
    hmin: float = eqx.field(static=True)

    def __init__(self, key, channels=1, hidden=8, hmin=1e-3, bias_init=1.0):
        kc, kl = jax.random.split(key, 2)
        self.conv = eqx.nn.Conv2d(channels, hidden, 3, padding=1, key=kc)
        lin = eqx.nn.Linear(hidden, 1, key=kl)
        # init bias
        lin = eqx.tree_at(lambda l: l.bias, lin, jnp.array([bias_init]))
        self.lin = lin
        self.hmin = hmin

    def __call__(self, x):
        feat = jax.nn.relu(self.conv(x))
        pooled = jnp.mean(feat, axis=(1, 2))
        return jax.nn.softplus(self.lin(pooled))[0] + self.hmin


class ODEClassifier(eqx.Module):
    ode: AITNeuralODE | NeuralODE
    head: eqx.nn.Linear

    def __init__(
        self, key, model="ait", channels=1, nf=64, hw=28, t_max=1.0, eps=1e-3, tol=1e-3
    ):
        k = jax.random.split(key, 3)
        f = ConvField(k[0], channels, nf)
        if model == "ait":
            self.ode = AITNeuralODE(
                f, HaltUnit(k[1], channels), t_max=t_max, eps=eps, tol=tol
            )
        else:
            self.ode = NeuralODE(f, T=t_max, tol=tol)
        self.head = eqx.nn.Linear(channels * hw * hw, 10, key=k[2])

    def __call__(self, imgs):  # (B,1,28,28)
        x_out, T = self.ode(imgs)
        logits = jax.vmap(lambda z: self.head(z.reshape(-1)))(x_out)
        return logits, T


def main():
    p = base_parser("MNIST AIT-NODE / NODE")
    p.add_argument("--nf", type=int, default=64)
    args = p.parse_args()

    logger = get_logger("mnist")
    logger.info(f"jax devices: {jax.devices()} | model={args.model} | lam={args.lam}")

    def model_factory(key):
        return ODEClassifier(key, model=args.model, nf=args.nf, t_max=args.t_max)

    def loaders_factory(seed):
        return get_loaders(args.batch_size, args.subset, seed=seed)

    rows = train_sweep(
        model_factory,
        loaders_factory,
        config_from_args(args),
        task_loss_fn=ce_loss,
        score_fn=accuracy,
        logger=logger,
    )

    lam_str = f"{args.lam:.10f}".rstrip("0").rstrip(".")
    out_path = os.path.join("results", f"{args.model}_mnist_{lam_str}.csv")
    save_csv(rows, out_path, logger)


if __name__ == "__main__":
    main()
