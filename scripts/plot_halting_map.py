import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiments.training import load_model
from experiments.annuli.model import VecODEModel
from experiments.annuli.data import gd


def build(**hp):
    return VecODEModel(jax.random.PRNGKey(0), **hp)


def main():
    p = argparse.ArgumentParser(description="AIT halting-time map over the input plane")
    p.add_argument("ckpt")
    p.add_argument("--outdir", default="plots")
    p.add_argument("--res", type=int, default=200)
    p.add_argument("--extent", type=float, default=2.0)
    args = p.parse_args()

    model, hp = load_model(args.ckpt, build)
    if hp["dim"] != 2:
        raise SystemExit(f"halting map needs a dim=2 model, got dim={hp['dim']}")

    g = jnp.linspace(-args.extent, args.extent, args.res)
    xx, yy = jnp.meshgrid(g, g)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    pred, T, steps = model(grid)
    pred = np.asarray(pred).reshape(args.res, args.res)
    T = np.asarray(T).reshape(args.res, args.res)
    steps = np.asarray(steps).reshape(args.res, args.res)

    # reference data for context
    x, y = gd(800, 800, d=2)
    y = y.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fields = [(axes[0], T, "halting time $T^*$"), (axes[1], steps, "solver steps")]
    for ax, field, title in fields:
        pcm = ax.pcolormesh(xx, yy, field, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        ax.contour(xx, yy, pred, levels=[0.0], colors="white", linewidths=1.5)
        ax.scatter(
            x[:, 0], x[:, 1], c=y, cmap="coolwarm", s=4, alpha=0.35, linewidths=0
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    fig.suptitle(f"{hp['model'].upper()}  (annuli2d, white = decision boundary)")
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out = os.path.join(args.outdir, f"halting_map-{name}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
