import argparse
import glob
import math
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


def lam_of(path, experiment):
    name = os.path.splitext(os.path.basename(path))[0]  # ait_<exp>_<lam>
    return float(name[len(f"ait_{experiment}_") :])


def eval_grid(model, res, extent):
    g = jnp.linspace(-extent, extent, res)
    xx, yy = jnp.meshgrid(g, g)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    pred, T, steps = model(grid)
    r = lambda a: np.asarray(a).reshape(res, res)
    return np.asarray(xx), np.asarray(yy), r(pred), r(T), r(steps)


def main():
    p = argparse.ArgumentParser(
        description="AIT halting-time / steps maps across lambdas"
    )
    p.add_argument("experiment", help="e.g. annuli2d; globs models/ait_<exp>_<lam>.eqx")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--outdir", default="plots")
    p.add_argument("--res", type=int, default=200)
    p.add_argument("--extent", type=float, default=2.0)
    args = p.parse_args()

    ckpts = sorted(
        glob.glob(os.path.join(args.models_dir, f"ait_{args.experiment}_*.eqx")),
        key=lambda pth: lam_of(pth, args.experiment),
    )
    if not ckpts:
        raise SystemExit(
            f"no checkpoints ait_{args.experiment}_*.eqx in {args.models_dir}"
        )

    x, y = gd(800, 800, d=2)
    y = y.ravel()

    results = []  # (lam, xx, yy, pred, T, steps)
    for ck in ckpts:
        model, hp = load_model(ck, build)
        if hp["dim"] != 2:
            raise SystemExit(f"halting map needs dim=2 models, got dim={hp['dim']}")
        xx, yy, pred, T, steps = eval_grid(model, args.res, args.extent)
        results.append((lam_of(ck, args.experiment), xx, yy, pred, T, steps))

    n = len(results)
    ncols = 2
    nrows = math.ceil(n / ncols)
    os.makedirs(args.outdir, exist_ok=True)

    tvmin = min(r[4].min() for r in results)
    tvmax = max(r[4].max() for r in results)
    svmin = min(r[5].min() for r in results)
    svmax = max(r[5].max() for r in results)

    sep = ncols  # thin spacer column between the two blocks
    fig, axes = plt.subplots(
        nrows,
        2 * ncols + 1,
        figsize=(4.0 * 2 * ncols + 1.5, 4.0 * nrows),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"width_ratios": [1] * ncols + [0.35] + [1] * ncols},
    )
    for row in range(nrows):
        axes[row][sep].axis("off")
    pcm_t = pcm_s = None
    for i, r in enumerate(results):
        lam, xx, yy, pred, T, steps = r
        row, col = i // ncols, i % ncols
        ax_t, ax_s = axes[row][col], axes[row][sep + 1 + col]
        pcm_t = ax_t.pcolormesh(
            xx, yy, T, shading="auto", cmap="viridis", vmin=tvmin, vmax=tvmax
        )
        pcm_s = ax_s.pcolormesh(
            xx, yy, steps, shading="auto", cmap="viridis", vmin=svmin, vmax=svmax
        )
        for ax in (ax_t, ax_s):
            ax.contour(xx, yy, pred, levels=[0.0], colors="white", linewidths=1.2)
            ax.scatter(
                x[:, 0], x[:, 1], c=y, cmap="coolwarm", s=3, alpha=0.3, linewidths=0
            )
            ax.set_title(f"$\\lambda={lam:g}$")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    for j in range(n, nrows * ncols):  # hide unused cells in both blocks
        row, col = j // ncols, j % ncols
        axes[row][col].axis("off")
        axes[row][sep + 1 + col].axis("off")
    assert pcm_t is not None and pcm_s is not None

    fig.colorbar(
        pcm_t,
        ax=axes[:, :ncols].ravel().tolist(),
        location="right",
        label="halting time $T^*$",
    )
    fig.colorbar(
        pcm_s,
        ax=axes[:, sep + 1:].ravel().tolist(),
        location="right",
        label="solver steps",
    )

    out = os.path.join(args.outdir, f"halting_maps-{args.experiment}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
