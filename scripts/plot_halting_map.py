"""AIT halting-time / solver-step maps over the input plane, across lambdas.

Works for any 2D task: a task supplies how to rebuild its checkpoints, what to
scatter on top, and how far out the interesting region goes. Register a new one
in TASKS below.

    uv run --no-sync python -m scripts.plot_halting_map annuli2d
    uv run --no-sync python -m scripts.plot_halting_map cnf_8gaussians
"""

import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiments.training import load_model


@dataclass(frozen=True)
class Task2D:
    """How to render halting maps for one family of 2D checkpoints.

    build    -- (**hyperparams) -> model, matching `load_model`'s contract
    samples  -- () -> (points (N, 2), colours (N,) or None) drawn over each map
    extent   -- half-width of the square grid to evaluate on
    contour  -- draw the model's zero level set; only meaningful where the
                first output is a decision function, not e.g. a log-density
    """

    build: Callable[..., object]
    samples: Callable[[], tuple[np.ndarray, np.ndarray | None]]
    extent: float
    contour: bool


def _annuli_task(_experiment):  # one spec covers annuli1d/2d/...
    from experiments.annuli.model import VecODEModel
    from experiments.annuli.data import gd

    def samples():
        x, y = gd(800, 800, d=2)
        return np.asarray(x), np.asarray(y).ravel()

    return Task2D(
        build=lambda **hp: VecODEModel(jax.random.PRNGKey(0), **hp),
        samples=samples,
        extent=2.0,
        contour=True,  # first output is the classifier margin
    )


def _cnf_task(experiment):
    from experiments.cnf.model import CNFModel
    from experiments.cnf.data import generate

    toy = experiment.split("_", 1)[1] if "_" in experiment else "checkerboard"

    def build(**hp):
        # force the exact trace: the map is a picture, and evaluating it needs
        # no eps, which `model(grid)` has no way to supply.
        return CNFModel(jax.random.PRNGKey(0), **{**hp, "hutchinson": False})

    return Task2D(
        build=build,
        samples=lambda: (generate(toy, 2000, rng=0), None),
        extent=4.0,
        contour=False,  # first output is a log-density; no zero level set
    )


TASKS = {"annuli": _annuli_task, "cnf": _cnf_task}


def resolve_task(experiment, name=None):
    """`--task` if given, else the leading word of the experiment tag."""
    if name is None:
        name = next((k for k in TASKS if experiment.startswith(k)), None)
    if name not in TASKS:
        raise SystemExit(
            f"cannot resolve a task for {experiment!r}; pass --task "
            f"{{{','.join(TASKS)}}}"
        )
    return TASKS[name](experiment)


def lam_of(path, experiment):
    name = os.path.splitext(os.path.basename(path))[0]  # ait_<exp>_<lam>
    return float(name[len(f"ait_{experiment}_") :])


def eval_grid(model, res, extent):
    g = jnp.linspace(-extent, extent, res)
    xx, yy = jnp.meshgrid(g, g)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    pred, T, steps = model(grid)
    r = lambda a: np.asarray(a).reshape(res, res)
    # T and steps are per-sample for AIT but scalars for a fixed-T NODE
    b = lambda a: np.broadcast_to(np.asarray(a), (res * res,))
    return np.asarray(xx), np.asarray(yy), r(pred), r(b(T)), r(b(steps))


def main():
    p = argparse.ArgumentParser(
        description="AIT halting-time / steps maps across lambdas"
    )
    p.add_argument(
        "experiment",
        help="e.g. annuli2d. globs models/ait_<exp>_<lam>.eqx",
    )
    p.add_argument("--task", choices=sorted(TASKS), default=None)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--outdir", default="plots")
    p.add_argument("--res", type=int, default=200)
    p.add_argument("--extent", type=float, default=None, help="default: per task")
    args = p.parse_args()

    task = resolve_task(args.experiment, args.task)
    extent = task.extent if args.extent is None else args.extent

    ckpts = sorted(
        glob.glob(os.path.join(args.models_dir, f"ait_{args.experiment}_*.eqx")),
        key=lambda pth: lam_of(pth, args.experiment),
    )
    if not ckpts:
        raise SystemExit(
            f"no checkpoints ait_{args.experiment}_*.eqx in {args.models_dir}"
        )

    x, colours = task.samples()

    results = []  # (lam, xx, yy, pred, T, steps)
    for ck in ckpts:
        model, hp = load_model(ck, task.build)
        if hp["dim"] != 2:
            raise SystemExit(f"halting map needs dim=2 models, got dim={hp['dim']}")
        xx, yy, pred, T, steps = eval_grid(model, args.res, extent)
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
            if task.contour:
                ax.contour(xx, yy, pred, levels=[0.0], colors="white", linewidths=1.2)
            ax.scatter(
                x[:, 0],
                x[:, 1],
                c="white" if colours is None else colours,
                cmap=None if colours is None else "coolwarm",
                s=3,
                alpha=0.3,
                linewidths=0,
            )
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
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
        ax=axes[:, sep + 1 :].ravel().tolist(),
        location="right",
        label="solver steps",
    )

    out = os.path.join(args.outdir, f"halting_maps-{args.experiment}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
