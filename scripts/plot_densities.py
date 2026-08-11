"""Assemble the Data / CNF / AIT-CNF comparison grid.

Reads whatever `results/densities/*.npz` the runs left behind and lays out one
row per toy, in the style of figure 4 of arxiv 1810.01367: square panels, no
axes, viridis, column titles on the top row only. Columns are

    x

so the ponder penalty's effect on the learned density is read left to right.

    uv run --no-sync python -m scripts.plot_densities
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.cnf.data import Toy  # noqa: E402

# canonical row order; toys with no densities on disk are dropped
ROW_ORDER = [Toy.EIGHT_GAUSSIANS, Toy.TWO_SPIRALS, Toy.CHECKERBOARD]


def load_densities(root):
    """{(model, toy, lam): npz} for everything under `root`."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*.npz"))):
        d = np.load(path, allow_pickle=False)
        out[(str(d["model"]), str(d["toy"]), float(d["lam"]))] = d
    return out


def data_histogram(samples, extent, bins=100):
    """2D histogram of the samples, oriented to match `imshow(origin='lower')`."""
    x0, x1, y0, y1 = extent
    h, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=bins, range=[[x0, x1], [y0, y1]]
    )
    return h.T  # histogram2d indexes [x, y]; imshow wants [row=y, col=x]


def panel(ax, img, extent, title=None):
    ax.imshow(
        img,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=8)


def lam_label(lam):
    return f"{lam:g}"


def main():
    p = argparse.ArgumentParser(description="CNF vs AIT-CNF density grid")
    p.add_argument("--densities", default="results/densities")
    p.add_argument("--out", default="plots/cnf-densities.png")
    p.add_argument("--bins", type=int, default=100, help="bins for the data column")
    args = p.parse_args()

    dens = load_densities(args.densities)
    if not dens:
        raise SystemExit(
            f"no .npz under {args.densities}; run experiments.cnf.cnf first"
        )

    toys = [t for t in ROW_ORDER if any(k[1] == t.value for k in dens)]
    lams = sorted({lam for model, _, lam in dens if model == "ait"})
    columns = [("data", "Data"), ("node", "FFJORD")]
    columns += [("ait", f"AIT-FFJORD\n$\\lambda$={lam_label(lam)}", lam) for lam in lams]

    fig, axes = plt.subplots(
        len(toys), len(columns), figsize=(2.1 * len(columns), 2.3 * len(toys))
    )
    axes = np.asarray(axes).reshape(len(toys), len(columns))

    for r, toy in enumerate(toys):
        # every npz for a toy carries the same samples/extent, so any will do
        ref = next(d for k, d in dens.items() if k[1] == toy.value)
        extent = tuple(ref["extent"].tolist())

        for c, col in enumerate(columns):
            ax, title = axes[r, c], col[1] if r == 0 else None
            if col[0] == "data":
                panel(ax, data_histogram(ref["samples"], extent, args.bins), extent, title)
                continue
            # NODE has no ponder penalty, so it is always the lam=0 run
            lam = col[2] if len(col) > 2 else 0.0
            d = dens.get((col[0], toy.value, lam))
            if d is None:
                ax.axis("off")
                continue
            panel(ax, np.exp(d["logp"]), extent, title)

        axes[r, 0].set_ylabel(toy.value, fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"saved {args.out} ({len(toys)} toys x {len(columns)} columns)")


if __name__ == "__main__":
    main()
