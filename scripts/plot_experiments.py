import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


def load(path):
    df = pd.read_csv(path)
    if "test_steps" not in df.columns:
        df["test_steps"] = 1.0
    return df


def series_label(model, lam):
    if model == "node":
        return "NODE"
    return f"AIT $\\lambda={lam:g}$"


def series_order(key):
    model, lam = key
    return (0 if model == "node" else 1, lam)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--outdir", default="plots")
    args = p.parse_args()

    sns.set_style("whitegrid")

    dfs = [load(path) for path in args.csvs]
    data = pd.concat(dfs, ignore_index=True)

    # mean / std across runs for each (model, lam, epoch)
    stats = (
        data.groupby(["model", "lam", "epoch"], as_index=False)
        .agg(
            task_mean=("test_task_loss", "mean"),
            task_std=("test_task_loss", "std"),
            steps_mean=("test_steps", "mean"),
            steps_std=("test_steps", "std"),
        )
        .fillna(0.0)  # std is NaN with a single run
    )

    uniq = stats[["model", "lam"]].drop_duplicates()
    keys = sorted(
        ((str(m), float(l)) for m, l in zip(uniq["model"], uniq["lam"])),
        key=series_order,
    )
    palette = sns.color_palette("colorblind", len(keys))
    color = dict(zip(keys, palette))
    label = {k: series_label(*k) for k in keys}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        ("epoch", "task_mean", "task_std", "epoch", "loss (task)"),
        ("epoch", "steps_mean", "steps_std", "epoch", "solver steps"),
        ("steps_mean", "task_mean", "task_std", "solver steps", "loss (task)"),
    ]

    for ax, (xcol, ycol, ecol, xlabel, ylabel) in zip(axes, panels):
        for key in keys:
            model, lam = key
            g = stats[(stats["model"] == model) & (stats["lam"] == lam)]
            g = g.sort_values(xcol)
            x, y, e = g[xcol], g[ycol], g[ecol]
            c = color[key]
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.2, linewidth=0)
            ax.plot(x, y, color=c, lw=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    handles = [Line2D([0], [0], color=color[k], lw=2, label=label[k]) for k in keys]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))

    os.makedirs(args.outdir, exist_ok=True)
    dataset = os.path.basename(args.csvs[0]).split("_")[1]
    tags = "-".join(f"{df['model'].iloc[0]}{df['lam'].iloc[0]}" for df in dfs)
    out = os.path.join(args.outdir, f"{dataset}-{tags}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
