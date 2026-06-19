import argparse
import glob
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
    p.add_argument("experiment", help="e.g. annuli1d, mnist; globs {model}_{exp}_{lam}.csv")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--outdir", default="plots")
    args = p.parse_args()

    sns.set_style("whitegrid")

    csvs = sorted(glob.glob(os.path.join(args.results_dir, f"*_{args.experiment}_*.csv")))
    if not csvs:
        raise SystemExit(f"no CSVs matching *_{args.experiment}_*.csv in {args.results_dir}")
    data = pd.concat([load(path) for path in csvs], ignore_index=True)

    # mean across runs for each (model, lam, epoch)
    stats = data.groupby(["model", "lam", "epoch"], as_index=False).agg(
        task_mean=("test_task_loss", "mean"),
        steps_mean=("test_steps", "mean"),
    )

    uniq = stats[["model", "lam"]].drop_duplicates()
    keys = sorted(
        ((str(m), float(l)) for m, l in zip(uniq["model"], uniq["lam"])),
        key=series_order,
    )
    palette = sns.color_palette("colorblind", len(keys))
    color = dict(zip(keys, palette))
    for key in keys:  # NODE is always black
        if key[0] == "node":
            color[key] = (0.0, 0.0, 0.0)
    label = {k: series_label(*k) for k in keys}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # epoch panels: faint per-run trajectories under a bold mean line
    epoch_panels = [
        (axes[0], "test_task_loss", "task_mean", "loss (task)"),
        (axes[1], "test_steps", "steps_mean", "solver steps"),
    ]
    for ax, yraw, ymean, ylabel in epoch_panels:
        for key in keys:
            model, lam = key
            c = color[key]
            d = data[(data["model"] == model) & (data["lam"] == lam)]
            for _, run_df in d.groupby("run"):
                run_df = run_df.sort_values("epoch")
                ax.plot(run_df["epoch"], run_df[yraw], color=c, lw=1.0, alpha=0.3)
            g = stats[(stats["model"] == model) & (stats["lam"] == lam)]
            g = g.sort_values("epoch")
            ax.plot(g["epoch"], g[ymean], color=c, lw=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)

    ax = axes[2]
    for key in keys:
        model, lam = key
        c = color[key]
        r = data[(data["model"] == model) & (data["lam"] == lam)]
        ax.scatter(
            r["test_steps"], r["test_task_loss"], color=c, alpha=0.3, linewidths=0
        )
    ax.set_xlabel("solver steps")
    ax.set_ylabel("loss (task)")
    ax.set_ylim(bottom=0)

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
    out = os.path.join(args.outdir, f"{args.experiment}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
