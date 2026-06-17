import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load(path):
    df = pd.read_csv(path)
    if "test_steps" not in df.columns:
        df["test_steps"] = 1.0
    # average over runs for each (lam, epoch)
    df = df.groupby(["lam", "epoch"], as_index=False).agg(
        {"test_task_loss": "mean", "test_steps": "mean", "model": "first"}
    )
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--outdir", default="plots")
    args = p.parse_args()

    sns.set_style("whitegrid")

    dfs = [load(path) for path in args.csvs]
    data = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (xcol, ycol, xlabel, ylabel) in zip(
        axes,
        [
            ("epoch", "test_task_loss", "epoch", "loss (task)"),
            ("epoch", "test_steps", "epoch", "solver steps"),
            ("test_steps", "test_task_loss", "solver steps", "loss (task)"),
        ],
    ):
        for lam, g in data.groupby("lam"):
            g = g.sort_values(xcol)
            ax.plot(g[xcol], g[ycol], label=f"$\\lambda={lam}$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    axes[0].set_title("epochs vs loss (task)")
    axes[1].set_title("epochs vs solver steps")
    axes[2].set_title("solver steps vs loss (task)")

    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    dataset = os.path.basename(args.csvs[0]).split("_")[1]
    tags = "-".join(f"{df['model'].iloc[0]}{df['lam'].iloc[0]}" for df in dfs)
    out = os.path.join(args.outdir, f"{dataset}-{tags}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
