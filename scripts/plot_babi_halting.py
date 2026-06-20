import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from experiments.training import load_model
from experiments.babi.model import TokenODEClassifier
from experiments.babi.data import build_vocab, parse_file, encode, task_files


def build(**hp):
    return TokenODEClassifier(jax.random.PRNGKey(0), **hp)


def main():
    p = argparse.ArgumentParser(
        description="bAbI: halting time T* vs number of reasoning hops"
    )
    p.add_argument("ckpt")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-len", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--outdir", default="plots")
    args = p.parse_args()

    model, hp = load_model(args.ckpt, build)
    vocab = build_vocab(args.data_dir, tuple(args.tasks))

    Ts, sup = [], []
    for path in task_files(args.data_dir, args.tasks, "test"):
        X, _, S = encode(parse_file(path), vocab, args.max_len)
        Xj = jnp.asarray(X)
        for i in range(0, len(X), args.batch_size):
            _, T, _ = model(Xj[i : i + args.batch_size])
            Ts.append(np.asarray(T))
        sup.append(S)
    T = np.concatenate(Ts)
    sup = np.concatenate(sup)

    hops = sorted({int(s) for s in sup if s > 0})
    means = [T[sup == h].mean() for h in hops]
    stds = [T[sup == h].std() for h in hops]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sup + np.random.uniform(-0.1, 0.1, sup.shape), T, s=4, alpha=0.15, color="gray")
    ax.errorbar(hops, means, yerr=stds, marker="o", capsize=4, lw=2, color="C0", label="mean $T^*$")
    ax.axhline(hp["t_max"], color="gray", ls=":", lw=1, label="$t_{max}$ (cap)")
    ax.set_xlabel("number of supporting facts (reasoning hops)")
    ax.set_ylabel("halting time $T^*$")
    ax.set_title("bAbI: adaptive depth vs reasoning hops")
    ax.set_xticks(hops)
    ax.legend()

    os.makedirs(args.outdir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out = os.path.join(args.outdir, f"babi_hops-{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
