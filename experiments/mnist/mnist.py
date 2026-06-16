"""
uv run python -m experiments.mnist.mnist --model ait  --epochs 5 --batch-size 128 --lam 0.00001 --t-max 1.0 --seed 42
uv run python -m experiments.mnist.mnist --model node --epochs 5 --batch-size 128 --lam 0.0 --t-max 1.0 --seed 42
"""

import os
import argparse
import time
import csv

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from ait import AITNeuralODE, NeuralODE

from .data import get_loaders


class ConvField(eqx.Module):
    """(C,H,W)->(C,H,W). 1x1 -> 3x3 -> 1x1"""

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
    lin: eqx.nn.Linear
    hmin: float = eqx.field(static=True)

    def __init__(self, key, channels=1, hmin=1e-3):
        self.lin = eqx.nn.Linear(channels, 1, key=key)
        self.hmin = hmin

    def __call__(self, x):  # (C,H,W)
        return jax.nn.softplus(self.lin(jnp.mean(x, axis=(1, 2))))[0] + self.hmin


class ODEClassifier(eqx.Module):
    ode: AITNeuralODE | NeuralODE
    head: eqx.nn.Linear

    def __init__(
        self, key, model="ait", channels=1, nf=64, hw=28, t_max=1.0, eps=1e-3, tol=1e-3
    ):
        k = jax.random.split(key, 3)
        f = ConvField(k[0], channels, nf)  # MISMO f para ambos
        if model == "ait":
            self.ode = AITNeuralODE(
                f, HaltUnit(k[1], channels), t_max=t_max, eps=eps, tol=tol
            )
        else:
            self.ode = NeuralODE(f, T=t_max, tol=tol)
        self.head = eqx.nn.Linear(channels * hw * hw, 10, key=k[2])  # flatten -> 10

    def __call__(self, imgs):  # (B,1,28,28)
        x_out, T, nfe = self.ode(imgs)  # ODE directo sobre la imagen; vmap interno
        logits = jax.vmap(lambda z: self.head(z.reshape(-1)))(x_out)
        return logits, T, nfe


def to_jax(xb, yb):
    return jnp.asarray(xb.numpy()), jnp.asarray(yb.numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ait", "node"], default="ait")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=0.00001)
    p.add_argument("--t-max", type=float, default=1.0)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=5)  # K
    args = p.parse_args()

    print(f"jax devices: {jax.devices()}  model={args.model}  lambda={args.lam}")
    key = jax.random.PRNGKey(args.seed)
    model = ODEClassifier(key, model=args.model, t_max=args.t_max)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            logits, T, nfe = m(x)  # 3 salidas
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return ce + args.lam * T.mean(), (T.mean(), nfe.mean())  # nfe como aux

        (loss, (T_mean, nfe_mean)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss, T_mean, nfe_mean

    @eqx.filter_jit
    def eval_batch(model, x):
        logits, T, nfe = model(x)
        return logits.argmax(1), T.sum(), nfe.sum()

    def run_experiment(run_idx):
        key = jax.random.PRNGKey(args.seed + run_idx)
        model = ODEClassifier(key, model=args.model, t_max=args.t_max)
        n_params = sum(p.size for p in eqx.filter(model, eqx.is_array))
        print(f"Run {run_idx} | model {args.model} | params {n_params}")
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        train_loader, test_loader = get_loaders(
            args.batch_size, args.subset, seed=args.seed + run_idx
        )
        rows = []
        for epoch in range(args.epochs):
            t0 = time.time()
            run_loss = run_T = run_nfe = 0.0
            for xb, yb in train_loader:
                x, y = to_jax(xb, yb)
                model, opt_state, loss, T_mean, nfe_mean = train_step(
                    model, opt_state, x, y
                )
                run_loss += float(loss)
                run_T += float(T_mean)
                run_nfe += float(nfe_mean)
            n = len(train_loader)

            correct = total = 0
            T_sum = 0.0
            nfe_sum = 0.0
            for xb, yb in test_loader:
                x, y = to_jax(xb, yb)
                pred, Tsum, nfest = eval_batch(model, x)
                correct += int((pred == y).sum())
                total += y.shape[0]
                T_sum += float(Tsum)
                nfe_sum += float(nfest)

            rows.append(
                dict(
                    run=run_idx,
                    model=args.model,
                    epoch=epoch,
                    train_loss=run_loss / n,
                    train_T=run_T / n,
                    test_acc=correct / total,
                    test_T=T_sum / total,
                    train_nfe=run_nfe / n,
                    test_nfe=nfe_sum / total,
                    epoch_time_s=round(time.time() - t0, 2),
                )
            )
            print(
                f"  run {run_idx} ep {epoch:02d} | acc {correct/total:.4f} | T* {T_sum/total:.3f}"
            )
        return rows

    all_rows = []
    for run_idx in range(args.runs):
        print(f"=== run {run_idx + 1}/{args.runs} ===")
        all_rows.extend(run_experiment(run_idx))

    out_path = os.path.join("results", f"{args.model}_mnist_{args.lam}.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved: {out_path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
