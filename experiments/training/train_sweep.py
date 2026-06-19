import os
import csv
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


def nparams(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(x.size for x in leaves)


def to_jax(xb, yb):
    return jnp.asarray(xb.numpy()), jnp.asarray(yb.numpy())


def make_step_fns(optimizer, lam, task_loss_fn, score_fn):
    @eqx.filter_jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m):
            out, T, _ = m(x)
            task = task_loss_fn(out, y)
            ponder = lam * T.mean()
            return task + ponder, (task, ponder)

        (_, (task, ponder)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, task, ponder

    @eqx.filter_jit
    def eval_batch(model, x, y):
        out, T, steps = model(x)
        task_sum = task_loss_fn(out, y) * y.shape[0]
        score_sum, _ = score_fn(out, y)
        return score_sum, task_sum, T.sum(), steps.sum()

    return train_step, eval_batch


def _run_one(
    model,
    optimizer,
    opt_state,
    train_step,
    eval_batch,
    train_loader,
    test_loader,
    epochs,
    lam,
    model_name,
    run_idx,
    logger,
    log_every=10,
):
    rows = []
    for epoch in range(epochs):
        t0 = time.time()

        task_acc = ponder_acc = 0.0
        for xb, yb in train_loader:
            x, y = to_jax(xb, yb)
            model, opt_state, task, ponder = train_step(model, opt_state, x, y)
            task_acc += float(task)
            ponder_acc += float(ponder)
        n = len(train_loader)

        score_acc = total = 0.0
        task_eval = T_acc = steps_acc = 0.0
        for xb, yb in test_loader:
            x, y = to_jax(xb, yb)
            score_sum, task_sum, Ts, steps = eval_batch(model, x, y)
            score_acc += float(score_sum)
            total += y.shape[0]
            task_eval += float(task_sum)
            T_acc += float(Ts)
            steps_acc += float(steps)

        row = dict(
            run=run_idx,
            model=model_name,
            epoch=epoch,
            lam=lam,
            task_loss=task_acc / n,
            ponder_loss=ponder_acc / n,
            test_score=score_acc / total,
            test_task_loss=task_eval / total,
            test_steps=steps_acc / total,
            test_t=T_acc / total,
            epoch_time_s=round(time.time() - t0, 2),
        )
        rows.append(row)
        if epoch % log_every == 0 or epoch == epochs - 1:
            logger.info(
                f"ep {epoch:02d} | score {row['test_score']:.4f} "
                f"| T* {row['test_t']:.3f} | {row['epoch_time_s']}s"
            )
    return model, rows


def train_sweep(model_factory, loaders_factory, config, task_loss_fn, score_fn, logger):
    """Runs `config['runs']` different trains with different seeds, same hyperparams.

    model_factory(key) -> model(x)->(out,T)
    loaders_factory(seed) -> (train_loader, test_loader)
    task_loss_fn(out, y) -> scalar (batch mean)
    score_fn(out, y) -> (agg_sum, count)  addable over batches.
    config: dict{lr, lam, runs, seed, epochs, model}
    """
    optimizer = optax.adam(config["lr"])
    train_step, eval_batch = make_step_fns(
        optimizer, config["lam"], task_loss_fn, score_fn
    )

    all_rows = []
    for run_idx in range(config["runs"]):
        key = jax.random.PRNGKey(config["seed"] + run_idx)
        model = model_factory(key)
        logger.info(
            f"=== run {run_idx + 1}/{config['runs']} | params {nparams(model):,} ==="
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        train_loader, test_loader = loaders_factory(config["seed"] + run_idx)
        _, rows = _run_one(
            model,
            optimizer,
            opt_state,
            train_step,
            eval_batch,
            train_loader,
            test_loader,
            config["epochs"],
            config["lam"],
            config["model"],
            run_idx,
            logger,
            config["log_every"],
        )
        all_rows.extend(rows)
    return all_rows


def save_csv(rows, path, logger):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"saved {path} ({len(rows)} rows)")
