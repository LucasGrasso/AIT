import time

import jax.numpy as jnp
import equinox as eqx


def to_jax(xb, yb):
    return jnp.asarray(xb.numpy()), jnp.asarray(yb.numpy())


class Trainer:
    """Trains AIT-NODE / NODE models for a single task.

    task_loss_fn(out, y) -> scalar (batch mean)
    score_fn(out, y)     -> (agg_sum, count)  addable over batches
    lam                  -> weight of the ponder penalty `lam * T.mean()`;
                            NODE runs pass lam=0.
    """

    def __init__(
        self,
        optimizer,
        task_loss_fn,
        score_fn,
        lam=0.0,
        log_every=10,
    ):
        self.optimizer = optimizer
        self.task_loss_fn = task_loss_fn
        self.score_fn = score_fn
        self.lam = lam
        self.log_every = log_every

    @eqx.filter_jit
    def train_step(self, model, opt_state, x, y):
        def loss_fn(m):
            out, T, _ = m(x)
            task = self.task_loss_fn(out, y)
            ponder = self.lam * T.mean()
            return task + ponder, (task, ponder)

        (_, (task, ponder)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model
        )
        updates, opt_state = self.optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, task, ponder

    @eqx.filter_jit
    def eval_step(self, model, x, y):
        out, T, steps = model(x)
        task_sum = self.task_loss_fn(out, y) * y.shape[0]
        score_sum, _ = self.score_fn(out, y)
        return score_sum, task_sum, T.sum(), steps.sum()

    def fit(self, model, train_loader, test_loader, epochs, model_name, run_idx, logger):
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        rows = []
        for epoch in range(epochs):
            t0 = time.time()

            task_acc = ponder_acc = 0.0
            for xb, yb in train_loader:
                x, y = to_jax(xb, yb)
                model, opt_state, task, ponder = self.train_step(
                    model, opt_state, x, y
                )
                task_acc += float(task)
                ponder_acc += float(ponder)
            n = len(train_loader)

            score_acc = total = 0.0
            task_eval = T_acc = steps_acc = 0.0
            for xb, yb in test_loader:
                x, y = to_jax(xb, yb)
                score_sum, task_sum, Ts, steps = self.eval_step(model, x, y)
                score_acc += float(score_sum)
                total += y.shape[0]
                task_eval += float(task_sum)
                T_acc += float(Ts)
                steps_acc += float(steps)

            row = dict(
                run=run_idx,
                model=model_name,
                epoch=epoch,
                lam=self.lam,
                task_loss=task_acc / n,
                ponder_loss=ponder_acc / n,
                test_score=score_acc / total,
                test_task_loss=task_eval / total,
                test_steps=steps_acc / total,
                test_t=T_acc / total,
                epoch_time_s=round(time.time() - t0, 2),
            )
            rows.append(row)
            if epoch % self.log_every == 0 or epoch == epochs - 1:
                logger.info(
                    f"ep {epoch:02d} | score {row['test_score']:.4f} "
                    f"| T* {row['test_t']:.3f} | {row['epoch_time_s']}s"
                )
        return model, rows
