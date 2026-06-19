import os

import jax
import optax

from ..training import (
    train_sweep,
    Trainer,
    save_csv,
    save_model,
    base_parser,
    config_from_args,
    smooth_l1_loss,
    sign_accuracy,
)
from ..logger import get_logger

from .model import VecODEModel
from .data import get_loaders


def main():
    p = base_parser("Annuli (ANODE gd) AIT-NODE / NODE")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--n-inner-train", type=int, default=1000)
    p.add_argument("--n-outer-train", type=int, default=2000)
    p.add_argument("--n-inner-test", type=int, default=1000)
    p.add_argument("--n-outer-test", type=int, default=2000)
    args = p.parse_args()

    logger = get_logger(f"annuli{args.dim}d")
    logger.info(
        f"jax devices: {jax.devices()} | dim={args.dim} "
        f"| model={args.model} | lam={args.lam}"
    )

    def model_factory(key):
        return VecODEModel(
            key,
            dim=args.dim,
            model=args.model,
            width=args.width,
            t_max=args.t_max,
            time_dependent=args.time_dependent,
        )

    def loaders_factory(seed):
        return get_loaders(
            args.batch_size,
            d=args.dim,
            n_inner_train=args.n_inner_train,
            n_outer_train=args.n_outer_train,
            n_inner_test=args.n_inner_test,
            n_outer_test=args.n_outer_test,
            seed=seed,
        )

    trainer = Trainer(
        optax.adam(args.lr),
        task_loss_fn=smooth_l1_loss,
        score_fn=sign_accuracy,
        lam=args.lam,
        log_every=args.log_every,
    )
    rows, models = train_sweep(
        model_factory,
        loaders_factory,
        config_from_args(args),
        trainer,
        logger,
    )

    lam_str = f"{args.lam:.10f}".rstrip("0").rstrip(".")
    tag = f"{args.model}_annuli{args.dim}d_{lam_str}"
    save_csv(rows, os.path.join("results", f"{tag}.csv"), logger)

    # checkpoint the best run: highest final-epoch score, then lowest loss
    last = max(r["epoch"] for r in rows)
    finals = [r for r in rows if r["epoch"] == last]
    best = max(finals, key=lambda r: (r["test_score"], -r["test_task_loss"]))
    hyperparams = dict(
        dim=args.dim,
        model=args.model,
        width=args.width,
        t_max=args.t_max,
        time_dependent=args.time_dependent,
    )
    os.makedirs("models", exist_ok=True)
    ckpt = os.path.join("models", f"{tag}.eqx")
    save_model(ckpt, models[best["run"]], hyperparams)
    logger.info(
        f"saved best run {best['run']} (score {best['test_score']:.4f}) -> {ckpt}"
    )


if __name__ == "__main__":
    main()
