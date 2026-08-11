import argparse
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
    nll_loss,
    mean_loglik,
)
from ..logger import get_logger

from .model import CNFModel, make_args
from .data import Toy, get_loaders, generate
from .density import density_grid, density_path, save_density, with_exact_trace


def run_toy(args, toy, logger):
    logger.info(f"=== toy {toy} | model={args.model} | lam={args.lam} ===")

    def model_factory(key, hutchinson=None):
        return CNFModel(
            key,
            dim=args.dim,
            model=args.model,
            width=args.width,
            t_max=args.t_max,
            hutchinson=args.hutchinson if hutchinson is None else hutchinson,
        )

    def loaders_factory(seed):
        return get_loaders(
            args.batch_size,
            toy=toy,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=seed,
        )

    trainer = Trainer(
        optax.adam(args.lr),
        task_loss_fn=nll_loss,
        score_fn=mean_loglik,
        lam=args.lam,
        log_every=args.log_every,
        # eps for the Hutchinson estimator, redrawn per step and pinned by
        # diffrax for the whole trajectory. None keeps the exact-trace path.
        args_fn=make_args if args.hutchinson else None,
    )
    rows, models = train_sweep(
        model_factory,
        loaders_factory,
        config_from_args(args),
        trainer,
        logger,
    )

    lam_str = f"{args.lam:.10f}".rstrip("0").rstrip(".")
    tag = f"{args.model}_cnf_{toy}_{lam_str}"
    save_csv(rows, os.path.join("results", f"{tag}.csv"), logger)

    # checkpoint the best run: highest final-epoch mean log-likelihood
    last = max(r["epoch"] for r in rows)
    finals = [r for r in rows if r["epoch"] == last]
    best = max(finals, key=lambda r: r["test_score"])
    best_model = models[best["run"]]

    hyperparams = dict(
        dim=args.dim,
        model=args.model,
        width=args.width,
        t_max=args.t_max,
        hutchinson=args.hutchinson,
    )
    os.makedirs("models", exist_ok=True)
    ckpt = os.path.join("models", f"{tag}.eqx")
    save_model(ckpt, best_model, hyperparams)
    logger.info(
        f"saved best run {best['run']} (logp {best['test_score']:.4f}) -> {ckpt}"
    )

    # Density readout uses the exact trace: this is a picture, and the
    # estimator's variance would show up as speckle.
    eval_model = with_exact_trace(
        best_model, model_factory(jax.random.PRNGKey(0), hutchinson=False)
    )
    logp = density_grid(eval_model, res=args.grid_res)
    path = density_path(args.model, toy, args.lam)
    save_density(
        path,
        logp,
        generate(toy, args.n_test, rng=args.seed + 1),
        args.model,
        toy,
        args.lam,
        best["test_score"],
    )
    logger.info(f"saved density grid -> {path}")


def main():
    p = base_parser("CNF toy densities: AIT-NODE / NODE")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--width", type=int, default=64)
    p.add_argument(
        "--toys",
        nargs="+",
        default=[t.value for t in Toy],
        choices=[t.value for t in Toy],
    )
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--grid-res", type=int, default=200)
    p.add_argument(
        "--hutchinson",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stochastic trace estimator (FFJORD Algorithm 1); off = exact trace",
    )
    args = p.parse_args()

    logger = get_logger("cnf")
    logger.info(
        f"jax devices: {jax.devices()} | toys={args.toys} "
        f"| model={args.model} | lam={args.lam} | hutchinson={args.hutchinson}"
    )
    for toy in args.toys:
        run_toy(args, toy, logger)


if __name__ == "__main__":
    main()
