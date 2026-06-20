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
    ce_loss,
    accuracy,
)
from ..logger import get_logger

from .model import TokenODEClassifier
from .data import get_loaders, build_vocab


def main():
    p = base_parser("bAbI latent Neural-ODE transformer (AIT adaptive depth)")
    p.add_argument("--data-dir", required=True, help="dir with qaN_*_{train,test}.txt")
    p.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-len", type=int, default=120)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument(
        "--h-min",
        type=float,
        default=None,
        help="halting floor; default 1/t_max guarantees A=1 fires within [0, t_max]",
    )
    args = p.parse_args()

    h_min = args.h_min if args.h_min is not None else 1.0 / args.t_max
    logger = get_logger("babi")
    vocab_size = len(build_vocab(args.data_dir, args.tasks))
    logger.info(
        f"jax devices: {jax.devices()} | model={args.model} | lam={args.lam} "
        f"| tasks={args.tasks} | vocab={vocab_size} | t_max={args.t_max} | h_min={h_min}"
    )

    def model_factory(key):
        return TokenODEClassifier(
            key,
            vocab_size,
            model=args.model,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            t_max=args.t_max,
            h_min=h_min,
        )

    def loaders_factory(seed):
        train, test, _ = get_loaders(
            args.batch_size,
            args.data_dir,
            tasks=tuple(args.tasks),
            max_len=args.max_len,
            seed=seed,
        )
        return train, test

    trainer = Trainer(
        optax.adam(args.lr),
        task_loss_fn=ce_loss,
        score_fn=accuracy,
        lam=args.lam,
        log_every=args.log_every,
    )
    rows, models = train_sweep(
        model_factory, loaders_factory, config_from_args(args), trainer, logger
    )

    lam_str = f"{args.lam:.10f}".rstrip("0").rstrip(".")
    tag = f"{args.model}_babi_{lam_str}"
    save_csv(rows, os.path.join("results", f"{tag}.csv"), logger)

    last = max(r["epoch"] for r in rows)
    finals = [r for r in rows if r["epoch"] == last]
    best = max(finals, key=lambda r: (r["test_score"], -r["test_task_loss"]))
    hyperparams = dict(
        vocab=vocab_size,
        model=args.model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        t_max=args.t_max,
        h_min=h_min,
    )
    os.makedirs("models", exist_ok=True)
    ckpt = os.path.join("models", f"{tag}.eqx")
    save_model(ckpt, models[best["run"]], hyperparams)
    logger.info(
        f"saved best run {best['run']} (score {best['test_score']:.4f}) -> {ckpt}"
    )


if __name__ == "__main__":
    main()
