import os

import jax
import optax

from ..training import (
    train_sweep,
    Trainer,
    save_csv,
    base_parser,
    config_from_args,
    ce_loss,
    accuracy,
)
from ..logger import get_logger

from .img_ode_classifier import ImgODEClassifier
from .mnist.data import get_loaders as mnist_loaders
from .cifar.data import get_loaders as cifar_loaders


DATASETS = {
    "mnist": dict(channels=1, hw=28, get_loaders=mnist_loaders),
    "cifar": dict(channels=3, hw=32, get_loaders=cifar_loaders),
}


def main():
    p = base_parser("Image AIT-NODE / NODE")
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--nf", type=int, default=64)
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    logger = get_logger(args.dataset)
    logger.info(
        f"jax devices: {jax.devices()} | dataset={args.dataset} "
        f"| model={args.model} | lam={args.lam}"
    )

    def model_factory(key):
        return ImgODEClassifier(
            key,
            model=args.model,
            channels=spec["channels"],
            nf=args.nf,
            hw=spec["hw"],
            t_max=args.t_max,
        )

    def loaders_factory(seed):
        return spec["get_loaders"](args.batch_size, args.subset, seed=seed)

    trainer = Trainer(
        optax.adam(args.lr),
        task_loss_fn=ce_loss,
        score_fn=accuracy,
        lam=args.lam,
        log_every=args.log_every,
    )
    rows = train_sweep(
        model_factory,
        loaders_factory,
        config_from_args(args),
        trainer,
        logger,
    )

    lam_str = f"{args.lam:.10f}".rstrip("0").rstrip(".")
    out_path = os.path.join("results", f"{args.model}_{args.dataset}_{lam_str}.csv")
    save_csv(rows, out_path, logger)


if __name__ == "__main__":
    main()
