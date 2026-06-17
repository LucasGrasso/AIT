import os

import jax

from ...training import (
    train_sweep,
    save_csv,
    base_parser,
    config_from_args,
    ce_loss,
    accuracy,
)
from ...logger import get_logger

from ..img_ode_classifier import ImgODEClassifier  # mismo modelo, parametrizado
from .data import get_loaders


def main():
    p = base_parser("CIFAR-10 AIT-NODE / NODE")
    p.add_argument("--nf", type=int, default=64)
    args = p.parse_args()

    logger = get_logger("cifar")
    logger.info(f"jax devices: {jax.devices()} | model={args.model} | lam={args.lam}")

    def model_factory(key):
        return ImgODEClassifier(
            key, model=args.model, channels=3, nf=args.nf, hw=32, t_max=args.t_max
        )

    def loaders_factory(seed):
        return get_loaders(args.batch_size, args.subset, seed=seed)

    rows = train_sweep(
        model_factory,
        loaders_factory,
        config_from_args(args),
        task_loss_fn=ce_loss,
        score_fn=accuracy,
        logger=logger,
    )

    lam_str = f"{args.lam:.10f}".rstrip("0").rstrip(".")
    out_path = os.path.join("results", f"{args.model}_cifar_{lam_str}.csv")
    save_csv(rows, out_path, logger)


if __name__ == "__main__":
    main()
