import argparse


def base_parser(description=""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", choices=["ait", "node"], default="ait")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--t-max", type=float, default=1.0)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--log-every", type=int, default=10)
    return p


def config_from_args(args):
    return dict(
        runs=args.runs,
        seed=args.seed,
        epochs=args.epochs,
        model=args.model,
    )
