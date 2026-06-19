import jax
import equinox as eqx


def nparams(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(x.size for x in leaves)


def torch_seed(key):
    """torch's DataLoader Generator wants a Python int, not a JAX key."""
    return int(jax.random.randint(key, (), 0, 2**31 - 1))


def train_sweep(model_factory, loaders_factory, config, trainer, logger):
    """Runs `config['runs']` trains with different seeds, same hyperparams.

    model_factory(key)    -> model(x) -> (out, T, steps)
    loaders_factory(seed) -> (train_loader, test_loader)
    trainer               -> Trainer instance
    config: dict{runs, seed, epochs, model}
    """
    root = jax.random.PRNGKey(config["seed"])
    all_rows = []
    for run_idx in range(config["runs"]):
        model_key, data_key = jax.random.split(jax.random.fold_in(root, run_idx))
        model = model_factory(model_key)
        logger.info(
            f"=== run {run_idx + 1}/{config['runs']} | params {nparams(model):,} ==="
        )
        train_loader, test_loader = loaders_factory(torch_seed(data_key))
        _, rows = trainer.fit(
            model,
            train_loader,
            test_loader,
            config["epochs"],
            config["model"],
            run_idx,
            logger,
        )
        all_rows.extend(rows)
    return all_rows
