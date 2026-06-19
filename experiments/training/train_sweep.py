import jax
import equinox as eqx


def nparams(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(x.size for x in leaves)


def train_sweep(model_factory, loaders_factory, config, trainer, logger):
    """Runs `config['runs']` trains with different seeds, same hyperparams.

    model_factory(key)    -> model(x) -> (out, T, steps)
    loaders_factory(seed) -> (train_loader, test_loader)
    trainer               -> Trainer instance
    config: dict{runs, seed, epochs, model}
    """
    all_rows = []
    for run_idx in range(config["runs"]):
        key = jax.random.PRNGKey(config["seed"] + run_idx)
        model = model_factory(key)
        logger.info(
            f"=== run {run_idx + 1}/{config['runs']} | params {nparams(model):,} ==="
        )
        train_loader, test_loader = loaders_factory(config["seed"] + run_idx)
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
