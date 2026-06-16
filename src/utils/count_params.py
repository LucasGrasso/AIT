import jax
import equinox as eqx


def nparams(model: eqx.Module) -> int:
    return sum(
        x.size
        for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    )
