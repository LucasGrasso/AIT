import jax.numpy as jnp
import optax


def ce_loss(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def mse_loss(pred, y):
    return jnp.mean((pred - y) ** 2)


def smooth_l1_loss(pred, y, delta=1.0):
    return jnp.mean(optax.huber_loss(pred, y, delta=delta))


def nll_loss(logp, y):
    """Negative log-likelihood in nats/sample for a density model.

    `logp` is (B, 1); density estimation has no targets, so `y` is ignored (the
    loaders pair every sample with itself to satisfy the Trainer contract)."""
    del y
    return -jnp.mean(logp)
