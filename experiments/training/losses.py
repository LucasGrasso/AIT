import jax.numpy as jnp
import optax


def ce_loss(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def mse_loss(pred, y):
    return jnp.mean((pred - y) ** 2)
