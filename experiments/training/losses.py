import jax.numpy as jnp
import optax


def ce_loss(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def mse_loss(pred, y):
    return jnp.mean((pred - y) ** 2)


def smooth_l1_loss(pred, y, delta=1.0):
    return jnp.mean(optax.huber_loss(pred, y, delta=delta))
