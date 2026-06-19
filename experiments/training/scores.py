import jax.numpy as jnp


def accuracy(logits, y):
    correct = (logits.argmax(-1) == y).sum()
    return correct, y.shape[0]


def neg_mse_score(pred, y):
    return -jnp.sum((pred - y) ** 2), y.shape[0]


def sign_accuracy(pred, y):
    correct = (jnp.sign(pred) == jnp.sign(y)).sum()
    return correct, y.shape[0]
