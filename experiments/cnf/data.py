from enum import Enum

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Toy(str, Enum):
    CHECKERBOARD = "checkerboard"
    TWO_SPIRALS = "2spirals"
    EIGHT_GAUSSIANS = "8gaussians"


def generate_checkerboard(n_samples, rng=None):
    """2D checkerboard density: 8 uniform squares on a 4x4 grid over [-4, 4]^2."""
    rng = np.random.default_rng(rng)
    x1 = rng.uniform(-2.0, 2.0, n_samples)
    x2_ = rng.uniform(0.0, 1.0, n_samples) - rng.integers(0, 2, n_samples) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return (np.stack([x1, x2], 1) * 2).astype(np.float32)


def generate_2spirals(n_samples, rng=None):
    """Two interleaved 1.5-turn Archimedean spirals with gaussian jitter."""
    rng = np.random.default_rng(rng)
    half = (n_samples + 1) // 2  # one arm each; trimmed back to n_samples below
    n = np.sqrt(rng.uniform(0.0, 1.0, (half, 1))) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + rng.uniform(0.0, 1.0, (half, 1)) * 0.5
    d1y = np.sin(n) * n + rng.uniform(0.0, 1.0, (half, 1)) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += rng.standard_normal(x.shape) * 0.1
    return x[:n_samples].astype(np.float32)


_EIGHT_GAUSSIANS_CENTERS = 4.0 * np.array(
    [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1 / np.sqrt(2), 1 / np.sqrt(2)),
        (1 / np.sqrt(2), -1 / np.sqrt(2)),
        (-1 / np.sqrt(2), 1 / np.sqrt(2)),
        (-1 / np.sqrt(2), -1 / np.sqrt(2)),
    ]
)


def generate_8gaussians(n_samples, rng=None):
    """Equal-weight mixture of 8 isotropic gaussians on a ring, sigma 0.5."""
    rng = np.random.default_rng(rng)
    idx = rng.integers(0, len(_EIGHT_GAUSSIANS_CENTERS), n_samples)
    x = rng.standard_normal((n_samples, 2)) * 0.5 + _EIGHT_GAUSSIANS_CENTERS[idx]
    return (x / 1.414).astype(np.float32)


GENERATORS = {
    Toy.CHECKERBOARD: generate_checkerboard,
    Toy.TWO_SPIRALS: generate_2spirals,
    Toy.EIGHT_GAUSSIANS: generate_8gaussians,
}


def generate(toy, n_samples, rng=None):
    return GENERATORS[Toy(toy)](n_samples, rng)


def get_loaders(
    batch_size, toy=Toy.CHECKERBOARD, n_train=10000, n_test=10000, seed=0
):
    """Density estimation has no targets; each sample is paired with itself so
    the batches match the (xb, yb) contract the shared Trainer expects."""
    xtr = generate(toy, n_train, rng=seed)
    xte = generate(toy, n_test, rng=seed + 1)
    train = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(xtr))
    test = TensorDataset(torch.from_numpy(xte), torch.from_numpy(xte))
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size, shuffle=True, drop_last=True, generator=g),
        DataLoader(test, batch_size, shuffle=False, drop_last=False),
    )
