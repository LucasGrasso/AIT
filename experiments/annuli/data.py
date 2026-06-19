import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def gd(n_inner=1000, n_outer=2000, d=2, r_inner=0.5, r_outer=(1.0, 1.5), rng=None):
    """
    ||x|| <= r_inner                  -> -1
    r_outer[0] <= ||x|| <= r_outer[1] -> +1
    """
    rng = np.random.default_rng(rng)

    def sample(count, lo, hi):
        v = rng.standard_normal((count, d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        u = rng.uniform(0.0, 1.0, count)
        r = (lo**d + u * (hi**d - lo**d)) ** (1.0 / d)
        return v * r[:, None]

    inner = sample(n_inner, 0.0, r_inner)
    outer = sample(n_outer, r_outer[0], r_outer[1])
    x = np.concatenate([inner, outer]).astype(np.float32)
    y = np.concatenate([-np.ones(n_inner), np.ones(n_outer)]).astype(np.float32)[
        :, None
    ]
    return x, y


def get_loaders(
    batch_size,
    d=2,
    n_inner_train=1000,
    n_outer_train=2000,
    n_inner_test=1000,
    n_outer_test=2000,
    seed=0,
    r_inner=0.5,
    r_outer=(1.0, 1.5),
):
    xtr, ytr = gd(n_inner_train, n_outer_train, d, r_inner, r_outer, rng=seed)
    xte, yte = gd(n_inner_test, n_outer_test, d, r_inner, r_outer, rng=seed + 1)
    train = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    test = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte))
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size, shuffle=True, drop_last=True, generator=g),
        DataLoader(test, batch_size, shuffle=False, drop_last=False),
    )
