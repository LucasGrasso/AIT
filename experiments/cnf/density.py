"""Evaluate a trained CNF model's density on a grid and cache it to disk.

Training and plotting are separated: runs write one .npz per (model, toy, lam)
into `results/densities/`, and `plot_densities.py` assembles whatever is there
into the comparison grid. That way the plot can be re-styled without retraining,
and a lambda sweep just accumulates candidates.
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# All three toys live inside [-4, 4]^2 (checkerboard by construction, the
# spirals top out at 3*pi/3 ~ 3.14, the gaussian ring at 4/1.414 + 3 sigma).
EXTENT = (-4.0, 4.0, -4.0, 4.0)


def with_exact_trace(model, template):
    """Trained weights of `model` on the static structure of `template`.

    `hutchinson` is a static field, so it cannot be flipped with `tree_at`;
    instead build a twin with the same hyperparams but `hutchinson=False` and
    graft the arrays across. Worth it because the estimator's variance would
    show up as speckle in the plot, and at dim=2 the exact trace is one extra
    JVP.

    The graft is by leaf order, not `eqx.combine`: static fields are part of the
    treedef, so combining across two structures that differ in `hutchinson` is
    exactly the mismatch equinox refuses. Leaf order is well defined here
    because the two differ *only* in that flag.
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    arrays, static = eqx.partition(template, eqx.is_array)
    treedef = jax.tree_util.tree_structure(arrays)
    if treedef.num_leaves != len(leaves):
        raise ValueError(
            f"template has {treedef.num_leaves} array leaves, model has "
            f"{len(leaves)}; they must differ only in `hutchinson`"
        )
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)


@eqx.filter_jit
def _logp_batch(model, xb):
    logp, _, _ = model(xb)
    return logp[:, 0]


def density_grid(model, res=200, extent=EXTENT, batch_size=5000):
    """log p on a res x res grid, shaped (res, res) for `imshow(origin='lower')`.

    `model` must be exact-trace; batches are padded to a fixed size so the
    jitted forward pass compiles once instead of once per ragged tail.
    """
    x0, x1, y0, y1 = extent
    gx, gy = np.meshgrid(
        np.linspace(x0, x1, res, dtype=np.float32),
        np.linspace(y0, y1, res, dtype=np.float32),
    )
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    n = pts.shape[0]
    padded = np.concatenate([pts, np.zeros(((-n) % batch_size, 2), np.float32)])
    chunks = [
        np.asarray(_logp_batch(model, jnp.asarray(padded[i : i + batch_size])))
        for i in range(0, padded.shape[0], batch_size)
    ]
    return np.concatenate(chunks)[:n].reshape(res, res)


def density_path(model_name, toy, lam, root="results/densities"):
    lam_str = f"{lam:.10f}".rstrip("0").rstrip(".")
    return os.path.join(root, f"{model_name}_{toy}_{lam_str}.npz")


def save_density(path, logp, samples, model_name, toy, lam, test_score):
    """`test_score` (mean log-likelihood) rides along so the plotter can pick the
    best lambda per (model, toy) without re-reading the CSVs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        logp=logp,
        samples=samples,
        extent=np.asarray(EXTENT),
        model=model_name,
        toy=str(toy),
        lam=lam,
        test_score=test_score,
    )
