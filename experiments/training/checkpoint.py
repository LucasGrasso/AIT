import json

import equinox as eqx


def save_model(path, model, hyperparams):
    """Self-describing checkpoint: a JSON hyperparam header line, then leaves."""
    with open(path, "wb") as f:
        f.write((json.dumps(hyperparams) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(path, build):
    """`build(**hyperparams)` returns a template model to deserialise leaves into."""
    with open(path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        template = build(**hyperparams)
        model = eqx.tree_deserialise_leaves(f, template)
    return model, hyperparams
