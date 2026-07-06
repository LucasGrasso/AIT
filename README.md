<div align="center">

# Adaptive Integration Time for Neural ODEs

</div>
<p align="center">Jax Implementation of the Adaptive Integration Time (AIT) algorithm for Neural ODEs.</p>

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- Optional: an NVIDIA GPU with CUDA 13 for the `cuda` extra

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended). CPU-only:

```bash
uv sync
```

With CUDA 13 GPU support:

```bash
uv sync --extra cuda
```

This creates a virtual environment in `.venv/` and installs the locked
dependencies from `uv.lock`. Run commands with `uv run`, e.g.:

```bash
uv run python scripts/plot_experiments.py results/ait_mnist_0.001.csv results/node_mnist_0.csv
```

The experiment scripts below use `uv run --no-sync`, so they respect
whichever extras you synced with (a plain `uv run` would re-sync without the
`cuda` extra and uninstall the GPU wheels).

Alternatively, install into an existing environment with pip:

```bash
pip install -e .
```

## Reproduce the experiments:

```bash
# Make the scripts executable
chmod +x experiments/run_ait.sh
chmod +x experiments/run_node.sh

# Run the experiments
./experiments/run_ait.sh g2
./experiments/run_node.sh g2
```
