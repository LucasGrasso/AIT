<div align="center">

# Adaptive Integration Time for Neural ODEs

</div>
<p align="center">Jax Implementation of the Adaptive Integration Time (AIT) algorithm for Neural ODEs.</p>

## Requirements

- Linux (x86_64)
- Python 3.11
- NVIDIA GPU with CUDA 13 (JAX is installed with the `cuda13` extra)
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

This creates a virtual environment in `.venv/` and installs the locked
dependencies from `uv.lock`. Run commands with `uv run`, e.g.:

```bash
uv run python scripts/plot_experiments.py results/ait_mnist_0.001.csv results/node_mnist_0.csv
```

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