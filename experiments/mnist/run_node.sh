#!/bin/bash
set -euo pipefail             

uv sync --no-dev

echo "=== Running NODE ==="
uv run python -m experiments.mnist.mnist \
	--model node --epochs 10 --batch-size 128 \
	--lam 0.0 --t-max 1.0 --seed 42 --runs 5