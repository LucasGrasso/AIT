#!/bin/bash
set -euo pipefail             

uv sync --no-dev

lambdas=(0.0001 0.001 0.01)  

echo "=== Running AIT-NODE ==="
for lam in "${lambdas[@]}"; do           
    echo "=== lambda = $lam ==="
    uv run python -m experiments.mnist.mnist \
        --model ait --epochs 10 --batch-size 128 \
        --lam "$lam" --t-max 1.0 --seed 42 --runs 2
done