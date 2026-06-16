#!/bin/bash
set -euo pipefail             

uv sync --no-dev

lambdas=(0.000001 0.00001 0.0001 0.001)  

for lam in "${lambdas[@]}"; do           
    echo "=== lambda = $lam ==="
    uv run python -m experiments.mnist.mnist \
        --model ait --epochs 100 --batch-size 128 \
        --lam "$lam" --t-max 1.0 --seed 42 --runs 5
done