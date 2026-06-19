#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resolve_experiment.sh"

resolve_experiment "$@" || exit 1
uv sync --no-dev

lambdas=(0.0001 0.001 0.01 0.1)

echo "=== Running AIT-NODE: $module ==="
for lam in "${lambdas[@]}"; do
  echo "=== lambda = $lam ==="
  uv run python -m "$module" --model ait --lam "$lam" "${ARGS[@]}"
done