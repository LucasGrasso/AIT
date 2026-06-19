#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resolve_experiment.sh"

resolve_experiment "$@" || exit 1
uv sync --no-dev

echo "=== Running NODE: $module ==="
uv run python -m "$module" \
	--model node --lam 0.0 "${ARGS[@]}"