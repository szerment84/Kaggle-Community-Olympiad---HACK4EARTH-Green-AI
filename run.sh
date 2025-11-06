#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-optimized}"    # baseline | optimized
DATA="${2:-./data}"       # domyślnie lokalny folder ./data

export MODE
export DATA_PATH="$DATA"

echo "Running Green AI Optimizer in mode: $MODE"
echo "Using data path: $DATA_PATH"
echo "--------------------------------------------"

python -m src.pipeline --mode "$MODE" --data "$DATA"

echo "Done. Results saved to repository root (or outputs/)."
