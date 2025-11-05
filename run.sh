#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-optimized}"   # baseline | optimized
DATA="${2:-/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai}"

export MODE
export DATA_PATH="$DATA"

python -m src.pipeline --mode "$MODE" --data "$DATA"
