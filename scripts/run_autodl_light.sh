#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-../data/preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-results_autodl}"

"$PYTHON_BIN" scripts/check_data.py --data-path "$DATA_PATH"

"$PYTHON_BIN" run_benchmark.py "$DATA_PATH" \
  --preset light \
  --output-dir "$OUTPUT_DIR" \
  --include-vus-metrics \
  --vus-max-points 50000 \
  --vus-max-buffer 50 \
  --vus-max-thresholds 30

