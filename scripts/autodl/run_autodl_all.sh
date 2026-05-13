#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-/root/autodl-tmp/data/preprocessed}"
MISSION_ROOT="${MISSION_ROOT:-/root/autodl-tmp/data/esa-adb/mission1/ESA-Mission1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/results_autodl}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")/../.."

"$PYTHON_BIN" scripts/tools/check_data.py \
  --data-path "$DATA_PATH" \
  --mission-root "$MISSION_ROOT"

"$PYTHON_BIN" run_benchmark.py "$DATA_PATH" \
  --preset all \
  --output-dir "$OUTPUT_DIR" \
  --include-vus-metrics \
  --vus-max-points 50000 \
  --vus-max-buffer 50 \
  --vus-max-thresholds 30 \
  --official-binary-fraction 0.05 \
  --official-merge-gap-points 30 \
  --official-min-event-points 2
