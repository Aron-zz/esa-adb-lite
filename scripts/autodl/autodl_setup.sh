#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/3] Python"
"$PYTHON_BIN" --version

echo "[2/3] Install dependencies"
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[3/4] Ensure official metric dependency"
OFFICIAL_REPO="${OFFICIAL_REPO:-/root/esa-adb-classical}"
AFFILIATION_DIR="$OFFICIAL_REPO/timeeval/metrics/affiliation_based_metrics_repo/affiliation"
if [ ! -d "$AFFILIATION_DIR" ]; then
  echo "Cloning ESA-ADB affiliation metric dependency to $OFFICIAL_REPO"
  rm -rf "$OFFICIAL_REPO"
  git clone --filter=blob:none --sparse https://github.com/kplabs-pl/ESA-ADB.git "$OFFICIAL_REPO"
  (
    cd "$OFFICIAL_REPO"
    git sparse-checkout set timeeval/metrics/affiliation_based_metrics_repo mission1_experiments.py
  )
else
  echo "Official metric dependency already exists: $AFFILIATION_DIR"
fi

echo "[4/4] Check imports"
"$PYTHON_BIN" - <<'PY'
import numpy
import pandas
import sklearn
import portion
import psutil
import sys
from pathlib import Path

affiliation_repo = Path("/root/esa-adb-classical/timeeval/metrics/affiliation_based_metrics_repo")
if affiliation_repo.exists():
    sys.path.insert(0, str(affiliation_repo))
import affiliation

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("portion ok")
print("psutil ok")
print("affiliation ok")
PY

echo "AutoDL setup complete."
