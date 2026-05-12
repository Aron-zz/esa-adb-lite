#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/3] Python"
"$PYTHON_BIN" --version

echo "[2/3] Install dependencies"
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[3/3] Check imports"
"$PYTHON_BIN" - <<'PY'
import numpy
import pandas
import sklearn
import portion
import psutil

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("portion ok")
print("psutil ok")
PY

echo "AutoDL setup complete."

