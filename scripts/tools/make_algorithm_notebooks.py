#!/usr/bin/env python3
"""Generate per-algorithm Jupyter notebooks for ESA-ADB Lite experiments."""
import json
from pathlib import Path


ALGORITHMS = [
    ("PCC", "Fast PCA reconstruction-error baseline."),
    ("HBOS", "Fast per-channel histogram outlier score baseline."),
    ("STD", "Fast mean/std threshold baseline."),
    ("iForest", "Strong per-channel Isolation Forest baseline; slower on target channels."),
    ("COPOD", "Fast empirical-tail-probability baseline; promising in subset experiments."),
    ("RobustPCA", "Robust-scaled PCA reconstruction-error baseline."),
    ("LOF", "Local Outlier Factor; heavy on the full ESA test split."),
    ("KNN", "Multivariate nearest-neighbor distance; heavy on large splits."),
    ("Subsequence_IF", "Windowed Isolation Forest for subsequence anomalies; cloud recommended."),
    ("Subsequence_KNN", "Windowed kNN for subsequence anomalies; cloud recommended."),
]


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def notebook_for_algorithm(name, description):
    heavy_note = ""
    if name in {"LOF", "KNN", "Subsequence_IF", "Subsequence_KNN", "iForest"}:
        heavy_note = "\n\n> This algorithm can be slow on `target` channels. Start with `subset` and small splits locally; use AutoDL for larger runs."

    cells = [
        markdown_cell(
            f"""
# ESA-ADB Lite: {name}

{description}{heavy_note}

This notebook is a thin experiment launcher. The actual implementation lives in `algorithms.py` and `run_benchmark.py`, so notebook runs and command-line runs stay consistent.
"""
        ),
        code_cell(
            f"""
from pathlib import Path
import subprocess
import sys
import pandas as pd

REPO = Path.cwd()
if not (REPO / "run_benchmark.py").exists():
    REPO = REPO.parent

ALGORITHM = "{name}"
DATA_PATH = REPO.parent / "data" / "preprocessed"
OUTPUT_DIR = REPO / "results" / "notebooks"

# Good local defaults. For larger runs, change these before running.
DATASETS = ["3_months"]
CHANNEL_GROUPS = ["subset"]
INCLUDE_VUS = False
VUS_MAX_POINTS = 50000
SKIP_OFFICIAL_METRICS = True
"""
        ),
        markdown_cell(
            """
## Run

Adjust `DATASETS`, `CHANNEL_GROUPS`, and `INCLUDE_VUS` above, then run this cell.
"""
        ),
        code_cell(
            """
cmd = [
    sys.executable,
    "run_benchmark.py",
    str(DATA_PATH),
    "--preset",
    "extended",
    "--output-dir",
    str(OUTPUT_DIR),
    "--datasets",
    *DATASETS,
    "--channel-groups",
    *CHANNEL_GROUPS,
    "--algorithms",
    ALGORITHM,
]

if SKIP_OFFICIAL_METRICS:
    cmd.append("--skip-official-metrics")
if INCLUDE_VUS:
    cmd.extend([
        "--include-vus-metrics",
        "--vus-max-points",
        str(VUS_MAX_POINTS),
        "--vus-max-buffer",
        "50",
        "--vus-max-thresholds",
        "30",
    ])

print(" ".join(str(x) for x in cmd))
subprocess.run(cmd, cwd=REPO, check=True)
"""
        ),
        markdown_cell(
            """
## Inspect Latest Result

This loads the latest notebook run for this algorithm.
"""
        ),
        code_cell(
            """
run_dirs = sorted((OUTPUT_DIR / "extended").glob("*"))
if not run_dirs:
    raise FileNotFoundError("No notebook run found.")

latest = run_dirs[-1]
print(latest)
df = pd.read_csv(latest / "results.csv")
df
"""
        ),
        code_cell(
            """
summary_path = latest / "summary_by_algorithm.csv"
if summary_path.exists():
    pd.read_csv(summary_path)
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, description in ALGORITHMS:
        notebook = notebook_for_algorithm(name, description)
        path = out_dir / f"{name}.ipynb"
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index = "\n".join(
        [
            "# ESA-ADB Lite Algorithm Notebooks",
            "",
            "Each notebook launches one algorithm through `run_benchmark.py`.",
            "Keep algorithm implementations in `algorithms.py`; use notebooks for interactive experiments and result inspection.",
            "",
            *[f"- [{name}]({name}.ipynb): {description}" for name, description in ALGORITHMS],
            "",
        ]
    )
    (out_dir / "README.md").write_text(index, encoding="utf-8")
    print(f"Wrote {len(ALGORITHMS)} notebooks to {out_dir}")


if __name__ == "__main__":
    main()
