#!/usr/bin/env python3
"""Generate per-algorithm Jupyter notebooks for ESA-ADB Lite experiments."""
import json
from pathlib import Path


ALGORITHMS = [
    ("PCC", "Fast PCA reconstruction-error baseline.", "快速 PCA 重构误差基线。"),
    ("HBOS", "Fast per-channel histogram outlier score baseline.", "快速逐通道直方图异常分数基线。"),
    ("STD", "Fast mean/std threshold baseline.", "快速均值/标准差阈值基线。"),
    ("iForest", "Strong per-channel Isolation Forest baseline; slower on target channels.", "较强的逐通道 Isolation Forest 基线；target 通道组上较慢。"),
    ("COPOD", "Fast empirical-tail-probability baseline; promising in subset experiments.", "快速经验尾概率基线；subset 实验表现有潜力。"),
    ("RobustPCA", "Robust-scaled PCA reconstruction-error baseline.", "鲁棒缩放后的 PCA 重构误差基线。"),
    ("LOF", "Local Outlier Factor; heavy on the full ESA test split.", "局部离群因子；完整 ESA 测试集上较重。"),
    ("KNN", "Multivariate nearest-neighbor distance; heavy on large splits.", "多变量近邻距离；大 split 上较重。"),
    ("Subsequence_IF", "Windowed Isolation Forest for subsequence anomalies; cloud recommended.", "面向子序列异常的窗口 Isolation Forest；建议云端运行。"),
    ("Subsequence_KNN", "Windowed kNN for subsequence anomalies; cloud recommended.", "面向子序列异常的窗口 kNN；建议云端运行。"),
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


def notebook_for_algorithm(name, description_en, description_zh):
    heavy_note = ""
    if name in {"LOF", "KNN", "Subsequence_IF", "Subsequence_KNN", "iForest"}:
        heavy_note = "\n\n> EN: This algorithm can be slow on `target` channels. Start with `subset` and small splits locally; use AutoDL for larger runs.\n>\n> 中文：该算法在 `target` 通道组上可能较慢。本机建议先跑 `subset` 和小 split，大规模实验放到 AutoDL。"

    cells = [
        markdown_cell(
            f"""
# ESA-ADB Lite: {name}

EN: {description_en}

中文：{description_zh}{heavy_note}

EN: This notebook is a thin experiment launcher. The implementation lives in `algorithms.py` and `run_benchmark.py`, so notebook runs and command-line runs stay consistent.

中文：这个 notebook 只是单算法实验入口。算法实现仍在 `algorithms.py` 和 `run_benchmark.py` 中，因此 notebook 与命令行实验保持一致。
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

# EN: Good local defaults. For larger runs, change these before running.
# 中文：适合本机的默认配置。大规模实验运行前请修改这些开关。
DATASETS = ["3_months"]
CHANNEL_GROUPS = ["subset"]

# EN: Metric switches.
# 中文：指标开关。
INCLUDE_VUS = False
VUS_MAX_POINTS = 50000
SKIP_OFFICIAL_METRICS = True
SKIP_SLICE_METRICS = False

# EN: Common presets you may copy:
# 中文：可复制使用的常见配置：
# Local quick check / 本机快速验证:
# DATASETS = ["3_months"]; CHANNEL_GROUPS = ["subset"]; INCLUDE_VUS = False
# Cloud subset formal / 云端 subset 正式:
# DATASETS = ["3_months", "10_months", "21_months", "42_months", "84_months"]; CHANNEL_GROUPS = ["subset"]; INCLUDE_VUS = True
# Cloud target missing algorithms / 云端 target 补跑:
# DATASETS = ["3_months", "10_months", "21_months", "42_months", "84_months"]; CHANNEL_GROUPS = ["target"]; INCLUDE_VUS = True
"""
        ),
        markdown_cell(
            """
## Switches / 开关说明

- `DATASETS`: training split list, e.g. `["3_months"]` or all five splits. / 训练集 split 列表。
- `CHANNEL_GROUPS`: `["subset"]`, `["target"]`, or both. / 通道组。
- `INCLUDE_VUS`: add VUS interval metrics. Slower but useful for subsequence anomalies. / 是否计算 VUS 区间指标。
- `SKIP_OFFICIAL_METRICS`: keep `True` unless the official affiliation dependency is available. / 没配好官方依赖时保持 True。
- `SKIP_SLICE_METRICS`: set `False` to produce `slice_metrics.csv`; recommended for research analysis. / 是否跳过异常类型切片指标，研究分析建议 False。

## Run / 运行

Adjust the switches above, then run this cell.
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
if SKIP_SLICE_METRICS:
    cmd.append("--skip-slice-metrics")
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
## Inspect Latest Result / 查看最新结果

This loads the latest notebook run for this algorithm.

下面会读取当前算法最近一次 notebook 输出。
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
        markdown_cell(
            """
## Slice Metrics / 异常类型切片指标

`slice_metrics.csv` helps answer which algorithm works best for each anomaly type.

`slice_metrics.csv` 用来回答：不同异常类型、长度、局部/全局、多变量/单变量下，哪种算法更好。
"""
        ),
        code_cell(
            """
slice_path = latest / "slice_metrics.csv"
if slice_path.exists():
    slice_df = pd.read_csv(slice_path)
    display(slice_df.head(20))
else:
    print("slice_metrics.csv not found. It may have been disabled with SKIP_SLICE_METRICS=True.")
"""
        ),
        code_cell(
            """
slice_summary_path = latest / "summary_slice_metrics.csv"
if slice_summary_path.exists():
    summary_slice = pd.read_csv(slice_summary_path)
    display(summary_slice.sort_values(["slice_type", "slice_value", "Point_AUPR"], ascending=[True, True, False]).head(30))
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

    for name, description_en, description_zh in ALGORITHMS:
        notebook = notebook_for_algorithm(name, description_en, description_zh)
        path = out_dir / f"{name}.ipynb"
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index = "\n".join(
        [
            "# ESA-ADB Lite Algorithm Notebooks",
            "",
            "EN: Each notebook launches one algorithm through `run_benchmark.py`.",
            "中文：每个 notebook 都通过 `run_benchmark.py` 启动一个算法实验。",
            "",
            "EN: Keep algorithm implementations in `algorithms.py`; use notebooks for interactive experiments and result inspection.",
            "中文：算法实现保留在 `algorithms.py`，notebook 用于交互式实验和查看结果。",
            "",
            "## Switch Guide / 开关指南",
            "",
            "- `DATASETS`: training splits / 训练集 split",
            "- `CHANNEL_GROUPS`: `subset` or `target` / 通道组",
            "- `INCLUDE_VUS`: interval-aware VUS metrics / 区间友好的 VUS 指标",
            "- `SKIP_OFFICIAL_METRICS`: skip official ESA event metrics / 跳过官方 ESA 事件指标",
            "- `SKIP_SLICE_METRICS`: skip anomaly-type slice metrics / 跳过异常类型切片指标",
            "",
            "## Notebooks / 算法 Notebook",
            "",
            *[f"- [{name}]({name}.ipynb): {description_en} / {description_zh}" for name, description_en, description_zh in ALGORITHMS],
            "",
        ]
    )
    (out_dir / "README.md").write_text(index, encoding="utf-8")
    print(f"Wrote {len(ALGORITHMS)} notebooks to {out_dir}")


if __name__ == "__main__":
    main()
