# Notebook Usage / Notebook 使用说明

This project supports notebook-first experiments. Each notebook in `notebooks/` launches exactly one algorithm through `run_benchmark.py`.

本项目支持以 notebook 为主的实验方式。`notebooks/` 目录下每个 notebook 都通过 `run_benchmark.py` 启动一个算法。

## Recommended Workflow / 推荐流程

1. Open one algorithm notebook, for example `notebooks/COPOD.ipynb`.
2. Edit the switch cell.
3. Run the benchmark cell.
4. Inspect `results.csv`, `summary_by_algorithm.csv`, and `slice_metrics.csv`.

中文：

1. 打开某个算法 notebook，例如 `notebooks/COPOD.ipynb`。
2. 修改开关配置单元。
3. 运行 benchmark 单元。
4. 查看 `results.csv`、`summary_by_algorithm.csv` 和 `slice_metrics.csv`。

## Main Switches / 主要开关

```python
DATASETS = ["3_months"]
CHANNEL_GROUPS = ["subset"]
INCLUDE_VUS = False
VUS_MAX_POINTS = 50000
SKIP_OFFICIAL_METRICS = True
SKIP_SLICE_METRICS = False
```

| Switch | Meaning | 中文说明 |
| --- | --- | --- |
| `DATASETS` | Training splits to run. | 要运行的训练集 split。 |
| `CHANNEL_GROUPS` | `subset`, `target`, or both. | 通道组，可选 `subset`、`target` 或二者。 |
| `INCLUDE_VUS` | Compute interval-aware VUS metrics. | 是否计算区间友好的 VUS 指标。 |
| `VUS_MAX_POINTS` | Max compressed points for VUS. | VUS 计算时压缩后的最大点数。 |
| `SKIP_OFFICIAL_METRICS` | Skip ESA official event metrics. | 是否跳过 ESA 官方事件指标。 |
| `SKIP_SLICE_METRICS` | Skip anomaly-type slice metrics. | 是否跳过异常类型切片指标。 |

## Local Quick Check / 本机快速验证

Use this for fast sanity checks:

```python
DATASETS = ["3_months"]
CHANNEL_GROUPS = ["subset"]
INCLUDE_VUS = False
SKIP_OFFICIAL_METRICS = True
SKIP_SLICE_METRICS = False
```

中文：用于快速检查代码和数据是否正常。

## AutoDL Formal Subset Run / AutoDL subset 正式实验

```python
DATASETS = ["3_months", "10_months", "21_months", "42_months", "84_months"]
CHANNEL_GROUPS = ["subset"]
INCLUDE_VUS = True
SKIP_OFFICIAL_METRICS = True
SKIP_SLICE_METRICS = False
```

中文：用于完整 subset 通道组实验，建议云端运行。

## AutoDL Target Run / AutoDL target 实验

```python
DATASETS = ["3_months", "10_months", "21_months", "42_months", "84_months"]
CHANNEL_GROUPS = ["target"]
INCLUDE_VUS = True
SKIP_OFFICIAL_METRICS = True
SKIP_SLICE_METRICS = False
```

中文：用于 57 个目标通道实验，耗时更长，建议只补跑缺失算法。

## Metric Outputs / 指标输出

Each notebook run writes to:

```text
results/notebooks/extended/<timestamp>/
```

Important files:

- `results.csv`: overall algorithm metrics / 算法总体指标
- `summary_by_algorithm.csv`: mean metrics by algorithm / 按算法汇总
- `slice_metrics.csv`: metrics by anomaly type / 按异常类型切片指标
- `summary_slice_metrics.csv`: averaged slice metrics / 切片指标汇总
- `run_config.json`: run switches / 运行配置
- `run.log`: execution log / 运行日志

## When to Use Official Metrics / 何时使用官方指标

Keep `SKIP_OFFICIAL_METRICS = True` for normal notebook experiments unless the official affiliation metric dependency is installed.

中文：常规 notebook 实验建议保持 `SKIP_OFFICIAL_METRICS = True`。只有当官方 affiliation 指标依赖已经配置好时，再打开官方指标。

## Official Alignment / 官方算法对齐

The official ESA-ADB Mission1 classical set is documented in:

```text
docs/algorithm_alignment.md
```

中文：官方 ESA-ADB Mission1 classical 算法组的对齐关系见：

```text
docs/algorithm_alignment.md
```

In notebooks, run these algorithms for an official-aligned classical comparison:

```text
PCC, HBOS, STD3, STD5, iForest, Subsequence_IF, KNN
```

中文：如果使用 notebook 逐算法运行，官方对齐实验需要跑以上这些算法。`STD` 是兼容入口，等价于 `STD3`。
