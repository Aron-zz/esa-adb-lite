# AutoDL Full Run / AutoDL 全量实验

This page records the intended full cloud run for ESA-ADB Lite.

本文档记录 ESA-ADB Lite 在 AutoDL 上的全量实验入口。

## Scope / 实验范围

- Train splits / 训练集：`3_months`, `10_months`, `21_months`, `42_months`, `84_months`
- Test split / 测试集：`84_months.test.csv`
- Channel groups / 通道组：`Target_57`, `Subset_6`
- Algorithms / 算法：
  - `PCC`, `HBOS`, `STD`, `STD3`, `STD5`, `iForest`
  - `LOF`, `COPOD`, `RobustPCA`, `KNN`
  - `Subsequence_IF`, `Subsequence_KNN`

`Subset_6` is a subset of `Target_57`, not an independent dataset. It is kept because the official ESA-ADB Mission1 classical experiment evaluates both channel settings.

`Subset_6` 是 `Target_57` 的子集，不是独立数据集。保留它是因为官方 ESA-ADB Mission1 classical 实验同时评测这两种通道设置。

## Metrics / 指标

The full run enables:

全量实验开启：

- Point metrics / 点级指标：`AUROC`, `AUPR`, `F1`
- Channel-mean metrics / 按通道均值指标：`ChannelMean_*`
- VUS metrics / VUS 指标：`VUS_PR`, `VUS_ROC`
- ESA event-style metrics / ESA 事件风格指标：`ESAScoresLite`, `ADTQCLite`, `ChannelAwareFScoreLite`
- Slice metrics / 异常特征切片指标：`slice_metrics.csv`

Official event metrics convert continuous anomaly scores to binary event predictions with:

官方事件指标会先把连续异常分数转成二值事件预测：

```text
official_binary_fraction = 0.05
official_merge_gap_points = 30
official_min_event_points = 2
```

These values are saved to `run_config.json`.

这些参数会记录在 `run_config.json` 中。

## Command / 命令

```bash
cd /root/esa-adb-lite
bash scripts/autodl/autodl_setup.sh

screen -S esa_all
bash scripts/autodl/run_autodl_all.sh
```

Equivalent explicit command / 等价显式命令：

```bash
python run_benchmark.py /root/autodl-tmp/data/preprocessed \
  --preset all \
  --output-dir /root/autodl-tmp/results_autodl \
  --include-vus-metrics \
  --vus-max-points 50000 \
  --vus-max-buffer 50 \
  --vus-max-thresholds 30 \
  --official-binary-fraction 0.05 \
  --official-merge-gap-points 30 \
  --official-min-event-points 2
```

Do not add `--skip-official-metrics` for the full run.

全量实验不要添加 `--skip-official-metrics`。

## Expected Outputs / 预期输出

```text
/root/autodl-tmp/results_autodl/all/<timestamp>/
  run.log
  run_config.json
  results.csv
  summary_by_algorithm.csv
  summary_by_algorithm_channel.csv
  summary_key_metrics.csv
  slice_metrics.csv
  summary_slice_metrics.csv
```

## Monitor / 查看进度

```bash
tail -f /root/autodl-tmp/results_autodl/all/*/run.log
```

Detach from screen / 从 screen 退出但不中断任务：

```text
Ctrl+A, then D
```

Resume / 重新进入：

```bash
screen -r esa_all
```
