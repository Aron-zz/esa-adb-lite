# ESA-ADB Lite

Docker-free, lightweight ESA-ADB Mission1 classical benchmark.

This repository keeps the core of the official ESA-ADB Mission1 classical setup while avoiding the heavy TimeEval/Docker runtime. The official benchmark repository is https://github.com/kplabs-pl/ESA-ADB.

## Scope

- Mission: `ESA-Mission1`
- Train splits: `3_months`, `10_months`, `21_months`, `42_months`, `84_months`
- Test split: `84_months.test.csv`
- Channel groups:
  - `subset`: `channel_41` to `channel_46`
  - `target`: official 57 target channels
- Classical algorithms:
  - `PCC`
  - `HBOS`
  - `STD`
  - `iForest`
  - `LOF`
  - `COPOD`
  - `RobustPCA`
  - `KNN`
  - `Subsequence_IF`
  - `Subsequence_KNN`

`LOF`, `KNN`, `Subsequence_IF`, and `Subsequence_KNN` are available but computationally heavy on the full ESA test split. Prefer explicit targeted/cloud runs for them.

## Setup

Use the project virtual environment from the workspace root:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The benchmark expects official ESA-ADB preprocessed CSV files at:

```text
..\data\preprocessed\multivariate\ESA-Mission1-semi-supervised\
```

It also expects the raw Mission1 metadata at:

```text
..\data\esa-adb\mission1\ESA-Mission1\
```

## Run Protocols

Smoke test:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py --preset smoke
```

Light benchmark:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py --preset light
```

Full classical benchmark:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py --preset full
```

Extended benchmark with additional lightweight algorithms:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py --preset extended
```

Useful targeted runs:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py --datasets 3_months --channel-groups subset --algorithms HBOS STD
..\.venv\Scripts\python.exe run_benchmark.py --datasets 3_months --channel-groups subset --algorithms COPOD RobustPCA --skip-official-metrics
..\.venv\Scripts\python.exe run_benchmark.py --datasets 3_months --channel-groups subset --algorithms LOF --skip-official-metrics
..\.venv\Scripts\python.exe run_benchmark.py --preset smoke --skip-official-metrics
..\.venv\Scripts\python.exe run_benchmark.py --preset smoke --skip-official-metrics --include-vus-metrics
..\.venv\Scripts\python.exe run_benchmark.py --dry-run
```

Analyze anomaly feature slices and create figures:

```powershell
..\.venv\Scripts\python.exe scripts\analyze_anomaly_features.py `
  --mission-root ..\data\esa-adb\mission1\ESA-Mission1 `
  --results-csv results\light\<timestamp>\results.csv
```

Plot concrete anomaly windows on telemetry curves:

```powershell
..\.venv\Scripts\python.exe scripts\plot_anomaly_windows.py `
  --event-ids id_118 id_126 `
  --max-channels-per-event 4 `
  --context-hours 12 `
  --output-dir results\anomaly_windows_examples
```

Build the full visual report:

```powershell
..\.venv\Scripts\python.exe scripts\build_visual_report.py `
  --data-path ..\data\preprocessed `
  --mission-root ..\data\esa-adb\mission1\ESA-Mission1 `
  --results-csv results\results.csv `
  --output-dir results\visual_report
```

## AutoDL

On AutoDL or another Linux cloud host, clone this repository, place the data next to it, and run:

```bash
cd esa_adb_lite
bash scripts/autodl_setup.sh
python scripts/check_data.py --data-path ../data/preprocessed
```

Expected data layout:

```text
workspace/
  esa_adb_lite/
  data/
    preprocessed/
      multivariate/
        ESA-Mission1-semi-supervised/
          3_months.train.csv
          10_months.train.csv
          21_months.train.csv
          42_months.train.csv
          84_months.train.csv
          84_months.test.csv
    esa-adb/
      mission1/
        ESA-Mission1/
          labels.csv
          anomaly_types.csv
          channels.csv
```

Recommended first cloud run:

```bash
bash scripts/run_autodl_light.sh
```

Run anomaly feature analysis on AutoDL:

```bash
python scripts/analyze_anomaly_features.py \
  --mission-root /root/autodl-tmp/data/esa-adb/mission1/ESA-Mission1 \
  --results-csv /root/autodl-tmp/results_autodl/light/<timestamp>/results.csv \
  --output-dir /root/autodl-tmp/results_autodl/anomaly_feature_analysis
```

If memory is limited, start with:

```bash
python run_benchmark.py ../data/preprocessed \
  --preset light \
  --datasets 3_months 10_months 21_months 42_months \
  --channel-groups subset \
  --include-vus-metrics \
  --vus-max-points 50000
```

Recommended cloud-side extended run:

```bash
python run_benchmark.py /root/autodl-tmp/data/preprocessed \
  --preset extended \
  --output-dir /root/autodl-tmp/results_autodl \
  --channel-groups subset \
  --skip-official-metrics \
  --include-vus-metrics \
  --vus-max-points 50000 \
  --vus-max-buffer 50 \
  --vus-max-thresholds 30
```

Run `full` mainly on cloud, because `KNN`, `Subsequence_IF`, and `Subsequence_KNN` are much heavier than the light/extended algorithms.
The neighbor-based methods use deterministic training-sample caps by default, but they still query the full test split and can be slow:

- `LOF`: up to 50,000 training points per channel
- `KNN`: up to 100,000 multivariate training points
- `Subsequence_KNN`: up to 100,000 training windows per channel

## Outputs

Each run writes to:

```text
results\<preset>\<timestamp>\
```

Files:

- `run.log`
- `run_config.json`
- `results.csv`
- `summary_key_metrics.csv`
- `summary_by_algorithm.csv`
- `summary_by_algorithm_channel.csv`

The anomaly feature analysis writes:

- `event_summary.csv`
- `feature_distribution.csv`
- `duration_bucket_distribution.csv`
- `channel_event_distribution.csv`
- `README.md`
- `README.zh.md`
- `figures/*.png`

The full visual report writes:

- `visual_report.md`
- `visual_report.zh.md`
- `00_preprocessed_data/*.csv`
- `01_dataset_overview/*.csv`
- `01_dataset_overview/figures/*.png`
- `02_representative_events/plot_index.csv`
- `02_representative_events/figures/*.png`

## Metric Roadmap

Current metrics:

- Global point-level `Point_AUROC`, `Point_AUPR`, `Point_F1_Top5`
- Compatibility aliases: `AUROC`, `AUPR`, `F1`
- Per-channel mean `ChannelMean_*`
- Lightweight ESA-style event metrics via `official_metrics.py`
- Optional VUS metrics via `--include-vus-metrics`:
  - `VUS_PR`
  - `VUS_ROC`

Planned metrics:

- Standard/PA/Event/R/Affiliation F-scores for metric sensitivity analysis

VUS metrics are off by default because the ESA test split has more than 2.5 million rows. The current implementation uses a local lightweight adaptation of TimeEval's VUS metric and supports max-bin reduction through `--vus-max-points`, preserving anomalous labels and high-score peaks in each consecutive bin.
