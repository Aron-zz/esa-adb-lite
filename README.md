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
  - `KNN`
  - `Subsequence_IF`

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

Useful targeted runs:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py --datasets 3_months --channel-groups subset --algorithms HBOS STD
..\.venv\Scripts\python.exe run_benchmark.py --preset smoke --skip-official-metrics
..\.venv\Scripts\python.exe run_benchmark.py --preset smoke --skip-official-metrics --include-vus-metrics
..\.venv\Scripts\python.exe run_benchmark.py --dry-run
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

If memory is limited, start with:

```bash
python run_benchmark.py ../data/preprocessed \
  --preset light \
  --datasets 3_months 10_months 21_months 42_months \
  --channel-groups subset \
  --include-vus-metrics \
  --vus-max-points 50000
```

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
