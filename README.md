# ESA-ADB Lite

Docker-free, lightweight ESA-ADB Mission1 benchmark for classical time-series anomaly detection and visual analysis.

This repository keeps the core of the official ESA-ADB Mission1 setup while avoiding the heavy TimeEval/Docker runtime for day-to-day experiments. Official repository:

```text
https://github.com/kplabs-pl/ESA-ADB
```

## Project Layout

```text
esa_adb_lite/
  algorithms.py                 # Algorithm implementations
  data_loader.py                # CSV loading and channel helpers
  metrics.py                    # Point/channel metric dispatcher
  official_metrics.py           # ESA official metric adapter
  vus_metrics.py                # Lightweight VUS metrics
  run_benchmark.py              # Main benchmark CLI
  notebooks/                    # One notebook per algorithm
  scripts/
    autodl/                     # AutoDL setup/run helpers
    tools/                      # Data check and notebook generation
    visualization/              # Analysis and visual report scripts
  docs/
    data_preparation.md         # Data source and preparation
    notebook_usage.md           # Notebook-first workflow
    project_structure.md        # File structure notes
```

## Scope

- Mission: `ESA-Mission1`
- Train splits: `3_months`, `10_months`, `21_months`, `42_months`, `84_months`
- Test split: `84_months.test.csv`
- Channel groups:
  - `subset`: `channel_41` to `channel_46`
  - `target`: official 57 target channels
- Algorithms:
  - `PCC`, `HBOS`, `STD`, `iForest`
  - `COPOD`, `RobustPCA`
  - `LOF`, `KNN`, `Subsequence_IF`, `Subsequence_KNN`

`LOF`, `KNN`, `Subsequence_IF`, and `Subsequence_KNN` are available but computationally heavy on the full ESA test split. Prefer explicit targeted or AutoDL runs for them.

## Data

Data files are not included in this repository. See [docs/data_preparation.md](docs/data_preparation.md).

Expected local layout:

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

Two practical data paths:

- Use a shared archive of already preprocessed Mission1 data for fast local/AutoDL experiments.
- Reproduce preprocessing from the official ESA-ADB repository for full end-to-end reproducibility.

## Setup

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify data:

```powershell
..\.venv\Scripts\python.exe scripts\tools\check_data.py `
  --data-path ..\data\preprocessed `
  --mission-root ..\data\esa-adb\mission1\ESA-Mission1
```

## Benchmark Runs

Smoke test:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py ..\data\preprocessed --preset smoke --skip-official-metrics
```

Light benchmark:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py ..\data\preprocessed --preset light --skip-official-metrics
```

Extended benchmark:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py ..\data\preprocessed --preset extended --skip-official-metrics
```

Targeted run:

```powershell
..\.venv\Scripts\python.exe run_benchmark.py ..\data\preprocessed `
  --datasets 3_months `
  --channel-groups subset `
  --algorithms COPOD RobustPCA `
  --skip-official-metrics
```

Per-algorithm notebooks are in `notebooks/`. They call `run_benchmark.py`, so notebook and CLI experiments share the same implementation.

Notebook usage guide:

```text
docs/notebook_usage.md
```

Regenerate notebooks:

```powershell
..\.venv\Scripts\python.exe scripts\tools\make_algorithm_notebooks.py
```

## Visualization

Anomaly feature summary:

```powershell
..\.venv\Scripts\python.exe scripts\visualization\analyze_anomaly_features.py `
  --mission-root ..\data\esa-adb\mission1\ESA-Mission1 `
  --results-csv results\results.csv `
  --output-dir results\anomaly_feature_analysis_target
```

Event-window plots:

```powershell
..\.venv\Scripts\python.exe scripts\visualization\plot_anomaly_windows.py `
  --event-ids id_118 id_126 `
  --max-channels-per-event 4 `
  --context-hours 12 `
  --output-dir results\anomaly_windows_examples
```

Full bilingual visual report:

```powershell
..\.venv\Scripts\python.exe scripts\visualization\build_visual_report.py `
  --data-path ..\data\preprocessed `
  --mission-root ..\data\esa-adb\mission1\ESA-Mission1 `
  --results-csv results\results.csv `
  --output-dir results\visual_report
```

The visual report includes:

- English and Chinese markdown summaries
- anomaly feature distributions
- duration/channel-count analysis
- monthly event timeline
- subsystem/category heatmaps
- raw and z-score event-window plots

## AutoDL

Recommended setup:

```bash
cd /root/esa-adb-lite
bash scripts/autodl/autodl_setup.sh
python scripts/tools/check_data.py \
  --data-path /root/autodl-tmp/data/preprocessed \
  --mission-root /root/autodl-tmp/data/esa-adb/mission1/ESA-Mission1
```

Recommended extended run:

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

For `target` runs, avoid repeating algorithms already completed. Use `--algorithms` to run only missing algorithms, for example:

```bash
python run_benchmark.py /root/autodl-tmp/data/preprocessed \
  --preset extended \
  --output-dir /root/autodl-tmp/results_autodl \
  --channel-groups target \
  --algorithms COPOD RobustPCA \
  --skip-official-metrics \
  --include-vus-metrics \
  --vus-max-points 50000 \
  --vus-max-buffer 50 \
  --vus-max-thresholds 30
```

## Outputs

Benchmark runs write to:

```text
results/<preset>/<timestamp>/
```

Typical files:

- `run.log`
- `run_config.json`
- `results.csv`
- `slice_metrics.csv`
- `summary_key_metrics.csv`
- `summary_by_algorithm.csv`
- `summary_by_algorithm_channel.csv`
- `summary_slice_metrics.csv`

Generated outputs are ignored by Git. Commit code, notebooks, and documentation only.

## Metrics

- `metrics.py`: point-level and channel-mean metrics, plus dispatcher for optional metrics
- `slice_metrics.py`: anomaly-type slice metrics by category, length, locality, dimensionality, class, duration, and affected-channel count
- `vus_metrics.py`: lightweight VUS-PR/VUS-ROC for subsequence-aware evaluation
- `official_metrics.py`: adapter for ESA official/event metrics when official dependencies are available

Recommended experiment view:

```text
Point metrics -> channel means -> VUS interval metrics -> official ESA event metrics
```

Slice metrics are enabled by default and can be disabled with:

```powershell
--skip-slice-metrics
```

Summarize one or more slice metric files:

```powershell
..\.venv\Scripts\python.exe scripts\tools\summarize_slice_metrics.py `
  results\<preset>\<timestamp>\slice_metrics.csv `
  --metric Point_AUPR `
  --output-dir results\slice_analysis
```
