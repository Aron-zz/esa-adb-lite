# Project Structure

```text
esa_adb_lite/
  algorithms.py                 # Algorithm implementations
  data_loader.py                # CSV loading and channel selection helpers
  metrics.py                    # Main metric dispatcher
  official_metrics.py           # ESA official/event metric adapter
  vus_metrics.py                # Lightweight VUS metrics
  run_benchmark.py              # Main CLI benchmark runner
  notebooks/                    # Per-algorithm interactive launchers
  scripts/
    autodl/                     # AutoDL setup/run helpers
    tools/                      # Utility scripts
    visualization/              # Visual analysis and report scripts
  docs/                         # Data and project documentation
```

## Training/Benchmarking

Use `run_benchmark.py` or the notebooks in `notebooks/`.

## Visualization

Use scripts in `scripts/visualization/`:

- `analyze_anomaly_features.py`: anomaly feature distributions and summary figures
- `plot_anomaly_windows.py`: raw and z-score event window plots
- `build_visual_report.py`: complete bilingual visual report

## Utility Scripts

Use scripts in `scripts/tools/`:

- `check_data.py`: verify expected data layout
- `make_algorithm_notebooks.py`: regenerate per-algorithm notebooks

## AutoDL

Use scripts in `scripts/autodl/`:

- `autodl_setup.sh`
- `run_autodl_light.sh`
