# Project Structure / 项目结构

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

## Training/Benchmarking / 训练与评测

Use `run_benchmark.py` or the notebooks in `notebooks/`.

中文：命令行使用 `run_benchmark.py`；交互式实验优先使用 `notebooks/`。

## Visualization / 可视化

Use scripts in `scripts/visualization/`:

- `analyze_anomaly_features.py`: anomaly feature distributions and summary figures
- `plot_anomaly_windows.py`: raw and z-score event window plots
- `build_visual_report.py`: complete bilingual visual report

中文：可视化脚本都在 `scripts/visualization/`，用于异常统计、事件窗口图和完整双语可视化报告。

## Utility Scripts / 工具脚本

Use scripts in `scripts/tools/`:

- `check_data.py`: verify expected data layout
- `make_algorithm_notebooks.py`: regenerate per-algorithm notebooks

中文：工具脚本位于 `scripts/tools/`，用于检查数据和重新生成 notebook。

## AutoDL / 云端运行

Use scripts in `scripts/autodl/`:

- `autodl_setup.sh`
- `run_autodl_light.sh`
