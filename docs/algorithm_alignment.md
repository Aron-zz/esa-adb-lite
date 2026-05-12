# Algorithm Alignment / 算法对齐说明

This document records how ESA-ADB Lite maps to the official ESA-ADB Mission1 classical benchmark.

本文档说明 ESA-ADB Lite 与官方 ESA-ADB Mission1 classical benchmark 的算法对应关系。

Official repository / 官方仓库：

```text
https://github.com/kplabs-pl/ESA-ADB
```

## Official Classical Set / 官方 Classical 算法组

The `official` preset in this repository is intended to match the classical Mission1 algorithm list used by the official repository, while keeping execution Docker-free.

本仓库的 `official` preset 目标是对齐官方 Mission1 classical 算法列表，同时保留免 Docker、可直接运行的实验入口。

| Official method | Lite name | Main parameters | Status | 中文说明 |
| --- | --- | --- | --- | --- |
| PCC | `PCC` | PCA reconstruction error, `n_components=2` | aligned lightweight implementation | PCA 重构误差，轻量实现 |
| HBOS | `HBOS` | `n_bins=50` | aligned lightweight implementation | 直方图离群分数，参数对齐 |
| STD | `STD3` | `tol=3` | aligned | 3 倍标准差阈值 |
| STD | `STD5` | `tol=5` | aligned | 5 倍标准差阈值 |
| Isolation Forest | `iForest` | per-channel Isolation Forest | aligned baseline, sklearn implementation | 逐通道 iForest，使用 sklearn 实现 |
| Subsequence Isolation Forest | `Subsequence_IF` | `window_size=17`, `n_estimators=200` | aligned but heavy | 滑窗 iForest，参数对齐但耗时较长 |
| KNN | `KNN` | `n_neighbors=5` | approximated for large CSVs with deterministic sampling | kNN 距离分数，大数据上使用确定性抽样近似 |

Compatibility alias / 兼容别名：

- `STD` remains available and is equivalent to `STD3`.
- `STD` 仍可使用，等价于 `STD3`。

## Lite Extensions / Lite 扩展算法

These algorithms are useful for our research comparison but are not part of the official Mission1 classical set above:

以下算法适合我们的扩展实验，但不属于上面的官方 Mission1 classical 主算法组：

| Lite name | Role | 中文说明 |
| --- | --- | --- |
| `COPOD` | Fast empirical-tail baseline | 快速经验尾概率基线 |
| `RobustPCA` | Robust-scaled PCA reconstruction baseline | 鲁棒缩放 PCA 重构误差 |
| `LOF` | Local density baseline, heavy on full data | 局部密度基线，完整数据上较重 |
| `Subsequence_KNN` | Windowed nearest-neighbor baseline | 滑窗 kNN 基线 |

## Recommended Runs / 推荐运行方式

Fast local validation / 本机快速验证：

```powershell
..\.venv\Scripts\python.exe run_benchmark.py ..\data\preprocessed --preset smoke --skip-official-metrics
```

Official-aligned subset run / 官方对齐 subset 实验：

```powershell
..\.venv\Scripts\python.exe run_benchmark.py ..\data\preprocessed `
  --preset official `
  --channel-groups subset `
  --skip-official-metrics `
  --include-vus-metrics `
  --vus-max-points 50000 `
  --vus-max-buffer 50 `
  --vus-max-thresholds 30
```

For the `target` group, `Subsequence_IF` and `KNN` can be slow. Run them separately if budget is limited.

对于 `target` 通道组，`Subsequence_IF` 和 `KNN` 可能较慢。如果预算有限，建议先单独补跑。
