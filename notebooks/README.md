# ESA-ADB Lite Algorithm Notebooks

EN: Each notebook launches one algorithm through `run_benchmark.py`.
中文：每个 notebook 都通过 `run_benchmark.py` 启动一个算法实验。

EN: Keep algorithm implementations in `algorithms.py`; use notebooks for interactive experiments and result inspection.
中文：算法实现保留在 `algorithms.py`，notebook 用于交互式实验和查看结果。

## Switch Guide / 开关指南

- `DATASETS`: training splits / 训练集 split
- `CHANNEL_GROUPS`: `subset` or `target` / 通道组
- `INCLUDE_VUS`: interval-aware VUS metrics / 区间友好的 VUS 指标
- `SKIP_OFFICIAL_METRICS`: skip official ESA event metrics / 跳过官方 ESA 事件指标
- `SKIP_SLICE_METRICS`: skip anomaly-type slice metrics / 跳过异常类型切片指标

## Notebooks / 算法 Notebook

- [PCC](PCC.ipynb): Fast PCA reconstruction-error baseline. / 快速 PCA 重构误差基线。
- [HBOS](HBOS.ipynb): Fast per-channel histogram outlier score baseline. / 快速逐通道直方图异常分数基线。
- [STD](STD.ipynb): Fast mean/std threshold baseline. / 快速均值/标准差阈值基线。
- [iForest](iForest.ipynb): Strong per-channel Isolation Forest baseline; slower on target channels. / 较强的逐通道 Isolation Forest 基线；target 通道组上较慢。
- [COPOD](COPOD.ipynb): Fast empirical-tail-probability baseline; promising in subset experiments. / 快速经验尾概率基线；subset 实验表现有潜力。
- [RobustPCA](RobustPCA.ipynb): Robust-scaled PCA reconstruction-error baseline. / 鲁棒缩放后的 PCA 重构误差基线。
- [LOF](LOF.ipynb): Local Outlier Factor; heavy on the full ESA test split. / 局部离群因子；完整 ESA 测试集上较重。
- [KNN](KNN.ipynb): Multivariate nearest-neighbor distance; heavy on large splits. / 多变量近邻距离；大 split 上较重。
- [Subsequence_IF](Subsequence_IF.ipynb): Windowed Isolation Forest for subsequence anomalies; cloud recommended. / 面向子序列异常的窗口 Isolation Forest；建议云端运行。
- [Subsequence_KNN](Subsequence_KNN.ipynb): Windowed kNN for subsequence anomalies; cloud recommended. / 面向子序列异常的窗口 kNN；建议云端运行。
