# ESA-ADB Lite Algorithm Notebooks

Each notebook launches one algorithm through `run_benchmark.py`.
Keep algorithm implementations in `algorithms.py`; use notebooks for interactive experiments and result inspection.

- [PCC](PCC.ipynb): Fast PCA reconstruction-error baseline.
- [HBOS](HBOS.ipynb): Fast per-channel histogram outlier score baseline.
- [STD](STD.ipynb): Fast mean/std threshold baseline.
- [iForest](iForest.ipynb): Strong per-channel Isolation Forest baseline; slower on target channels.
- [COPOD](COPOD.ipynb): Fast empirical-tail-probability baseline; promising in subset experiments.
- [RobustPCA](RobustPCA.ipynb): Robust-scaled PCA reconstruction-error baseline.
- [LOF](LOF.ipynb): Local Outlier Factor; heavy on the full ESA test split.
- [KNN](KNN.ipynb): Multivariate nearest-neighbor distance; heavy on large splits.
- [Subsequence_IF](Subsequence_IF.ipynb): Windowed Isolation Forest for subsequence anomalies; cloud recommended.
- [Subsequence_KNN](Subsequence_KNN.ipynb): Windowed kNN for subsequence anomalies; cloud recommended.
