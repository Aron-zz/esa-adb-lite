#!/usr/bin/env python3
"""ESA-ADB 轻量化基准测试：6 算法 × 5 数据集 × 2 通道组 = 60 实验。"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import _read_csv, select_channels, TARGET_CHANNELS, SUBSET_CHANNELS
from algorithms import ALGORITHMS
from metrics import evaluate

PREPROCESSED = Path(__file__).resolve().parent.parent / "data" / "preprocessed"
BASE = PREPROCESSED / "multivariate" / "ESA-Mission1-semi-supervised"
TRAIN_NAMES = ["3_months", "10_months", "21_months", "42_months", "84_months"]
CHANNEL_GROUPS = [
    ("Target_57", TARGET_CHANNELS),
    ("Subset_6",  SUBSET_CHANNELS),
]


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PREPROCESSED
    base = data_path / "multivariate" / "ESA-Mission1-semi-supervised"

    print(f"Loading test data ...")
    test_X, test_y, test_channels = _read_csv(base / "84_months.test.csv")

    results = []
    total = len(TRAIN_NAMES) * len(ALGORITHMS) * len(CHANNEL_GROUPS)
    n = 0

    for ds_name in TRAIN_NAMES:
        # 只加载一份训练集
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")
        train_X, train_y, train_channels = _read_csv(base / f"{ds_name}.train.csv")

        for ch_label, ch_list in CHANNEL_GROUPS:
            # 筛选通道
            tr_idx = [i for i, c in enumerate(train_channels) if c in ch_list]
            te_idx = [i for i, c in enumerate(test_channels) if c in ch_list]
            X_tr = train_X[:, tr_idx]
            X_te = test_X[:, te_idx]
            y_te = test_y[:, te_idx]

            for algo_name, algo_fn in ALGORITHMS.items():
                n += 1
                t0 = time.time()
                try:
                    scores = algo_fn(X_tr, X_te)
                    m = evaluate(y_te, scores)
                except Exception as e:
                    m = {"AUROC": None, "AUPR": None, "F1": None, "error": str(e)[:100]}

                elapsed = time.time() - t0
                results.append({
                    "dataset": ds_name,
                    "algorithm": algo_name,
                    "channels": ch_label,
                    "time_s": round(elapsed, 1),
                    **m,
                })
                status = "OK" if m.get("AUROC") is not None else "ERR"
                print(f"[{n:3d}/{total}] {algo_name:15s} {ds_name:10s} {ch_label:10s}  "
                      f"AUROC={m.get('AUROC','-'):>6}  {elapsed:.1f}s  [{status}]")

        # 释放当前训练集内存
        del train_X, train_y
        gc.collect()

    # 保存
    df = pd.DataFrame(results)
    out = "benchmark_results.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Done. {len(df)} rows → {out}")

    # 汇总
    print("\n=== Mean AUROC by algorithm ===")
    print(df.groupby("algorithm")[["AUROC", "AUPR", "F1", "time_s"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
