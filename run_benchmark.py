#!/usr/bin/env python3
"""ESA-ADB 轻量化基准测试：6 算法 × 5 数据集 × 2 通道组 = 60 实验。"""
import gc
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import _read_csv, TARGET_CHANNELS, SUBSET_CHANNELS
from algorithms import ALGORITHMS
from metrics import evaluate

TRAIN_NAMES = ["3_months", "10_months", "21_months", "42_months", "84_months"]
CHANNEL_GROUPS = [
    ("Target_57", TARGET_CHANNELS),
    ("Subset_6",  SUBSET_CHANNELS),
]
RAM_WARN_GB = 2.0       # 低于此值警告
DISK_WARN_GB = 1.0      # 低于此值警告


def get_mem_free_gb():
    """返回可用内存 (GB)。"""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        return None


def get_disk_free_gb(path):
    """返回路径所在磁盘剩余空间 (GB)。"""
    return shutil.disk_usage(path).free / (1024**3)


def estimate_csv_mem(csv_path):
    """根据 CSV 文件大小估算加载后的内存占用 (GB)。"""
    file_size_gb = os.path.getsize(csv_path) / (1024**3)
    # 经验值：CSV 文本解析为 float32 后约占原大小的 0.3~0.5
    return file_size_gb * 0.5


def check_resources(csv_path, label):
    """检查内存和磁盘空间，不足时警告/退出。"""
    needed = estimate_csv_mem(csv_path)
    mem_free = get_mem_free_gb()
    disk_free = get_disk_free_gb(csv_path.parent)

    if mem_free is not None:
        if mem_free < needed + RAM_WARN_GB:
            logging.warning(f"内存紧张！可用 {mem_free:.1f}GB，预计需要 {needed:.1f}GB")
        if mem_free < needed * 0.5:
            logging.error(f"内存不足！终止运行。可用 {mem_free:.1f}GB < 需要 {needed:.1f}GB")
            sys.exit(1)
        logging.info(f"  {label}: 内存可用 {mem_free:.1f}GB，预计需要 {needed:.1f}GB")

    if disk_free < DISK_WARN_GB:
        logging.warning(f"磁盘空间不足 {disk_free:.1f}GB")


def setup_logging():
    """双输出：控制台 + 文件。"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"bench_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def main():
    log_file = setup_logging()
    t_total = time.time()

    logging.info("=" * 50)
    logging.info("ESA-ADB Lite Benchmark")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"算法: {', '.join(ALGORITHMS)}")
    logging.info(f"数据集: {', '.join(TRAIN_NAMES)}")
    logging.info("=" * 50)

    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/preprocessed")
    base = data_path / "multivariate" / "ESA-Mission1-semi-supervised"

    # ---- 加载测试集 ----
    test_csv = base / "84_months.test.csv"
    check_resources(test_csv, "test data")
    logging.info(f"加载测试集: {test_csv}")
    test_X, test_y, test_channels = _read_csv(test_csv)
    logging.info(f"  shape={test_X.shape}, 通道数={test_X.shape[1]}")

    results = []
    total = len(TRAIN_NAMES) * len(ALGORITHMS) * len(CHANNEL_GROUPS)
    n = 0

    for ds_name in TRAIN_NAMES:
        train_csv = base / f"{ds_name}.train.csv"
        check_resources(train_csv, ds_name)
        logging.info(f"加载: {ds_name}  文件大小={os.path.getsize(train_csv)/(1024**3):.1f}GB")
        t_ds = time.time()

        try:
            train_X, train_y, train_channels = _read_csv(train_csv)
        except MemoryError:
            logging.error(f"{ds_name} 内存不足，跳过此数据集")
            n += len(ALGORITHMS) * len(CHANNEL_GROUPS)
            continue

        logging.info(f"  shape={train_X.shape}, 加载耗时 {time.time()-t_ds:.1f}s")

        for ch_label, ch_list in CHANNEL_GROUPS:
            tr_idx = [i for i, c in enumerate(train_channels) if c in ch_list]
            te_idx = [i for i, c in enumerate(test_channels)  if c in ch_list]
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
                    m = {"AUROC": None, "AUPR": None, "F1": None}
                    logging.warning(f"[{n:3d}/{total}] {algo_name} {ds_name} {ch_label}  ERROR: {e!r}")

                elapsed = time.time() - t0
                results.append({
                    "dataset": ds_name,
                    "algorithm": algo_name,
                    "channels": ch_label,
                    "time_s": round(elapsed, 1),
                    **m,
                })
                status = "OK" if m.get("AUROC") is not None else "ERR"
                logging.info(f"[{n:3d}/{total}] {algo_name:14s} {ds_name:9s} {ch_label:10s}"
                             f"  AUROC={m.get('AUROC','-'):>7}  {elapsed:.1f}s  [{status}]")

        del train_X, train_y
        gc.collect()
        logging.info(f"  {ds_name} 子集完成，耗时 {time.time()-t_ds:.1f}s")

    # ---- 保存 ----
    df = pd.DataFrame(results)
    out = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out, index=False)
    logging.info(f"✅ {len(df)} 行结果 → {out}")

    logging.info("\n=== Mean AUROC by algorithm ===")
    logging.info("\n" + df.groupby("algorithm")[["AUROC", "AUPR", "F1", "time_s"]]
                  .mean().round(4).to_string())

    logging.info(f"\n总耗时: {(time.time()-t_total)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
