#!/usr/bin/env python3
"""ESA-ADB 轻量化基准测试。"""
import argparse
import gc
import inspect
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import _read_csv, TARGET_CHANNELS, SUBSET_CHANNELS, load_official_mission1_context
from algorithms import ALGORITHMS
from metrics import evaluate
from slice_metrics import evaluate_slices, load_slice_context
from vus_metrics import VUSConfig

TRAIN_NAMES = ["3_months", "10_months", "21_months", "42_months", "84_months"]
CHANNEL_GROUPS = {
    "target": ("Target_57", TARGET_CHANNELS),
    "subset": ("Subset_6", SUBSET_CHANNELS),
}
PRESETS = {
    "smoke": {
        "datasets": ["3_months"],
        "channel_groups": ["subset"],
        "algorithms": ["HBOS", "STD"],
    },
    "light": {
        "datasets": TRAIN_NAMES,
        "channel_groups": ["target", "subset"],
        "algorithms": ["PCC", "HBOS", "STD", "iForest"],
    },
    "extended": {
        "datasets": TRAIN_NAMES,
        "channel_groups": ["target", "subset"],
        "algorithms": ["PCC", "HBOS", "STD", "iForest", "COPOD", "RobustPCA"],
    },
    "official": {
        "datasets": TRAIN_NAMES,
        "channel_groups": ["target", "subset"],
        "algorithms": ["PCC", "HBOS", "STD3", "STD5", "iForest", "Subsequence_IF", "KNN"],
    },
    "full": {
        "datasets": TRAIN_NAMES,
        "channel_groups": ["target", "subset"],
        "algorithms": None,
    },
}
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


def setup_logging(run_dir):
    """双输出：控制台 + 文件。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"

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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent / "data" / "preprocessed"


def parse_args():
    parser = argparse.ArgumentParser(description="ESA-ADB Lite Benchmark")
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(DEFAULT_DATA),
        help="预处理数据根目录，默认使用仓库根目录下的 data/preprocessed",
    )
    parser.add_argument(
        "--preset",
        choices=["smoke", "light", "extended", "official", "full"],
        default="full",
        help=(
            "smoke: 最小验证；light: 4 个轻量算法；extended: 加入轻量扩展算法；"
            "official: 对齐 ESA-ADB Mission1 classical；full: 全部经典算法"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=TRAIN_NAMES,
        help="可选：显式指定训练 split，覆盖 --preset",
    )
    parser.add_argument(
        "--channel-groups",
        nargs="+",
        choices=sorted(CHANNEL_GROUPS),
        help="可选：显式指定通道组，覆盖 --preset。可选 target/subset",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="可选：显式指定要运行的算法名，覆盖 --preset",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="结果根目录，默认 results/<preset>/<timestamp>",
    )
    parser.add_argument(
        "--skip-official-metrics",
        action="store_true",
        help="只计算点级指标，跳过 ESA 官方事件指标",
    )
    parser.add_argument(
        "--include-vus-metrics",
        action="store_true",
        help="计算 TSB-AD/TimeEval 风格 VUS-PR/VUS-ROC，默认关闭",
    )
    parser.add_argument(
        "--vus-max-buffer",
        type=int,
        default=100,
        help="VUS 最大 buffer size，默认 100",
    )
    parser.add_argument(
        "--vus-max-thresholds",
        type=int,
        default=50,
        help="VUS 阈值采样数量，默认 50",
    )
    parser.add_argument(
        "--vus-max-points",
        type=int,
        default=200_000,
        help="VUS 最大点数；超过则均匀抽样，默认 200000；设为 0 表示不抽样",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="打印解析后的运行计划，不加载数据、不运行算法",
    )
    parser.add_argument(
        "--skip-slice-metrics",
        action="store_true",
        help="跳过异常类型切片指标 slice_metrics.csv，默认开启",
    )
    return parser.parse_args()


def resolve_algorithms(args):
    if args.algorithms:
        unknown = [name for name in args.algorithms if name not in ALGORITHMS]
        if unknown:
            raise ValueError(f"Unknown algorithms: {', '.join(unknown)}")
        return {name: ALGORITHMS[name] for name in args.algorithms}

    preset_algorithms = PRESETS[args.preset]["algorithms"]
    if preset_algorithms is not None:
        return {name: ALGORITHMS[name] for name in preset_algorithms}

    return ALGORITHMS


def resolve_datasets(args):
    return args.datasets if args.datasets else PRESETS[args.preset]["datasets"]


def resolve_channel_groups(args):
    names = args.channel_groups if args.channel_groups else PRESETS[args.preset]["channel_groups"]
    return [CHANNEL_GROUPS[name] for name in names]


def make_run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(args.output_dir) / args.preset / timestamp


def write_run_config(run_dir, args, datasets, channel_groups, selected_algorithms, data_path):
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "data_path": str(data_path),
        "datasets": datasets,
        "channel_groups": [label for label, _ in channel_groups],
        "algorithms": list(selected_algorithms),
        "include_official_metrics": not args.skip_official_metrics,
        "include_slice_metrics": not args.skip_slice_metrics,
        "include_vus_metrics": args.include_vus_metrics,
        "vus": {
            "max_buffer": args.vus_max_buffer,
            "max_thresholds": args.vus_max_thresholds,
            "max_points": args.vus_max_points,
            "reduction": "max_bin",
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config


def main():
    args = parse_args()
    selected_datasets = resolve_datasets(args)
    selected_channel_groups = resolve_channel_groups(args)
    selected_algorithms = resolve_algorithms(args)
    run_dir = make_run_dir(args)
    log_file = setup_logging(run_dir)
    t_total = time.time()

    data_path = Path(args.data_path)
    config = write_run_config(
        run_dir,
        args,
        selected_datasets,
        selected_channel_groups,
        selected_algorithms,
        data_path,
    )

    logging.info("=" * 50)
    logging.info("ESA-ADB Lite Benchmark")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"结果目录: {run_dir}")
    logging.info(f"运行预设: {args.preset}")
    logging.info(f"算法: {', '.join(selected_algorithms)}")
    logging.info(f"数据集: {', '.join(selected_datasets)}")
    logging.info(f"通道组: {', '.join(config['channel_groups'])}")
    logging.info(f"官方事件指标: {'ON' if not args.skip_official_metrics else 'OFF'}")
    logging.info(f"VUS 指标: {'ON' if args.include_vus_metrics else 'OFF'}")
    logging.info("=" * 50)

    if args.dry_run:
        logging.info("dry-run: 仅输出运行计划，不加载数据。")
        return

    base = data_path / "multivariate" / "ESA-Mission1-semi-supervised"

    # ---- 加载测试集 ----
    test_csv = base / "84_months.test.csv"
    check_resources(test_csv, "test data")
    logging.info(f"加载测试集: {test_csv}")
    test_X, test_y, test_channels, test_timestamps = _read_csv(test_csv, return_timestamps=True)
    labels_df = test_data_scores = subsystems_mapping = None
    if not args.skip_official_metrics:
        labels_df, test_data_scores, subsystems_mapping = load_official_mission1_context(test_csv)
    logging.info(f"  shape={test_X.shape}, 通道数={test_X.shape[1]}")

    results = []
    slice_results = []
    slice_context_cache = {}
    total = len(selected_datasets) * len(selected_algorithms) * len(selected_channel_groups)
    n = 0

    for ds_name in selected_datasets:
        train_csv = base / f"{ds_name}.train.csv"
        check_resources(train_csv, ds_name)
        logging.info(f"加载: {ds_name}  文件大小={os.path.getsize(train_csv)/(1024**3):.1f}GB")
        t_ds = time.time()

        try:
            train_X, train_y, train_channels = _read_csv(train_csv)
        except MemoryError:
            logging.error(f"{ds_name} 内存不足，跳过此数据集")
            n += len(selected_algorithms) * len(selected_channel_groups)
            continue

        logging.info(f"  shape={train_X.shape}, 加载耗时 {time.time()-t_ds:.1f}s")

        for ch_label, ch_list in selected_channel_groups:
            tr_idx = [i for i, c in enumerate(train_channels) if c in ch_list]
            te_idx = [i for i, c in enumerate(test_channels)  if c in ch_list]
            if not tr_idx or not te_idx:
                logging.warning(f"{ds_name} {ch_label}: 未找到匹配通道，跳过")
                n += len(selected_algorithms)
                continue
            X_tr = train_X[:, tr_idx]
            y_tr = train_y[:, tr_idx]
            X_te = test_X[:, te_idx]
            y_te = test_y[:, te_idx]
            channel_names = [test_channels[i] for i in te_idx]
            slice_events = None
            if not args.skip_slice_metrics:
                cache_key = tuple(channel_names)
                if cache_key not in slice_context_cache:
                    slice_context_cache[cache_key] = load_slice_context(test_csv, test_timestamps, channel_names)
                slice_events = slice_context_cache[cache_key]

            for algo_name, algo_fn in selected_algorithms.items():
                n += 1
                t0 = time.time()
                try:
                    kwargs = {}
                    if "y_train" in inspect.signature(algo_fn).parameters:
                        kwargs["y_train"] = y_tr
                    scores = algo_fn(X_tr, X_te, **kwargs)
                    m = evaluate(
                        y_te,
                        scores,
                        timestamps=test_timestamps,
                        channel_names=channel_names,
                        labels_df=labels_df,
                        test_data_scores=test_data_scores,
                        subsystems_mapping=subsystems_mapping,
                        include_official=not args.skip_official_metrics,
                        include_vus=args.include_vus_metrics,
                        vus_config=VUSConfig(
                            max_buffer_size=args.vus_max_buffer,
                            max_threshold_samples=args.vus_max_thresholds,
                            max_points=args.vus_max_points,
                        ),
                    )
                except Exception as e:
                    m = {"AUROC": None, "AUPR": None, "F1": None}
                    logging.warning(f"[{n:3d}/{total}] {algo_name} {ds_name} {ch_label}  ERROR: {e!r}")

                elapsed = time.time() - t0
                if m.get("AUROC") is not None and not args.skip_slice_metrics:
                    slice_results.extend(
                        evaluate_slices(
                            y_true=y_te,
                            y_scores=scores,
                            timestamps=test_timestamps,
                            slice_events=slice_events,
                            dataset=ds_name,
                            algorithm=algo_name,
                            channels=ch_label,
                        )
                    )
                results.append({
                    "dataset": ds_name,
                    "algorithm": algo_name,
                    "channels": ch_label,
                    "time_s": round(elapsed, 1),
                    **m,
                })
                status = "OK" if m.get("AUROC") is not None else "ERR"
                auroc_str = f"{m.get('AUROC', '-'):>7}" if m.get('AUROC') is not None else "     -"
                logging.info(f"[{n:3d}/{total}] {algo_name:14s} {ds_name:9s} {ch_label:10s}"
                             f"  AUROC={auroc_str}  {elapsed:.1f}s  [{status}]")

        del train_X, train_y
        gc.collect()
        logging.info(f"  {ds_name} 子集完成，耗时 {time.time()-t_ds:.1f}s")

    # ---- 保存 ----
    df = pd.DataFrame(results)
    out = run_dir / "results.csv"
    df.to_csv(out, index=False)
    logging.info(f"保存 {len(df)} 行结果到 {out}")

    numeric_cols = [c for c in df.columns if c not in {"dataset", "algorithm", "channels"}]
    summary_by_algorithm = df.groupby("algorithm")[numeric_cols].mean(numeric_only=True).round(4)
    summary_by_algorithm_channel = df.groupby(["algorithm", "channels"])[numeric_cols].mean(numeric_only=True).round(4)
    summary_by_algorithm.to_csv(run_dir / "summary_by_algorithm.csv")
    summary_by_algorithm_channel.to_csv(run_dir / "summary_by_algorithm_channel.csv")

    key_cols = [
        "dataset",
        "algorithm",
        "channels",
        "time_s",
        "Point_AUPR",
        "Point_AUROC",
        "Point_F1_Top5",
        "VUS_PR",
        "VUS_ROC",
        "Global_Anomaly_AFF_F_0.50",
        "Global_Rare Event_Anomaly_AFF_F_0.50",
        "PC_Anomaly_F",
        "PC_Rare Event_Anomaly_F",
    ]
    present_key_cols = [col for col in key_cols if col in df.columns]
    if present_key_cols:
        df[present_key_cols].to_csv(run_dir / "summary_key_metrics.csv", index=False)

    if slice_results:
        slice_df = pd.DataFrame(slice_results)
        slice_out = run_dir / "slice_metrics.csv"
        slice_df.to_csv(slice_out, index=False)
        logging.info(f"保存 {len(slice_df)} 行切片指标到 {slice_out}")

        slice_numeric_cols = [
            c for c in slice_df.columns
            if c not in {"dataset", "algorithm", "channels", "slice_type", "slice_value"}
        ]
        slice_summary = (
            slice_df
            .groupby(["algorithm", "channels", "slice_type", "slice_value"])[slice_numeric_cols]
            .mean(numeric_only=True)
            .round(4)
            .reset_index()
        )
        slice_summary.to_csv(run_dir / "summary_slice_metrics.csv", index=False)

    logging.info("\n=== Mean metrics by algorithm ===")
    logging.info("\n" + summary_by_algorithm.to_string())

    logging.info(f"\n总耗时: {(time.time()-t_total)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
