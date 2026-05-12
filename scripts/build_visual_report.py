#!/usr/bin/env python3
"""Build a compact visual report for ESA-ADB Mission1."""
import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_anomaly_features import (
    duration_buckets,
    event_summary,
    feature_distribution,
    read_mission_tables,
    save_bar,
    save_heatmap,
)
from plot_anomaly_windows import (
    choose_channels,
    first_last_timestamp,
    load_metadata,
    plot_event,
)


TRAIN_NAMES = ["3_months", "10_months", "21_months", "42_months", "84_months"]
FEATURE_SAMPLES = [
    ("Length", "Point"),
    ("Length", "Subsequence"),
    ("Category", "Anomaly"),
    ("Category", "Rare Event"),
    ("Locality", "Global"),
    ("Locality", "Local"),
    ("Dimensionality", "Multivariate"),
    ("Dimensionality", "Univariate"),
]


class PlotArgs:
    def __init__(self, max_channels_per_event, channels=None):
        self.max_channels_per_event = max_channels_per_event
        self.channels = channels


def parse_args():
    parser = argparse.ArgumentParser(description="Build ESA-ADB Mission1 visual report")
    parser.add_argument(
        "--data-path",
        default="../data/preprocessed",
        help="Preprocessed data root containing multivariate/ESA-Mission1-semi-supervised",
    )
    parser.add_argument(
        "--mission-root",
        default="../data/esa-adb/mission1/ESA-Mission1",
        help="Path containing labels.csv, anomaly_types.csv, channels.csv",
    )
    parser.add_argument("--results-csv", help="Optional benchmark results.csv")
    parser.add_argument("--output-dir", default="results/visual_report", help="Report output directory")
    parser.add_argument("--max-events", type=int, default=12, help="Maximum representative events to plot")
    parser.add_argument("--max-channels-per-event", type=int, default=4)
    parser.add_argument("--context-hours", type=float, default=12.0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser.parse_args()


def data_base(data_path):
    return Path(data_path) / "multivariate" / "ESA-Mission1-semi-supervised"


def read_csv_header(csv_path):
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    value_cols = [c for c in columns if c != "timestamp" and not c.startswith("is_anomaly_")]
    label_cols = [c for c in columns if c.startswith("is_anomaly_")]
    return columns, value_cols, label_cols


def load_metadata_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list) and payload:
        return payload[0]
    return payload if isinstance(payload, dict) else {}


def summarize_preprocessed_data(base):
    rows = []
    test_csv = base / "84_months.test.csv"
    _, value_cols, label_cols = read_csv_header(test_csv)

    for name in TRAIN_NAMES:
        csv_path = base / f"{name}.train.csv"
        metadata = load_metadata_json(base / f"{name}.metadata.json")
        rows.append({
            "split": name,
            "role": "train",
            "csv_file": csv_path.name,
            "size_gb": round(csv_path.stat().st_size / (1024 ** 3), 3),
            "rows": metadata.get("length"),
            "value_columns": len(value_cols),
            "label_columns": len(label_cols),
            "dimensions": metadata.get("dimensions"),
            "is_train_metadata": metadata.get("is_train"),
        })

    rows.append({
        "split": "84_months",
        "role": "test",
        "csv_file": test_csv.name,
        "size_gb": round(test_csv.stat().st_size / (1024 ** 3), 3),
        "rows": None,
        "value_columns": len(value_cols),
        "label_columns": len(label_cols),
        "dimensions": len(value_cols),
        "is_train_metadata": False,
    })
    return pd.DataFrame(rows), value_cols, label_cols


def select_representative_events(events, max_events):
    selected = []
    used = set()

    def add_event(df):
        for event_id in df.sort_values(["StartTime", "ID"])["ID"]:
            if event_id not in used:
                used.add(event_id)
                selected.append(event_id)
                return

    for feature, value in FEATURE_SAMPLES:
        add_event(events[events[feature] == value])

    top_classes = events["Class"].value_counts().head(4).index.tolist()
    for cls in top_classes:
        add_event(events[events["Class"] == cls])

    if len(selected) < max_events:
        add_event(events)

    return events[events["ID"].isin(selected[:max_events])].copy()


def create_overview(labels, output_dir):
    overview_dir = output_dir / "01_dataset_overview"
    figures_dir = overview_dir / "figures"
    overview_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    events = event_summary(labels)
    distributions = feature_distribution(events)
    buckets = duration_buckets(events)

    events.to_csv(overview_dir / "event_summary.csv", index=False)
    distributions.to_csv(overview_dir / "feature_distribution.csv", index=False)
    buckets.to_csv(overview_dir / "duration_bucket_distribution.csv", index=False)

    for feature in ["Category", "Dimensionality", "Locality", "Length"]:
        part = distributions[distributions["Feature"] == feature]
        save_bar(part, "Value", "EventCount", f"{feature} distribution", figures_dir / f"{feature.lower()}_distribution.png")
    save_bar(
        distributions[distributions["Feature"] == "Class"],
        "Value",
        "EventCount",
        "Top anomaly classes",
        figures_dir / "top_class_distribution.png",
        horizontal=True,
        top_n=12,
    )
    save_bar(buckets, "DurationBucket", "EventCount", "Event duration buckets", figures_dir / "duration_bucket_distribution.png")
    save_heatmap(events, "Category", "Length", figures_dir / "category_by_length_heatmap.png").to_csv(
        overview_dir / "category_by_length.csv"
    )
    save_heatmap(events, "Dimensionality", "Locality", figures_dir / "dimensionality_by_locality_heatmap.png").to_csv(
        overview_dir / "dimensionality_by_locality.csv"
    )
    return events, distributions


def create_result_chart(results_csv, output_dir):
    if not results_csv:
        return None
    path = Path(results_csv)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    metric = "Point_AUPR" if "Point_AUPR" in df.columns else "AUPR"
    if metric not in df.columns:
        return None
    summary = df.groupby(["algorithm", "channels"], dropna=False)[metric].mean().reset_index()
    figures_dir = output_dir / "04_algorithm_overview" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    ax = summary.pivot(index="algorithm", columns="channels", values=metric).plot(kind="bar", figsize=(8, 4.5))
    ax.set_title(f"Overall mean {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figures_dir / "overall_algorithm_metric.png", dpi=160)
    fig.clear()
    summary.to_csv(output_dir / "04_algorithm_overview" / "overall_algorithm_metric_summary.csv", index=False)
    return metric


def create_event_plots(test_csv, mission_root, events, output_dir, args):
    labels_raw, _ = load_metadata(mission_root)
    test_start, test_end = first_last_timestamp(test_csv)
    candidates = events[(events["EndTime"] >= test_start) & (events["StartTime"] <= test_end)].copy()
    selected = select_representative_events(candidates, args.max_events)
    plot_args = PlotArgs(args.max_channels_per_event)
    figures_dir = output_dir / "02_representative_events" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    context = pd.Timedelta(hours=args.context_hours)
    specs = []
    for event in selected.to_dict("records"):
        channels = choose_channels(labels_raw, event["ID"], plot_args)
        window_start = max(test_start, event["StartTime"] - context)
        window_end = min(test_end, event["EndTime"] + context)
        specs.append({
            "event": event,
            "channels": channels,
            "window_start": window_start,
            "window_end": window_end,
            "frames": [],
        })

    needed_cols = {"timestamp"}
    for spec in specs:
        for channel in spec["channels"]:
            needed_cols.add(channel)
            needed_cols.add(f"is_anomaly_{channel}")

    for chunk in pd.read_csv(test_csv, usecols=lambda col: col in needed_cols, chunksize=args.chunk_size):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        for spec in specs:
            mask = (chunk["timestamp"] >= spec["window_start"]) & (chunk["timestamp"] <= spec["window_end"])
            if mask.any():
                cols = ["timestamp"]
                for channel in spec["channels"]:
                    cols.append(channel)
                    label_col = f"is_anomaly_{channel}"
                    if label_col in chunk.columns:
                        cols.append(label_col)
                spec["frames"].append(chunk.loc[mask, cols].copy())

    index_rows = []
    for spec in specs:
        event = spec["event"]
        channels = spec["channels"]
        if not spec["frames"]:
            continue
        df = pd.concat(spec["frames"], ignore_index=True)
        if df.empty:
            continue
        filename = f"{event['ID']}_{event['Category']}_{event['Length']}.png"
        out = figures_dir / filename
        plot_event(df, labels_raw, event, channels, out)
        index_rows.append({
            "ID": event["ID"],
            "Category": event["Category"],
            "Dimensionality": event["Dimensionality"],
            "Locality": event["Locality"],
            "Length": event["Length"],
            "Class": event["Class"],
            "Subclass": event["Subclass"],
            "Channels": ",".join(channels),
            "WindowStart": spec["window_start"],
            "WindowEnd": spec["window_end"],
            "Figure": f"02_representative_events/figures/{filename}",
            "RowsPlotted": len(df),
        })
    index = pd.DataFrame(index_rows)
    index.to_csv(output_dir / "02_representative_events" / "plot_index.csv", index=False)
    return index


def _distribution_text(distributions, feature):
    part = distributions[distributions["Feature"] == feature].head(8)
    return ", ".join(f"{item.Value}={item.EventCount}" for item in part.itertuples())


def write_report_en(output_dir, preprocessed_summary, value_cols, label_cols, events, distributions, plot_index, result_metric):
    lines = [
        "# ESA-ADB Mission1 Visual Report",
        "",
        "## Preprocessed Data",
        "",
        "The preprocessed Mission1 files are wide time-series CSV tables.",
        "Each row is one timestamp. Value columns contain normalized telemetry or telecommand values, and each value column has a matching binary `is_anomaly_*` label column.",
        "",
        f"- Value columns: {len(value_cols)}",
        f"- Label columns: {len(label_cols)}",
        f"- Example value columns: {', '.join(value_cols[:8])}",
        f"- Example label columns: {', '.join(label_cols[:4])}",
        "",
        "Files:",
        "",
    ]
    for row in preprocessed_summary.itertuples(index=False):
        rows_text = "unknown" if pd.isna(row.rows) else str(int(row.rows))
        lines.append(f"- `{row.csv_file}`: {row.role}, {row.size_gb} GB, rows={rows_text}, dimensions={row.dimensions}")

    lines.extend([
        "",
        "## Anomaly Slices",
        "",
        f"- Events: {len(events)}",
        f"- Median duration hours: {events['DurationHours'].median():.3f}",
        f"- Median affected channels: {events['ChannelCount'].median():.1f}",
        "",
    ])
    for feature in ["Category", "Dimensionality", "Locality", "Length", "Class", "Subclass"]:
        lines.append(f"- {feature}: {_distribution_text(distributions, feature)}")

    lines.extend([
        "",
        "## Figures",
        "",
        "- `01_dataset_overview/figures/category_distribution.png`",
        "- `01_dataset_overview/figures/length_distribution.png`",
        "- `01_dataset_overview/figures/dimensionality_by_locality_heatmap.png`",
        "- `01_dataset_overview/figures/duration_bucket_distribution.png`",
        "- `02_representative_events/figures/*.png`",
        "",
        "## Representative Events",
        "",
    ])
    for row in plot_index.itertuples(index=False):
        lines.append(f"- `{row.ID}`: {row.Category}, {row.Dimensionality}, {row.Locality}, {row.Length}, {row.Class}, `{row.Figure}`")

    if result_metric:
        lines.extend([
            "",
            "## Algorithm Overview",
            "",
            f"- Overall benchmark chart uses `{result_metric}` from the supplied `results.csv`.",
            "- This is still a whole-test summary, not per-event score visualization.",
            "- The next research step is to save event-level score curves and build algorithm-overlaid plots.",
        ])

    (output_dir / "visual_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_report_zh(output_dir, preprocessed_summary, value_cols, label_cols, events, distributions, plot_index, result_metric):
    lines = [
        "# ESA-ADB Mission1 可视化报告",
        "",
        "## 预处理数据",
        "",
        "Mission1 的预处理文件是宽表形式的时序 CSV。",
        "每一行对应一个时间戳；数值列保存归一化后的遥测/遥控量，每个数值列都有一个对应的二值异常标签列 `is_anomaly_*`。",
        "",
        f"- 数值列数量: {len(value_cols)}",
        f"- 标签列数量: {len(label_cols)}",
        f"- 数值列示例: {', '.join(value_cols[:8])}",
        f"- 标签列示例: {', '.join(label_cols[:4])}",
        "",
        "文件:",
        "",
    ]
    for row in preprocessed_summary.itertuples(index=False):
        rows_text = "未知" if pd.isna(row.rows) else str(int(row.rows))
        role = "训练集" if row.role == "train" else "测试集"
        lines.append(f"- `{row.csv_file}`: {role}, {row.size_gb} GB, 行数={rows_text}, 维度={row.dimensions}")

    lines.extend([
        "",
        "## 异常切片",
        "",
        f"- 异常事件数: {len(events)}",
        f"- 事件持续时间中位数/小时: {events['DurationHours'].median():.3f}",
        f"- 受影响通道数中位数: {events['ChannelCount'].median():.1f}",
        "",
    ])
    zh_names = {
        "Category": "类别",
        "Dimensionality": "维度类型",
        "Locality": "局部/全局",
        "Length": "长度类型",
        "Class": "类别编号",
        "Subclass": "子类别编号",
    }
    for feature in ["Category", "Dimensionality", "Locality", "Length", "Class", "Subclass"]:
        lines.append(f"- {zh_names[feature]} ({feature}): {_distribution_text(distributions, feature)}")

    lines.extend([
        "",
        "## 图表",
        "",
        "- `01_dataset_overview/figures/category_distribution.png`: 异常类别分布",
        "- `01_dataset_overview/figures/length_distribution.png`: 点异常/子序列异常分布",
        "- `01_dataset_overview/figures/dimensionality_by_locality_heatmap.png`: 维度类型 x 局部/全局热力图",
        "- `01_dataset_overview/figures/duration_bucket_distribution.png`: 事件持续时间分桶",
        "- `02_representative_events/figures/*.png`: 代表性异常事件时序窗口图",
        "",
        "## 代表性事件",
        "",
    ])
    for row in plot_index.itertuples(index=False):
        lines.append(
            f"- `{row.ID}`: {row.Category}, {row.Dimensionality}, {row.Locality}, "
            f"{row.Length}, {row.Class}, `{row.Figure}`"
        )

    if result_metric:
        lines.extend([
            "",
            "## 算法总览",
            "",
            f"- 算法总览图使用 `results.csv` 中的 `{result_metric}` 指标。",
            "- 这仍然是整段测试集的总分，不是逐事件或逐异常类型的算法分数曲线。",
            "- 下一步研究工作是保存事件级 score 曲线，并在异常窗口图上叠加算法分数。",
        ])

    (output_dir / "visual_report.zh.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(output_dir, preprocessed_summary, value_cols, label_cols, events, distributions, plot_index, result_metric):
    write_report_en(output_dir, preprocessed_summary, value_cols, label_cols, events, distributions, plot_index, result_metric)
    write_report_zh(output_dir, preprocessed_summary, value_cols, label_cols, events, distributions, plot_index, result_metric)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = data_base(args.data_path)
    test_csv = base / "84_months.test.csv"

    labels, _, _ = read_mission_tables(args.mission_root)
    preprocessed_summary, value_cols, label_cols = summarize_preprocessed_data(base)
    preprocessed_dir = output_dir / "00_preprocessed_data"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_summary.to_csv(preprocessed_dir / "preprocessed_data_summary.csv", index=False)
    pd.DataFrame({"value_column": value_cols}).to_csv(preprocessed_dir / "value_columns.csv", index=False)
    pd.DataFrame({"label_column": label_cols}).to_csv(preprocessed_dir / "label_columns.csv", index=False)

    events, distributions = create_overview(labels, output_dir)
    plot_index = create_event_plots(test_csv, args.mission_root, events, output_dir, args)
    result_metric = create_result_chart(args.results_csv, output_dir)
    write_report(output_dir, preprocessed_summary, value_cols, label_cols, events, distributions, plot_index, result_metric)
    print(f"Wrote visual report to {output_dir}")


if __name__ == "__main__":
    main()
