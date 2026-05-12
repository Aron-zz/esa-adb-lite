#!/usr/bin/env python3
"""Analyze ESA-ADB Mission1 anomaly slices and create lightweight figures."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FEATURE_COLUMNS = ["Category", "Dimensionality", "Locality", "Length", "Class", "Subclass"]


def parse_args():
    parser = argparse.ArgumentParser(description="ESA-ADB Mission1 anomaly feature analysis")
    parser.add_argument(
        "--mission-root",
        default="../data/esa-adb/mission1/ESA-Mission1",
        help="Path containing labels.csv, anomaly_types.csv, channels.csv",
    )
    parser.add_argument(
        "--results-csv",
        help="Optional benchmark results.csv. Used for overall algorithm charts only.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/anomaly_feature_analysis",
        help="Output directory for CSV summaries, figures, and markdown report.",
    )
    parser.add_argument(
        "--top-classes",
        type=int,
        default=12,
        help="Number of anomaly classes to show in class distribution charts.",
    )
    return parser.parse_args()


def read_mission_tables(mission_root):
    root = Path(mission_root)
    labels = pd.read_csv(root / "labels.csv", parse_dates=["StartTime", "EndTime"])
    anomaly_types = pd.read_csv(root / "anomaly_types.csv")
    channels = pd.read_csv(root / "channels.csv")

    labels["StartTime"] = labels["StartTime"].dt.tz_localize(None)
    labels["EndTime"] = labels["EndTime"].dt.tz_localize(None)
    labels = labels.merge(anomaly_types, on="ID", how="left")
    labels = labels.merge(channels[["Channel", "Subsystem", "Group", "Target"]], on="Channel", how="left")
    labels["DurationSeconds"] = (labels["EndTime"] - labels["StartTime"]).dt.total_seconds()
    return labels, anomaly_types, channels


def event_summary(labels):
    rows = []
    for event_id, group in labels.groupby("ID", sort=False):
        row = {
            "ID": event_id,
            "StartTime": group["StartTime"].min(),
            "EndTime": group["EndTime"].max(),
            "DurationSeconds": (group["EndTime"].max() - group["StartTime"].min()).total_seconds(),
            "DurationHours": (group["EndTime"].max() - group["StartTime"].min()).total_seconds() / 3600.0,
            "ChannelCount": group["Channel"].nunique(),
            "SubsystemCount": group["Subsystem"].nunique(),
            "Subsystems": ",".join(sorted(str(x) for x in group["Subsystem"].dropna().unique())),
        }
        for col in FEATURE_COLUMNS:
            values = group[col].dropna().unique()
            row[col] = values[0] if len(values) else "Unknown"
        rows.append(row)
    return pd.DataFrame(rows)


def channel_summary(labels):
    grouped = labels.groupby(["Channel", "Subsystem"], dropna=False)
    out = grouped.agg(
        LabelRows=("ID", "size"),
        EventCount=("ID", "nunique"),
        TotalDurationHours=("DurationSeconds", lambda x: round(float(x.sum()) / 3600.0, 3)),
    ).reset_index()
    return out.sort_values(["EventCount", "TotalDurationHours"], ascending=False)


def feature_distribution(events):
    frames = []
    for col in FEATURE_COLUMNS:
        counts = events[col].fillna("Unknown").value_counts().rename_axis("Value").reset_index(name="EventCount")
        counts.insert(0, "Feature", col)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def duration_buckets(events):
    bins = [-1, 5 / 60, 1, 6, 24, 7 * 24, float("inf")]
    labels = ["<=5min", "5min-1h", "1h-6h", "6h-1d", "1d-7d", ">7d"]
    out = events.copy()
    out["DurationBucket"] = pd.cut(out["DurationHours"], bins=bins, labels=labels)
    return out.groupby("DurationBucket", observed=False).size().reset_index(name="EventCount")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_bar(df, x_col, y_col, title, path, *, horizontal=False, top_n=None):
    data = df.head(top_n).copy() if top_n else df.copy()
    plt.figure(figsize=(10, max(4, min(8, len(data) * 0.45))) if horizontal else (8, 4.5))
    if horizontal:
        plt.barh(data[x_col].astype(str), data[y_col])
        plt.gca().invert_yaxis()
        plt.xlabel(y_col)
    else:
        plt.bar(data[x_col].astype(str), data[y_col])
        plt.ylabel(y_col)
        plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_heatmap(events, row_feature, col_feature, path):
    table = pd.crosstab(events[row_feature].fillna("Unknown"), events[col_feature].fillna("Unknown"))
    fig, ax = plt.subplots(figsize=(8, max(4, len(table) * 0.45)))
    im = ax.imshow(table.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(table.columns)), table.columns.astype(str), rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), table.index.astype(str))
    ax.set_title(f"{row_feature} x {col_feature}")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = int(table.values[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return table


def save_results_charts(results_csv, figures_dir):
    if not results_csv:
        return None
    path = Path(results_csv)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    metric = "Point_AUPR" if "Point_AUPR" in df.columns else "AUPR"
    if metric not in df.columns:
        return None
    summary = df.groupby(["algorithm", "channels"], dropna=False)[metric].mean().reset_index()
    pivot = summary.pivot(index="algorithm", columns="channels", values=metric)
    ax = pivot.plot(kind="bar", figsize=(8, 4.5))
    ax.set_title(f"Overall benchmark mean {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = figures_dir / "overall_algorithm_metric.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return summary


def _feature_values(distributions, feature):
    part = distributions[distributions["Feature"] == feature].head(8)
    return ", ".join(f"{row.Value}={row.EventCount}" for row in part.itertuples())


def write_report(output_dir, events, distributions, channel_dist, results_summary):
    report = output_dir / "README.md"
    lines = [
        "# ESA-ADB Mission1 Anomaly Feature Analysis",
        "",
        "## Dataset Slices",
        "",
        f"- Events: {len(events)}",
        f"- Label rows: {int(channel_dist['LabelRows'].sum())}",
        f"- Channels with labels: {channel_dist['Channel'].nunique()}",
        f"- Median event duration hours: {events['DurationHours'].median():.3f}",
        f"- Median affected channels per event: {events['ChannelCount'].median():.1f}",
        "",
        "## Feature Distributions",
        "",
    ]
    for feature in FEATURE_COLUMNS:
        lines.append(f"- {feature}: {_feature_values(distributions, feature)}")
    lines.extend([
        "",
        "## Figures",
        "",
        "- `figures/category_distribution.png`",
        "- `figures/dimensionality_distribution.png`",
        "- `figures/locality_distribution.png`",
        "- `figures/length_distribution.png`",
        "- `figures/top_class_distribution.png`",
        "- `figures/duration_bucket_distribution.png`",
        "- `figures/category_by_length_heatmap.png`",
        "- `figures/dimensionality_by_locality_heatmap.png`",
        "- `figures/top_channel_events.png`",
        "",
    ])
    if results_summary is not None:
        lines.extend([
            "## Benchmark Result Note",
            "",
            "The optional benchmark chart summarizes whole-test metrics from `results.csv`.",
            "It is not yet a per-anomaly-type performance slice because the current benchmark output does not store per-timestamp scores.",
            "Use this together with future `slice_metrics.csv` or rerun scoring with slice-aware metrics.",
            "",
        ])
    report.write_text("\n".join(lines), encoding="utf-8")

    zh_report = output_dir / "README.zh.md"
    zh_lines = [
        "# ESA-ADB Mission1 异常特征分析",
        "",
        "## 数据集切片",
        "",
        f"- 异常事件数: {len(events)}",
        f"- 标签行数: {int(channel_dist['LabelRows'].sum())}",
        f"- 有异常标签的通道数: {channel_dist['Channel'].nunique()}",
        f"- 事件持续时间中位数/小时: {events['DurationHours'].median():.3f}",
        f"- 每个事件受影响通道数中位数: {events['ChannelCount'].median():.1f}",
        "",
        "## 特征分布",
        "",
    ]
    zh_names = {
        "Category": "类别",
        "Dimensionality": "维度类型",
        "Locality": "局部/全局",
        "Length": "长度类型",
        "Class": "类别编号",
        "Subclass": "子类别编号",
    }
    for feature in FEATURE_COLUMNS:
        zh_lines.append(f"- {zh_names.get(feature, feature)} ({feature}): {_feature_values(distributions, feature)}")
    zh_lines.extend([
        "",
        "## 图表",
        "",
        "- `figures/category_distribution.png`: 异常类别分布",
        "- `figures/dimensionality_distribution.png`: 单变量/多变量分布",
        "- `figures/locality_distribution.png`: 局部/全局分布",
        "- `figures/length_distribution.png`: 点异常/子序列异常分布",
        "- `figures/top_class_distribution.png`: 高频异常类别",
        "- `figures/duration_bucket_distribution.png`: 持续时间分桶",
        "- `figures/category_by_length_heatmap.png`: 类别 x 长度热力图",
        "- `figures/dimensionality_by_locality_heatmap.png`: 维度类型 x 局部/全局热力图",
        "- `figures/top_channel_events.png`: 异常事件最多的通道",
        "",
    ])
    if results_summary is not None:
        zh_lines.extend([
            "## Benchmark 结果说明",
            "",
            "`results.csv` 生成的图只表示整段测试集上的算法总分。",
            "它还不是按异常类型切片后的性能，因为当前 benchmark 输出没有保存逐时间点分数。",
            "后续需要结合 `slice_metrics.csv` 或重新运行带切片指标的评分流程。",
            "",
        ])
    zh_report.write_text("\n".join(zh_lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    labels, _, _ = read_mission_tables(args.mission_root)
    events = event_summary(labels)
    channel_dist = channel_summary(labels)
    distributions = feature_distribution(events)
    buckets = duration_buckets(events)

    events.to_csv(output_dir / "event_summary.csv", index=False)
    channel_dist.to_csv(output_dir / "channel_event_distribution.csv", index=False)
    distributions.to_csv(output_dir / "feature_distribution.csv", index=False)
    buckets.to_csv(output_dir / "duration_bucket_distribution.csv", index=False)

    for feature in ["Category", "Dimensionality", "Locality", "Length"]:
        part = distributions[distributions["Feature"] == feature]
        save_bar(
            part,
            "Value",
            "EventCount",
            f"{feature} distribution",
            figures_dir / f"{feature.lower()}_distribution.png",
        )

    class_part = distributions[distributions["Feature"] == "Class"]
    save_bar(
        class_part,
        "Value",
        "EventCount",
        "Top anomaly classes",
        figures_dir / "top_class_distribution.png",
        horizontal=True,
        top_n=args.top_classes,
    )
    save_bar(
        buckets,
        "DurationBucket",
        "EventCount",
        "Event duration buckets",
        figures_dir / "duration_bucket_distribution.png",
    )
    save_heatmap(events, "Category", "Length", figures_dir / "category_by_length_heatmap.png").to_csv(
        output_dir / "category_by_length.csv"
    )
    save_heatmap(events, "Dimensionality", "Locality", figures_dir / "dimensionality_by_locality_heatmap.png").to_csv(
        output_dir / "dimensionality_by_locality.csv"
    )
    save_bar(
        channel_dist.head(20),
        "Channel",
        "EventCount",
        "Top channels by event count",
        figures_dir / "top_channel_events.png",
        horizontal=True,
    )

    results_summary = save_results_charts(args.results_csv, figures_dir)
    if results_summary is not None:
        results_summary.to_csv(output_dir / "overall_algorithm_metric_summary.csv", index=False)

    write_report(output_dir, events, distributions, channel_dist, results_summary)
    print(f"Wrote anomaly feature analysis to {output_dir}")


if __name__ == "__main__":
    main()
