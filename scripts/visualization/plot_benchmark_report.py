#!/usr/bin/env python3
"""Create presentation-ready figures from ESA-ADB Lite benchmark outputs."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "lines.linewidth": 2.4,
})


PALETTE = {
    "HBOS": "#2f6f8f",
    "iForest": "#4d8b31",
    "KNN": "#b86b25",
    "Subsequence_IF": "#8a5fbf",
    "PCC": "#555f6e",
    "STD3": "#b94444",
    "STD5": "#d08a39",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ESA-ADB Lite benchmark report figures")
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory containing results.csv")
    parser.add_argument("--output-dir", help="Output directory. Default: <run-dir>/presentation_figures")
    parser.add_argument("--primary-metric", default="Point_AUPR")
    parser.add_argument("--secondary-metric", default="Point_AUROC")
    parser.add_argument("--min-events", type=int, default=3)
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def read_required(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def colors_for(items):
    fallback = plt.cm.tab10(np.linspace(0, 1, max(1, len(items))))
    colors = []
    for i, item in enumerate(items):
        colors.append(PALETTE.get(str(item), fallback[i % len(fallback)]))
    return colors


def save_barh(df, label_col, value_col, title, path, xlabel=None):
    data = df.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(12.5, max(6.2, len(data) * 0.75)))
    labels = data[label_col].astype(str).tolist()
    ax.barh(labels, data[value_col], color=colors_for(labels), alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel or value_col)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    for i, value in enumerate(data[value_col]):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_overall(results, out_dir, primary_metric, secondary_metric):
    metric_cols = [primary_metric, secondary_metric, "F1", "VUS_ROC", "VUS_PR"]
    metric_cols = [col for col in metric_cols if col in results.columns]
    summary = (
        results.groupby("algorithm")[metric_cols + ["time_s"]]
        .mean(numeric_only=True)
        .round(6)
        .reset_index()
        .sort_values(primary_metric, ascending=False)
    )
    summary.to_csv(out_dir / "overall_algorithm_summary.csv", index=False)

    save_barh(
        summary[["algorithm", primary_metric]],
        "algorithm",
        primary_metric,
        f"Overall mean {primary_metric}",
        out_dir / "overall_mean_primary_metric.png",
        primary_metric,
    )

    plot_cols = [col for col in [primary_metric, secondary_metric, "VUS_ROC", "F1"] if col in summary.columns]
    plot_df = summary.set_index("algorithm")[plot_cols]
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    plot_df.plot(kind="bar", ax=ax, color=["#2f6f8f", "#4d8b31", "#8a5fbf", "#b86b25"][: len(plot_cols)])
    ax.set_title("Overall algorithm comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, max(1.0, plot_df.max().max() * 1.12))
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, ncols=2)
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_multi_metric_bars.png", dpi=220)
    plt.close(fig)
    return summary


def plot_channel_comparison(results, out_dir, primary_metric, secondary_metric):
    keep_cols = [primary_metric, secondary_metric, "F1", "VUS_ROC", "time_s"]
    keep_cols = [col for col in keep_cols if col in results.columns]
    summary = (
        results.groupby(["channels", "algorithm"])[keep_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(out_dir / "channel_algorithm_summary.csv", index=False)

    for channel in sorted(summary["channels"].dropna().unique()):
        part = summary[summary["channels"] == channel].sort_values(primary_metric, ascending=False)
        save_barh(
            part[["algorithm", primary_metric]],
            "algorithm",
            primary_metric,
            f"{channel}: mean {primary_metric}",
            out_dir / f"{channel}_primary_metric_ranking.png",
            primary_metric,
        )

    pivot = summary.pivot(index="algorithm", columns="channels", values=primary_metric)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    pivot.plot(kind="bar", ax=ax, color=["#2f6f8f", "#b86b25", "#4d8b31"][: len(pivot.columns)])
    ax.set_title(f"{primary_metric}: Target vs Subset")
    ax.set_xlabel("")
    ax.set_ylabel(primary_metric)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, title="")
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "target_vs_subset_primary_metric.png", dpi=220)
    plt.close(fig)
    return summary


def plot_time_tradeoff(channel_summary, out_dir, primary_metric):
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    for channel, group in channel_summary.groupby("channels"):
        ax.scatter(
            group["time_s"],
            group[primary_metric],
            s=95,
            alpha=0.82,
            label=channel,
            edgecolor="#222222",
            linewidth=0.4,
        )
        for row in group.itertuples(index=False):
            ax.text(row.time_s * 1.05, getattr(row, primary_metric), row.algorithm, fontsize=12, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per job, seconds (log scale)")
    ax.set_ylabel(primary_metric)
    ax.set_title("Effectiveness vs runtime")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "effectiveness_vs_runtime.png", dpi=220)
    plt.close(fig)


def reliable_slice_summary(slice_metrics, metric, min_events):
    group_cols = ["algorithm", "channels", "slice_type", "slice_value"]
    numeric_cols = [
        col for col in slice_metrics.columns
        if col not in {"dataset", *group_cols}
    ]
    summary = (
        slice_metrics.groupby(group_cols)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary = summary[summary["event_count"] >= min_events].copy()
    summary = summary.sort_values(["channels", "slice_type", "slice_value", metric], ascending=[True, True, True, False])
    summary["rank"] = summary.groupby(["channels", "slice_type", "slice_value"]).cumcount() + 1
    return summary


def plot_win_counts(reliable, out_dir, metric):
    wins = []
    for channels, group in reliable.groupby("channels"):
        total = group[["slice_type", "slice_value"]].drop_duplicates().shape[0]
        for algo, sub in group.groupby("algorithm"):
            wins.append({
                "channels": channels,
                "algorithm": algo,
                "slice_count": total,
                "top1": int((sub["rank"] == 1).sum()),
                "top2": int((sub["rank"] <= 2).sum()),
                "top3": int((sub["rank"] <= 3).sum()),
                f"mean_{metric}": sub[metric].mean(),
            })
    wins = pd.DataFrame(wins)
    wins.to_csv(out_dir / f"reliable_slice_win_counts_{metric}.csv", index=False)

    for channels in sorted(wins["channels"].unique()):
        part = wins[wins["channels"] == channels].sort_values(["top1", "top3"], ascending=False)
        fig, ax = plt.subplots(figsize=(13.0, 7.0))
        x = np.arange(len(part))
        width = 0.25
        ax.bar(x - width, part["top1"], width=width, label="Top-1", color="#2f6f8f")
        ax.bar(x, part["top2"], width=width, label="Top-2", color="#4d8b31")
        ax.bar(x + width, part["top3"], width=width, label="Top-3", color="#b86b25")
        ax.set_xticks(x, part["algorithm"], rotation=18, ha="right")
        ax.set_ylabel("Reliable slice count")
        ax.set_title(f"{channels}: reliable-slice top-k counts ({metric})")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{channels}_reliable_slice_topk_counts_{metric}.png", dpi=220)
        plt.close(fig)
    return wins


def plot_slice_heatmaps(reliable, out_dir, metric):
    core_types = ["Category", "Locality", "DurationBucket", "Dimensionality", "Length"]
    for channels in sorted(reliable["channels"].unique()):
        part = reliable[(reliable["channels"] == channels) & (reliable["slice_type"].isin(core_types))]
        if part.empty:
            continue
        best = part[part["rank"] == 1].copy()
        best["slice"] = best["slice_type"] + "=" + best["slice_value"].astype(str)
        best = best.sort_values(["slice_type", "slice_value"])

        fig, ax = plt.subplots(figsize=(14.0, max(6.2, len(best) * 0.72)))
        y = np.arange(len(best))
        bars = ax.barh(y, best[metric], color=colors_for(best["algorithm"]), alpha=0.88)
        ax.set_yticks(y, best["slice"])
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(f"{channels}: best algorithm by reliable slice")
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        for bar, algo in zip(bars, best["algorithm"]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {algo}", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{channels}_best_algorithm_by_slice_{metric}.png", dpi=220)
        plt.close(fig)

        matrix = part.pivot_table(index="slice_value", columns="algorithm", values=metric, aggfunc="mean")
        matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(13.5, max(6.2, len(matrix) * 0.75)))
        im = ax.imshow(matrix.values, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index.astype(str))
        ax.set_title(f"{channels}: reliable slice {metric} heatmap")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{channels}_slice_metric_heatmap_{metric}.png", dpi=220)
        plt.close(fig)


def plot_dataset_split_trends(results, out_dir, primary_metric):
    order = ["3_months", "10_months", "21_months", "42_months", "84_months"]
    for channels in sorted(results["channels"].unique()):
        part = results[results["channels"] == channels].copy()
        part["dataset"] = pd.Categorical(part["dataset"], categories=order, ordered=True)
        fig, ax = plt.subplots(figsize=(13.5, 7.2))
        for algorithm, group in part.sort_values("dataset").groupby("algorithm"):
            ax.plot(group["dataset"].astype(str), group[primary_metric], marker="o", linewidth=1.5, label=algorithm)
        ax.set_title(f"{channels}: {primary_metric} across training lengths")
        ax.set_xlabel("Training split")
        ax.set_ylabel(primary_metric)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False, ncols=2, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{channels}_training_length_trend_{primary_metric}.png", dpi=220)
        plt.close(fig)


def write_report(out_dir, overall, channel_summary, wins, metric, secondary_metric, min_events):
    lines = [
        "# Benchmark Presentation Figures / 实验汇报图表",
        "",
        f"Primary metric / 主指标: `{metric}`",
        f"Reliable slice threshold / 可靠切片阈值: `event_count >= {min_events}`",
        "",
        "## Recommended PPT Figures / 推荐放入 PPT 的图",
        "",
        "- `overall_multi_metric_bars.png`: overall algorithm comparison / 总体算法对比",
        "- `target_vs_subset_primary_metric.png`: target vs subset contrast / Target 与 Subset 对比",
        "- `effectiveness_vs_runtime.png`: metric-cost tradeoff / 效果与耗时权衡",
        f"- `Target_57_reliable_slice_topk_counts_{metric}.png`: Target reliable-slice win counts / Target 可靠切片胜率",
        f"- `Subset_6_reliable_slice_topk_counts_{metric}.png`: Subset reliable-slice win counts / Subset 可靠切片胜率",
        f"- `Target_57_best_algorithm_by_slice_{metric}.png`: Target best algorithm by slice / Target 分切片最佳算法",
        f"- `Subset_6_best_algorithm_by_slice_{metric}.png`: Subset best algorithm by slice / Subset 分切片最佳算法",
        "- `Target_57_training_length_trend_Point_AUPR.png`: training length sensitivity / 训练长度敏感性",
        "- `Subset_6_training_length_trend_Point_AUPR.png`: training length sensitivity / 训练长度敏感性",
        "",
        "## Overall Ranking / 总体排序",
        "",
        "```text",
        overall[["algorithm", metric, secondary_metric, "F1", "VUS_ROC", "time_s"]]
        .to_string(index=False),
        "```",
        "",
        "## Channel-Specific Summary / 分通道组摘要",
        "",
        "```text",
        channel_summary[["channels", "algorithm", metric, secondary_metric, "F1", "VUS_ROC", "time_s"]]
        .sort_values(["channels", metric], ascending=[True, False])
        .to_string(index=False),
        "```",
        "",
        "## Reliable Slice Win Counts / 可靠切片胜率",
        "",
        "```text",
        wins.to_string(index=False),
        "```",
        "",
        "## Interpretation / 解读",
        "",
        "- Target_57 should be discussed separately from Subset_6 because their difficulty differs greatly.",
        "- Reliable slice rankings should be the main evidence for anomaly-type conclusions.",
        "- All-slice rankings are useful for diagnosis, but low event-count slices should not be over-interpreted.",
        "",
        "中文：",
        "",
        "- Target_57 和 Subset_6 难度差异明显，PPT 中应分开汇报。",
        "- 异常类型结论应优先使用可靠切片排名。",
        "- 全部切片结果可以用于诊断，但事件数过少的切片不宜过度解读。",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "presentation_figures"
    ensure_dir(out_dir)

    results = read_required(run_dir / "results.csv")
    slice_metrics = read_required(run_dir / "slice_metrics.csv")

    overall = plot_overall(results, out_dir, args.primary_metric, args.secondary_metric)
    channel_summary = plot_channel_comparison(results, out_dir, args.primary_metric, args.secondary_metric)
    plot_time_tradeoff(channel_summary, out_dir, args.primary_metric)
    reliable = reliable_slice_summary(slice_metrics, args.primary_metric, args.min_events)
    reliable.to_csv(out_dir / "reliable_slice_algorithm_ranking.csv", index=False)
    wins = plot_win_counts(reliable, out_dir, args.primary_metric)
    plot_slice_heatmaps(reliable, out_dir, args.primary_metric)
    plot_dataset_split_trends(results, out_dir, args.primary_metric)
    write_report(out_dir, overall, channel_summary, wins, args.primary_metric, args.secondary_metric, args.min_events)
    print(f"Wrote benchmark presentation figures to {out_dir}")


if __name__ == "__main__":
    main()
