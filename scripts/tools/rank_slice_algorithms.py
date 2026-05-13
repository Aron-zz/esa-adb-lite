#!/usr/bin/env python3
"""Rank algorithms by anomaly slices and summarize stable winners."""
import argparse
from pathlib import Path

import pandas as pd


ID_COLS = ["channels", "slice_type", "slice_value"]


def parse_args():
    parser = argparse.ArgumentParser(description="Rank ESA-ADB Lite algorithms by anomaly slice")
    parser.add_argument("slice_metrics", nargs="+", help="One or more slice_metrics.csv files")
    parser.add_argument("--metric", default="Point_AUPR", help="Metric used for ranking")
    parser.add_argument("--secondary-metric", default="Point_AUROC", help="Tie-breaker metric")
    parser.add_argument("--output-dir", default="results/slice_ranking", help="Output directory")
    parser.add_argument("--min-events", type=int, default=3, help="Reliable slice event-count threshold")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Optional channel groups to keep, e.g. Target_57 Subset_6",
    )
    parser.add_argument(
        "--slice-types",
        nargs="+",
        default=None,
        help="Optional slice types to keep, e.g. Category Locality DurationBucket",
    )
    return parser.parse_args()


def read_inputs(paths):
    frames = []
    for path in paths:
        p = Path(path)
        df = pd.read_csv(p)
        df["source_file"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def mean_summary(df):
    group_cols = ["algorithm", *ID_COLS]
    numeric_cols = [
        col for col in df.columns
        if col not in {"dataset", "source_file", *group_cols}
    ]
    return (
        df.groupby(group_cols)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )


def rank_summary(summary, metric, secondary_metric):
    sort_cols = [*ID_COLS, metric]
    ascending = [True, True, True, False]
    if secondary_metric in summary.columns and secondary_metric != metric:
        sort_cols.append(secondary_metric)
        ascending.append(False)
    ranked = summary.sort_values(sort_cols, ascending=ascending).copy()
    ranked["rank"] = ranked.groupby(ID_COLS).cumcount() + 1
    return ranked


def win_counts(ranked, metric):
    rows = []
    for channels, group in ranked.groupby("channels"):
        algorithms = sorted(group["algorithm"].unique())
        total_slices = group[ID_COLS].drop_duplicates().shape[0]
        for algorithm in algorithms:
            sub = group[group["algorithm"] == algorithm]
            rows.append({
                "channels": channels,
                "algorithm": algorithm,
                "slice_count": total_slices,
                "top1_count": int((sub["rank"] == 1).sum()),
                "top2_count": int((sub["rank"] <= 2).sum()),
                "top3_count": int((sub["rank"] <= 3).sum()),
                f"mean_{metric}": round(sub[metric].mean(), 6),
                f"median_{metric}": round(sub[metric].median(), 6),
            })
    out = pd.DataFrame(rows)
    for col in ["top1_count", "top2_count", "top3_count"]:
        out[col.replace("_count", "_rate")] = (out[col] / out["slice_count"]).round(4)
    return out.sort_values(["channels", "top1_count", f"mean_{metric}"], ascending=[True, False, False])


def make_report(out_dir, metric, ranked, reliable_ranked, wins, reliable_wins):
    lines = [
        "# Slice Algorithm Ranking / 异常切片算法排名",
        "",
        f"Ranking metric / 排名指标：`{metric}`",
        "",
        "## Outputs / 输出文件",
        "",
        "- `slice_algorithm_ranking_all.csv`: all slice rankings / 全部切片排名",
        "- `slice_algorithm_ranking_reliable.csv`: filtered reliable slice rankings / 可靠切片排名",
        "- `top3_by_slice_reliable.csv`: top-3 algorithms per reliable slice / 每个可靠切片 Top-3",
        "- `algorithm_win_counts_all.csv`: top-k counts on all slices / 全部切片 Top-k 胜率",
        "- `algorithm_win_counts_reliable.csv`: top-k counts on reliable slices / 可靠切片 Top-k 胜率",
        "",
        "## Reliable Slice Summary / 可靠切片概览",
        "",
    ]

    if reliable_ranked.empty:
        lines.append("No reliable slices after filtering. / 过滤后没有可靠切片。")
    else:
        lines.append("Top-1 counts by channel group:")
        lines.append("")
        lines.append("```text")
        lines.append(reliable_wins.to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("Top-3 algorithms for reliable slices:")
        lines.append("")
        lines.append("```text")
        top3 = reliable_ranked[reliable_ranked["rank"] <= 3]
        cols = [*ID_COLS, "rank", "algorithm", metric, "event_count"]
        lines.append(top3[cols].to_string(index=False))
        lines.append("```")

    lines.append("")
    lines.append("## Interpretation / 解读")
    lines.append("")
    lines.append(
        "Use reliable rankings for research conclusions. Slices with very few events are useful "
        "for diagnosis, but should not be treated as strong evidence."
    )
    lines.append("")
    lines.append("中文：研究结论优先使用可靠切片排名。事件数很少的切片可用于诊断，但不适合作为强证据。")
    lines.append("")
    lines.append("All-slice top-1 counts:")
    lines.append("")
    lines.append("```text")
    lines.append(wins.to_string(index=False))
    lines.append("```")

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_inputs(args.slice_metrics)
    if args.metric not in df.columns:
        raise ValueError(f"Metric {args.metric!r} not found. Available: {', '.join(df.columns)}")
    if args.channels:
        df = df[df["channels"].isin(args.channels)]
    if args.slice_types:
        df = df[df["slice_type"].isin(args.slice_types)]

    summary = mean_summary(df)
    ranked = rank_summary(summary, args.metric, args.secondary_metric)
    reliable = summary[summary["event_count"] >= args.min_events].copy()
    reliable_ranked = rank_summary(reliable, args.metric, args.secondary_metric) if not reliable.empty else reliable

    wins = win_counts(ranked, args.metric)
    reliable_wins = (
        win_counts(reliable_ranked, args.metric)
        if not reliable_ranked.empty
        else pd.DataFrame()
    )

    ranked.to_csv(out_dir / "slice_algorithm_ranking_all.csv", index=False)
    reliable_ranked.to_csv(out_dir / "slice_algorithm_ranking_reliable.csv", index=False)
    ranked[ranked["rank"] <= 3].to_csv(out_dir / "top3_by_slice_all.csv", index=False)
    if not reliable_ranked.empty:
        reliable_ranked[reliable_ranked["rank"] <= 3].to_csv(out_dir / "top3_by_slice_reliable.csv", index=False)
    wins.to_csv(out_dir / "algorithm_win_counts_all.csv", index=False)
    reliable_wins.to_csv(out_dir / "algorithm_win_counts_reliable.csv", index=False)
    make_report(out_dir, args.metric, ranked, reliable_ranked, wins, reliable_wins)

    print(f"Wrote slice rankings to {out_dir}")
    if not reliable_wins.empty:
        print(reliable_wins.to_string(index=False))


if __name__ == "__main__":
    main()
