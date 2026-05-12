#!/usr/bin/env python3
"""Summarize slice_metrics.csv files and select best algorithms per anomaly slice."""
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ESA-ADB Lite slice metrics")
    parser.add_argument("slice_metrics", nargs="+", help="One or more slice_metrics.csv files")
    parser.add_argument("--metric", default="Point_AUPR", help="Metric used for best-algorithm ranking")
    parser.add_argument("--output-dir", default="results/slice_analysis", help="Output directory")
    parser.add_argument(
        "--min-events",
        type=int,
        default=1,
        help="Ignore slices with fewer than this many events",
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


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_inputs(args.slice_metrics)
    if args.metric not in df.columns:
        raise ValueError(f"Metric {args.metric!r} not found. Available columns: {', '.join(df.columns)}")

    df = df[df["event_count"] >= args.min_events].copy()
    group_cols = ["algorithm", "channels", "slice_type", "slice_value"]
    numeric_cols = [
        col for col in df.columns
        if col not in {"dataset", "source_file", *group_cols}
    ]
    summary = (
        df.groupby(group_cols)[numeric_cols]
        .mean(numeric_only=True)
        .round(4)
        .reset_index()
    )
    summary.to_csv(out_dir / "slice_metric_summary.csv", index=False)

    ranking = summary.sort_values(
        ["channels", "slice_type", "slice_value", args.metric],
        ascending=[True, True, True, False],
    )
    best = ranking.groupby(["channels", "slice_type", "slice_value"], as_index=False).head(1)
    best.to_csv(out_dir / f"best_by_slice_{args.metric}.csv", index=False)

    pivot = summary.pivot_table(
        index=["channels", "slice_type", "slice_value"],
        columns="algorithm",
        values=args.metric,
        aggfunc="mean",
    ).reset_index()
    pivot.to_csv(out_dir / f"slice_metric_pivot_{args.metric}.csv", index=False)

    print(f"Wrote slice analysis to {out_dir}")
    print(best[["channels", "slice_type", "slice_value", "algorithm", args.metric, "event_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
