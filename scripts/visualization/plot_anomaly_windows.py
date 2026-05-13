#!/usr/bin/env python3
"""Plot Mission1 telemetry windows with anomaly intervals highlighted."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})


FEATURE_COLUMNS = ["Class", "Subclass", "Category", "Dimensionality", "Locality", "Length"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ESA-ADB Mission1 anomaly time-series windows")
    parser.add_argument(
        "--test-csv",
        default="../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv",
        help="Preprocessed 84_months.test.csv",
    )
    parser.add_argument(
        "--mission-root",
        default="../data/esa-adb/mission1/ESA-Mission1",
        help="Path containing labels.csv, anomaly_types.csv, channels.csv",
    )
    parser.add_argument("--event-ids", nargs="+", help="Specific event IDs, e.g. id_118 id_126")
    parser.add_argument("--categories", nargs="+", help="Filter by Category, e.g. Anomaly 'Rare Event'")
    parser.add_argument("--lengths", nargs="+", help="Filter by Length, e.g. Point Subsequence")
    parser.add_argument("--classes", nargs="+", help="Filter by Class, e.g. class_3 class_7")
    parser.add_argument("--channels", nargs="+", help="Force channels to plot instead of event channels")
    parser.add_argument("--max-events", type=int, default=6, help="Maximum number of events to plot")
    parser.add_argument("--max-channels-per-event", type=int, default=4, help="Maximum channels per event")
    parser.add_argument("--context-hours", type=float, default=12.0, help="Hours before/after event window")
    parser.add_argument(
        "--window-mode",
        choices=["fixed", "adaptive"],
        default="adaptive",
        help="fixed uses --context-hours; adaptive uses 4h/1d/event±7d standard windows by event span",
    )
    parser.add_argument("--chunk-size", type=int, default=200_000, help="CSV rows per chunk")
    parser.add_argument(
        "--output-dir",
        default="results/anomaly_windows",
        help="Output directory for figures and index CSV",
    )
    return parser.parse_args()


def normalize_datetime(series):
    return pd.to_datetime(series).dt.tz_localize(None)


def load_metadata(mission_root):
    root = Path(mission_root)
    labels = pd.read_csv(root / "labels.csv")
    labels["StartTime"] = normalize_datetime(labels["StartTime"])
    labels["EndTime"] = normalize_datetime(labels["EndTime"])
    types = pd.read_csv(root / "anomaly_types.csv")
    events = labels.groupby("ID", sort=False).agg(
        StartTime=("StartTime", "min"),
        EndTime=("EndTime", "max"),
        ChannelCount=("Channel", "nunique"),
    ).reset_index()
    events = events.merge(types, on="ID", how="left")
    return labels, events


def first_last_timestamp(test_csv):
    first = pd.read_csv(test_csv, usecols=["timestamp"], nrows=1)["timestamp"].iloc[0]
    last = None
    for chunk in pd.read_csv(test_csv, usecols=["timestamp"], chunksize=500_000):
        last = chunk["timestamp"].iloc[-1]
    return pd.to_datetime(first).tz_localize(None), pd.to_datetime(last).tz_localize(None)


def select_events(events, test_start, test_end, args):
    selected = events[(events["EndTime"] >= test_start) & (events["StartTime"] <= test_end)].copy()
    if args.event_ids:
        selected = selected[selected["ID"].isin(args.event_ids)]
    if args.categories:
        selected = selected[selected["Category"].isin(args.categories)]
    if args.lengths:
        selected = selected[selected["Length"].isin(args.lengths)]
    if args.classes:
        selected = selected[selected["Class"].isin(args.classes)]
    selected = selected.sort_values(["StartTime", "ID"]).head(args.max_events)
    if selected.empty:
        raise ValueError("No events matched the filters within the test split.")
    return selected


def choose_channels(labels, event_id, args):
    if args.channels:
        return args.channels[: args.max_channels_per_event]
    event_labels = labels.loc[labels["ID"] == event_id].copy()
    event_labels["DurationSeconds"] = (event_labels["EndTime"] - event_labels["StartTime"]).dt.total_seconds()
    ranked = (
        event_labels
        .groupby("Channel", as_index=False)["DurationSeconds"]
        .max()
        .sort_values(["DurationSeconds", "Channel"], ascending=[False, True])
    )
    return ranked["Channel"].head(args.max_channels_per_event).tolist()


def read_window(test_csv, channels, start, end, chunk_size):
    label_cols = [f"is_anomaly_{channel}" for channel in channels]
    usecols = ["timestamp", *channels, *label_cols]
    frames = []
    for chunk in pd.read_csv(test_csv, usecols=lambda col: col in usecols, chunksize=chunk_size):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        mask = (chunk["timestamp"] >= start) & (chunk["timestamp"] <= end)
        if mask.any():
            frames.append(chunk.loc[mask].copy())
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def event_title(event):
    parts = [event["ID"]]
    for col in FEATURE_COLUMNS:
        if pd.notna(event.get(col)):
            parts.append(f"{col}={event[col]}")
    return " | ".join(parts)


def choose_window(event, test_start, test_end, context_hours, window_mode):
    if window_mode == "fixed":
        context = pd.Timedelta(hours=context_hours)
        return (
            max(test_start, event["StartTime"] - context),
            min(test_end, event["EndTime"] + context),
            f"fixed_pm_{context_hours:g}h",
        )

    duration = event["EndTime"] - event["StartTime"]
    duration_hours = max(duration.total_seconds() / 3600.0, 0.0)
    if duration_hours <= 6:
        total = pd.Timedelta(hours=4)
        label = "adaptive_4h"
        midpoint = event["StartTime"] + duration / 2
        start = midpoint - total / 2
        end = midpoint + total / 2
    elif duration_hours <= 24:
        total = pd.Timedelta(days=1)
        label = "adaptive_1d"
        midpoint = event["StartTime"] + duration / 2
        start = midpoint - total / 2
        end = midpoint + total / 2
    else:
        # Long events need broad context around the full event interval.
        label = "adaptive_event_pm7d"
        start = event["StartTime"] - pd.Timedelta(days=7)
        end = event["EndTime"] + pd.Timedelta(days=7)
    if start < test_start:
        end = min(test_end, end + (test_start - start))
        start = test_start
    if end > test_end:
        start = max(test_start, start - (end - test_end))
        end = test_end
    return start, end, label


def plot_event(df, labels, event, channels, output_path):
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(15, max(4.2, 3.0 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    event_labels = labels[labels["ID"] == event["ID"]]
    for ax, channel in zip(axes, channels):
        ax.plot(df["timestamp"], df[channel], linewidth=1.35, color="#2f5d8c")
        rows = event_labels[event_labels["Channel"] == channel]
        point_times = []
        for row in rows.itertuples():
            duration_seconds = (row.EndTime - row.StartTime).total_seconds()
            if duration_seconds <= 1:
                point_times.append(row.StartTime)
                ax.axvline(row.StartTime, color="#8c1d18", linewidth=0.9, alpha=0.22, linestyle="--")
            else:
                ax.axvspan(row.StartTime, row.EndTime, color="#d94841", alpha=0.18)
                ax.axvline(row.StartTime, color="#8c1d18", linewidth=0.95, alpha=0.45)
                ax.axvline(row.EndTime, color="#8c1d18", linewidth=0.95, alpha=0.45)
        label_col = f"is_anomaly_{channel}"
        ymin, ymax = ax.get_ylim()
        if point_times:
            marker_y = ymax - (ymax - ymin) * 0.04
            ax.scatter(point_times, [marker_y] * len(point_times), marker="v", s=80,
                       color="#8c1d18", edgecolor="white", linewidth=0.8, zorder=5)
        elif label_col in df.columns and df[label_col].max() > 0:
            anom = df[df[label_col] > 0]
            marker_y = ymax - (ymax - ymin) * 0.04
            ax.scatter(anom["timestamp"], [marker_y] * len(anom), marker="|", s=26,
                       color="#8c1d18", alpha=0.45, zorder=5)
        ax.set_ylabel(channel)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)

    axes[0].set_title(event_title(event), fontsize=16)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_event_zscore(df, labels, event, channels, output_path):
    """Plot selected channels in one z-score panel for shape comparison."""
    fig, ax = plt.subplots(figsize=(15, 6.6))
    colors = ["#2f5d8c", "#2f8c5d", "#8c6d2f", "#7a4f9a", "#5c6f7c", "#b56b45"]
    offsets = []
    for i, channel in enumerate(channels):
        values = df[channel].astype(float)
        std = values.std()
        z = (values - values.mean()) / (std if std else 1.0)
        offset = i * 5.0
        offsets.append(offset)
        ax.plot(df["timestamp"], z + offset, linewidth=1.35, color=colors[i % len(colors)], label=channel)

    event_labels = labels[labels["ID"] == event["ID"]]
    for row in event_labels.itertuples():
        if row.Channel not in channels:
            continue
        duration_seconds = (row.EndTime - row.StartTime).total_seconds()
        if duration_seconds <= 1:
            ax.axvline(row.StartTime, color="#8c1d18", linewidth=0.9, alpha=0.22, linestyle="--")
            ax.scatter([row.StartTime], [max(offsets) + 2.2], marker="v", s=80,
                       color="#8c1d18", edgecolor="white", linewidth=0.8, zorder=5)
        else:
            ax.axvspan(row.StartTime, row.EndTime, color="#d94841", alpha=0.14)

    ax.set_title(event_title(event) + " | z-score overlay", fontsize=16)
    ax.set_yticks(offsets, channels)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.25)
    ax.legend(loc="upper right", ncols=min(4, len(channels)), fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    test_csv = Path(args.test_csv)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels, events = load_metadata(args.mission_root)
    test_start, test_end = first_last_timestamp(test_csv)
    selected = select_events(events, test_start, test_end, args)

    rows = []
    for event in selected.to_dict("records"):
        channels = choose_channels(labels, event["ID"], args)
        window_start, window_end, window_label = choose_window(
            event,
            test_start,
            test_end,
            args.context_hours,
            args.window_mode,
        )
        df = read_window(test_csv, channels, window_start, window_end, args.chunk_size)
        if df.empty:
            continue
        safe_id = event["ID"].replace("/", "_")
        out = figures_dir / f"{safe_id}_{event['Category']}_{event['Length']}_{window_label}.png"
        z_out = figures_dir / f"{safe_id}_{event['Category']}_{event['Length']}_{window_label}_zscore.png"
        plot_event(df, labels, event, channels, out)
        plot_event_zscore(df, labels, event, channels, z_out)
        rows.append({
            "ID": event["ID"],
            "StartTime": event["StartTime"],
            "EndTime": event["EndTime"],
            "Category": event.get("Category"),
            "Dimensionality": event.get("Dimensionality"),
            "Locality": event.get("Locality"),
            "Length": event.get("Length"),
            "Class": event.get("Class"),
            "Subclass": event.get("Subclass"),
            "Channels": ",".join(channels),
            "WindowStart": window_start,
            "WindowEnd": window_end,
            "WindowMode": window_label,
            "Figure": str(out),
            "ZScoreFigure": str(z_out),
            "RowsPlotted": len(df),
        })

    index = pd.DataFrame(rows)
    index.to_csv(output_dir / "plot_index.csv", index=False)
    print(f"Wrote {len(index)} anomaly window plots to {figures_dir}")


if __name__ == "__main__":
    main()
