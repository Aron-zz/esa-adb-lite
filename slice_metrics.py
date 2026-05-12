"""Slice-level metrics for ESA-ADB Mission1 anomaly feature analysis."""
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import _basic_binary_metrics, _to_global_point_series, _to_2d


FEATURE_COLUMNS = ["Category", "Dimensionality", "Locality", "Length", "Class", "Subclass"]


def _mission_root_from_test_csv(test_csv_path):
    return Path(test_csv_path).resolve().parents[3] / "esa-adb" / "mission1" / "ESA-Mission1"


def _duration_bucket(hours):
    if hours <= 5 / 60:
        return "<=5min"
    if hours <= 1:
        return "5min-1h"
    if hours <= 6:
        return "1h-6h"
    if hours <= 24:
        return "6h-1d"
    if hours <= 7 * 24:
        return "1d-7d"
    return ">7d"


def _channel_count_bucket(count):
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return ">20"


def load_slice_context(test_csv_path, timestamps, channel_names, mission_root=None):
    """Load labels/anomaly types and precompute event-level slice columns."""
    root = Path(mission_root) if mission_root else _mission_root_from_test_csv(test_csv_path)
    labels = pd.read_csv(root / "labels.csv", parse_dates=["StartTime", "EndTime"])
    types = pd.read_csv(root / "anomaly_types.csv")

    labels["StartTime"] = labels["StartTime"].dt.tz_localize(None)
    labels["EndTime"] = labels["EndTime"].dt.tz_localize(None)
    labels = labels[labels["Channel"].isin(channel_names)].copy()
    if labels.empty:
        return pd.DataFrame()

    start = pd.Series(timestamps).min()
    end = pd.Series(timestamps).max()
    labels = labels[(labels["EndTime"] >= start) & (labels["StartTime"] <= end)].copy()
    labels = labels.merge(types, on="ID", how="left")

    events = labels.groupby("ID", sort=False).agg(
        StartTime=("StartTime", "min"),
        EndTime=("EndTime", "max"),
        ChannelCount=("Channel", "nunique"),
    ).reset_index()
    event_types = labels.groupby("ID", sort=False)[FEATURE_COLUMNS].first().reset_index()
    events = events.merge(event_types, on="ID", how="left")
    events["DurationHours"] = (events["EndTime"] - events["StartTime"]).dt.total_seconds() / 3600.0
    events["DurationBucket"] = events["DurationHours"].map(_duration_bucket)
    events["ChannelCountBucket"] = events["ChannelCount"].map(_channel_count_bucket)
    return events


def _mask_for_events(timestamps_np, events):
    mask = np.zeros(len(timestamps_np), dtype=bool)
    for row in events.itertuples(index=False):
        start = np.datetime64(row.StartTime)
        end = np.datetime64(row.EndTime)
        mask |= (timestamps_np >= start) & (timestamps_np <= end)
    return mask


def _iter_slices(events):
    slice_columns = [*FEATURE_COLUMNS, "DurationBucket", "ChannelCountBucket"]
    for column in slice_columns:
        if column not in events.columns:
            continue
        for value in events[column].fillna("Unknown").drop_duplicates():
            selected = events[events[column].fillna("Unknown") == value]
            if not selected.empty:
                yield column, str(value), selected


def evaluate_slices(
    y_true,
    y_scores,
    timestamps,
    slice_events,
    dataset,
    algorithm,
    channels,
):
    """
    Evaluate each anomaly slice against normal time points.

    Other anomaly slices are excluded from negatives. This makes the metric answer:
    "Can this algorithm rank this slice's anomaly windows above normal periods?"
    """
    if slice_events is None or slice_events.empty:
        return []

    y_true_2d = (_to_2d(y_true, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    y_scores_2d = _to_2d(y_scores, dtype=np.float64)
    global_true = _to_global_point_series(y_true_2d).astype(np.uint8, copy=False)
    global_scores = _to_global_point_series(y_scores_2d).astype(np.float64, copy=False)
    normal_mask = global_true == 0
    timestamps_np = pd.Series(timestamps).to_numpy(dtype="datetime64[ns]")

    rows = []
    for slice_type, slice_value, events in _iter_slices(slice_events):
        positive_mask = _mask_for_events(timestamps_np, events)
        if not positive_mask.any():
            continue
        eval_mask = positive_mask | normal_mask
        if eval_mask.sum() == 0:
            continue
        y_slice = positive_mask[eval_mask].astype(np.uint8)
        if len(np.unique(y_slice)) < 2:
            continue
        scores_slice = global_scores[eval_mask]
        metric_values = _basic_binary_metrics(y_slice, scores_slice)
        rows.append({
            "dataset": dataset,
            "algorithm": algorithm,
            "channels": channels,
            "slice_type": slice_type,
            "slice_value": slice_value,
            "event_count": int(events["ID"].nunique()),
            "positive_points": int(positive_mask.sum()),
            "eval_points": int(eval_mask.sum()),
            **metric_values,
        })
    return rows
