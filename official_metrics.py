"""Mission1 官方事件指标的轻量接入。"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional
import math
import sys

import numpy as np
import pandas as pd
import portion as P


AFFILIATION_REPO = Path(__file__).resolve().parents[1] / "esa-adb-classical" / "timeeval" / "metrics" / "affiliation_based_metrics_repo"
if str(AFFILIATION_REPO) not in sys.path:
    sys.path.insert(0, str(AFFILIATION_REPO))

try:
    from affiliation import pr_from_events  # type: ignore
except ModuleNotFoundError as exc:
    pr_from_events = None
    AFFILIATION_IMPORT_ERROR = exc
else:
    AFFILIATION_IMPORT_ERROR = None


NANOSECONDS_IN_MILLISECOND = 1e6
NANOSECONDS_IN_SECOND = NANOSECONDS_IN_MILLISECOND * 1000


def convert_time_series_to_events(vector):
    vector = np.asarray(vector)

    def find_runs(x):
        x = np.asanyarray(x)
        if x.ndim != 1:
            raise ValueError("only 1D array supported")
        n = x.shape[0]
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        run_values = x[loc_run_start]
        run_lengths = np.diff(np.append(run_starts, n))
        run_ends = run_starts + run_lengths
        return np.stack((run_starts[run_values > 0], run_ends[run_values > 0])).transpose()

    non_zero_runs = find_runs(vector[..., 1])
    events = []
    n = len(vector)
    for x, y in non_zero_runs:
        if y == n:
            events.append(P.closed(vector[..., 0][x], vector[..., 0][y - 1]))
        else:
            events.append(P.closedopen(vector[..., 0][x], vector[..., 0][y]))
    return P.Interval(*events)


def scale_scores(y_scores):
    y_scores = np.asarray(y_scores, dtype=np.float32)
    if y_scores.ndim == 1:
        y_scores = np.expand_dims(y_scores, -1)

    for i in range(y_scores.shape[-1]):
        column = y_scores[..., i]
        mask = np.isinf(column) | np.isneginf(column) | np.isnan(column)
        scores = column[~mask]
        if scores.size == 0:
            continue
        min_v = scores.min()
        max_v = scores.max()
        if max_v > min_v:
            column[~mask] = (scores - min_v) / (max_v - min_v)
        else:
            column[~mask] = 0.0
        y_scores[..., i] = column
    return y_scores


def topk_binarize(y_scores, fraction=0.05):
    scores = np.asarray(y_scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = np.expand_dims(scores, -1)

    binary = np.zeros_like(scores, dtype=np.uint8)
    for i in range(scores.shape[1]):
        col = scores[:, i]
        if len(col) == 0:
            continue
        n_anom = max(1, int(len(col) * fraction))
        threshold = np.partition(col, len(col) - n_anom)[len(col) - n_anom]
        binary[:, i] = (col >= threshold).astype(np.uint8)
    return binary


def postprocess_binary_events(y_binary, merge_gap_points=0, min_event_points=1):
    """Merge nearby binary detections and optionally remove very short events."""
    binary = np.asarray(y_binary, dtype=np.uint8).copy()
    if binary.ndim == 1:
        binary = np.expand_dims(binary, -1)

    merge_gap_points = max(0, int(merge_gap_points or 0))
    min_event_points = max(1, int(min_event_points or 1))
    if merge_gap_points == 0 and min_event_points <= 1:
        return binary

    out = np.zeros_like(binary, dtype=np.uint8)
    for i in range(binary.shape[1]):
        col = binary[:, i]
        idx = np.flatnonzero(col)
        if idx.size == 0:
            continue

        start = prev = int(idx[0])
        events = []
        for pos in idx[1:]:
            pos = int(pos)
            if pos - prev <= merge_gap_points + 1:
                prev = pos
            else:
                events.append((start, prev))
                start = prev = pos
        events.append((start, prev))

        for start, end in events:
            if end - start + 1 >= min_event_points:
                out[start:end + 1, i] = 1
    return out


def make_time_series(timestamps, values):
    return np.array(list(zip(pd.to_datetime(timestamps).tolist(), values.tolist())), dtype=object)


def _fbeta(precision, recall, beta):
    divider = (beta ** 2) * precision + recall
    if divider == 0:
        return 0.0
    return ((1 + beta ** 2) * precision * recall) / divider


class ESAScoresLite:
    def __init__(self, betas=0.5, select_labels: Optional[dict] = None, full_range: Optional[tuple] = None, name: Optional[str] = None):
        self._betas = np.atleast_1d(betas)
        self.full_range = full_range
        if select_labels is None or len(select_labels) == 0:
            self.selected_labels = dict()
            filter_string = "ALL"
        else:
            select_labels = {col: np.atleast_1d(val) for col, val in select_labels.items()}
            self.selected_labels = select_labels
            filter_string = "_".join(["_".join(val) for val in select_labels.values()])
        self.name = filter_string if name is None else name

    def score(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
        y_pred = np.asarray(y_pred)
        if len(y_pred) == 0:
            return {}

        if self.full_range is None:
            self.full_range = (min(y_true["StartTime"].min(), min(y_pred[..., 0])), max(y_true["EndTime"].max(), max(y_pred[..., 0])))

        if y_pred[0, 0] > self.full_range[0]:
            y_pred = np.array([np.array([self.full_range[0], y_pred[0, 1]], dtype=object), *y_pred], dtype=object)
        if y_pred[-1, 0] < self.full_range[1]:
            y_pred = np.array([*y_pred, np.array([self.full_range[1], y_pred[-1, 1]], dtype=object)], dtype=object)

        events_pred = convert_time_series_to_events(y_pred)

        filtered_y_true = y_true.copy()
        for col, val in self.selected_labels.items():
            filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]
        if filtered_y_true.empty:
            return {"alarming_precision": 0.0, "EW_precision": 0.0, "EW_recall": 0.0,
                    **{f"EW_F_{b:.2f}": 0.0 for b in self._betas},
                    "AFF_precision": 0.0, "AFF_recall": 0.0,
                    **{f"AFF_F_{b:.2f}": 0.0 for b in self._betas}}

        true_positives = 0
        false_positives = 0
        redundant_detections = 0
        false_negatives = 0
        matched_events_pred = [False for _ in events_pred]
        for aid in filtered_y_true["ID"].unique():
            gt = filtered_y_true[filtered_y_true["ID"] == aid]
            gt_intervals = P.Interval(*[P.closed(*row) for _, row in gt[["StartTime", "EndTime"]].iterrows()])

            already_detected = [0 for _ in gt_intervals]
            at_least_one_detected = False
            for p, pred in enumerate(events_pred):
                if pred.upper < gt_intervals.lower or pred.lower > gt_intervals.upper:
                    continue
                intersections = [not (pred & g).empty for g in gt_intervals]
                if not any(intersections):
                    continue
                matched_events_pred[p] = True
                if not at_least_one_detected:
                    true_positives += 1
                    at_least_one_detected = True
                for i, val in enumerate(intersections):
                    if val:
                        already_detected[i] += 1

            redundant_detections += sum(max(det - 1, 0) for det in already_detected)
            if not at_least_one_detected:
                false_negatives += 1

        events_gt = P.Interval(*[P.closed(*row) for _, row in y_true[["StartTime", "EndTime"]].iterrows()])
        for pred, matched in zip(events_pred, matched_events_pred):
            if not matched and (pred & events_gt).empty:
                false_positives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        precision_one_detection = true_positives / (true_positives + redundant_detections) if (true_positives + redundant_detections) else 0.0

        if precision > 0:
            nominal_interval = P.closed(*self.full_range) - events_gt
            false_positives_interval = nominal_interval & events_pred
            nominal_seconds = sum((interval.upper - interval.lower).value / NANOSECONDS_IN_SECOND for interval in nominal_interval)
            false_positive_seconds = sum((interval.upper - interval.lower).value / NANOSECONDS_IN_SECOND for interval in false_positives_interval)
            if nominal_seconds > 0:
                precision *= (1 - false_positive_seconds / nominal_seconds)

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
        result_dict = {"alarming_precision": precision_one_detection, "EW_precision": precision, "EW_recall": recall}
        for b in self._betas:
            result_dict[f"EW_F_{b:.2f}"] = _fbeta(precision, recall, b)

        events_pred_ns = [(e.lower.value, e.upper.value if e.lower.value != e.upper.value else e.upper.value + 1) for e in events_pred]
        events_gt_ns = [(e.lower.value, e.upper.value if e.lower.value != e.upper.value else e.upper.value + 1) for e in events_gt]
        if len(events_gt_ns) == 0:
            aff_precision = 0.0
            aff_recall = 0.0
        else:
            pred_upper = max((max(pair) for pair in events_pred_ns), default=self.full_range[1].value)
            gt_upper = max((max(pair) for pair in events_gt_ns), default=self.full_range[1].value)
            corrected_full_upper_range = max(pred_upper, gt_upper, self.full_range[1].value)
            score_dict = pr_from_events(events_pred_ns, events_gt_ns, (self.full_range[0].value, corrected_full_upper_range))

            precision_dict = defaultdict(list)
            recall_dict = defaultdict(list)
            for pr, rec, zone in zip(score_dict["individual_precision_probabilities"], score_dict["individual_recall_probabilities"], events_gt):
                intersections = [P.closed(*row) & zone for _, row in y_true[["StartTime", "EndTime"]].iterrows()]
                y_true_in_zone = y_true[[not inter.empty for inter in intersections]]
                filtered_y_true_in_zone = y_true_in_zone.copy()
                for col, val in self.selected_labels.items():
                    filtered_y_true_in_zone = filtered_y_true_in_zone[filtered_y_true_in_zone[col].isin(val)]
                if len(y_true_in_zone) > len(filtered_y_true_in_zone):
                    continue
                for id_ in y_true_in_zone["ID"]:
                    precision_dict[id_].append(pr)
                    recall_dict[id_].append(rec)

            precision_list = [np.mean(pr) for pr in precision_dict.values()]
            recall_list = [np.mean(rec) for rec in recall_dict.values()]
            aff_precision = float(np.mean(precision_list)) if precision_list else 0.0
            aff_recall = float(np.mean(recall_list)) if recall_list else 0.0

        result_dict["AFF_precision"] = aff_precision
        result_dict["AFF_recall"] = aff_recall
        for b in self._betas:
            result_dict[f"AFF_F_{b:.2f}"] = _fbeta(aff_precision, aff_recall, b)
        return result_dict


class ADTQCLite:
    def __init__(self, exponent=math.e, full_range: Optional[tuple] = None, select_labels: Optional[dict] = None, name: Optional[str] = None):
        self.exponent = exponent
        self.full_range = full_range
        if select_labels is None or len(select_labels) == 0:
            self.selected_labels = dict()
            filter_string = "ALL"
        else:
            select_labels = {col: np.atleast_1d(val) for col, val in select_labels.items()}
            self.selected_labels = select_labels
            filter_string = "_".join(["_".join(val) for val in select_labels.values()])
        self.name = f"ADTQC_{filter_string}" if name is None else name

    def timing_curve(self, x, a, b):
        if (a == pd.Timedelta(0) or b == pd.Timedelta(0)) and x == pd.Timedelta(0):
            return 1
        if x <= -a or x >= b:
            return 0
        if -a < x <= pd.Timedelta(0):
            return ((x + a) / a) ** self.exponent
        denom_part = x / (b - x)
        return 1.0 / (1.0 + denom_part ** self.exponent)

    def score(self, y_true: pd.DataFrame, y_pred: dict) -> dict:
        for channel, values in y_pred.items():
            y_pred[channel] = np.asarray(values)
        min_y_pred = min(np.concatenate([y[..., 0] for y in y_pred.values()]))
        max_y_pred = max(np.concatenate([y[..., 0] for y in y_pred.values()]))
        if self.full_range is None:
            self.full_range = (min(y_true["StartTime"].min(), min_y_pred), max(y_true["EndTime"].max(), max_y_pred))

        for channel in list(y_pred.keys()):
            if y_pred[channel][0, 0] > self.full_range[0]:
                y_pred[channel] = np.array([np.array([self.full_range[0], y_pred[channel][0, 1]], dtype=object), *y_pred[channel]], dtype=object)
            if y_pred[channel][-1, 0] < self.full_range[1]:
                y_pred[channel] = np.array([*y_pred[channel], np.array([self.full_range[1], y_pred[channel][-1, 1]], dtype=object)], dtype=object)

        events_pred_dict = {channel: convert_time_series_to_events(np.asarray(pred)) for channel, pred in y_pred.items()}
        filtered_y_true = y_true.copy()
        for col, val in self.selected_labels.items():
            filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]

        unique_anomaly_ids = filtered_y_true["ID"].unique()
        start_times = sorted([min(filtered_y_true[filtered_y_true["ID"] == aid]["StartTime"]) for aid in unique_anomaly_ids])
        before_tps = []
        after_tps = []
        curve_scores = []
        for aid in unique_anomaly_ids:
            gt = filtered_y_true[filtered_y_true["ID"] == aid]
            affected_channels = np.sort(gt["Channel"].unique())
            channels_intervals = {}
            for channel in affected_channels:
                c_gt = gt[gt["Channel"] == channel]
                channels_intervals[channel] = P.Interval(*[P.closed(*row) for _, row in c_gt[["StartTime", "EndTime"]].iterrows()])

            global_preds = []
            global_gts = []
            for channel in affected_channels:
                events_pred = [pred for pred in events_pred_dict[channel] if not (pred & channels_intervals[channel]).empty]
                global_preds.extend(events_pred)
                global_gts.append(channels_intervals[channel])
            global_preds = P.Interval(*global_preds)
            if global_preds.empty:
                continue
            global_gts = P.Interval(*global_gts)

            anomaly_length = global_gts.upper - global_gts.lower
            current_anomaly_idx = start_times.index(global_gts.lower)
            previous_anomaly_start = start_times[current_anomaly_idx - 1] if current_anomaly_idx > 0 else global_gts.lower - anomaly_length
            alpha = min(anomaly_length, global_gts.lower - previous_anomaly_start)
            latency = global_preds.lower - global_gts.lower
            metric_value = self.timing_curve(latency, alpha, anomaly_length)
            curve_scores.append(metric_value)
            if latency < pd.Timedelta(0):
                before_tps.append(metric_value)
            else:
                after_tps.append(metric_value)

        curve_scores = np.array(curve_scores)
        return {
            "Nb_Before": len(before_tps),
            "Nb_After": len(after_tps),
            "AfterRate": len(after_tps) / len(curve_scores) if len(curve_scores) > 0 else np.nan,
            "Total": float(np.mean(curve_scores)) if len(curve_scores) > 0 else np.nan,
        }


class ChannelAwareFScoreLite:
    def __init__(self, beta=0.5, select_labels: Optional[dict] = None, full_range: Optional[tuple] = None, name: Optional[str] = None):
        self._beta = beta
        self.full_range = full_range
        if select_labels is None or len(select_labels) == 0:
            self.selected_labels = dict()
            filter_string = "ALL"
        else:
            select_labels = {col: np.atleast_1d(val) for col, val in select_labels.items()}
            self.selected_labels = select_labels
            filter_string = "_".join(["_".join(val) for val in select_labels.values()])
        self.name = f"PC_{filter_string}" if name is None else name

    def get_pr_re_f_score(self, true_positives, false_positives, false_negatives):
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
        return precision, recall, _fbeta(precision, recall, self._beta)

    def score(self, y_true: pd.DataFrame, y_pred: dict, subsystems_mapping: dict = None) -> Dict[str, float]:
        all_channels = list(y_pred.keys())
        for c, preds in y_pred.items():
            y_pred[c] = np.asarray(preds)

        min_ts = min([preds[..., 0].min() for preds in y_pred.values()])
        max_ts = max([preds[..., 0].max() for preds in y_pred.values()])
        if self.full_range is None:
            self.full_range = (min(y_true["StartTime"].min(), min_ts), max(y_true["EndTime"].max(), max_ts))
        for c in list(y_pred.keys()):
            if y_pred[c][0, 0] > self.full_range[0]:
                y_pred[c] = np.array([np.array([self.full_range[0], y_pred[c][0, 1]], dtype=object), *y_pred[c]], dtype=object)
            if y_pred[c][-1, 0] < self.full_range[1]:
                y_pred[c] = np.array([*y_pred[c], np.array([self.full_range[1], y_pred[c][-1, 1]], dtype=object)], dtype=object)

        events_pred_per_channel = {channel_name: convert_time_series_to_events(np.asarray(channel_pred)) for channel_name, channel_pred in y_pred.items()}
        filtered_y_true = y_true.copy()
        for col, val in self.selected_labels.items():
            filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]
        point_anomalies = filtered_y_true["StartTime"] == filtered_y_true["EndTime"]
        filtered_y_true.loc[point_anomalies, "EndTime"] = filtered_y_true.loc[point_anomalies, "StartTime"] + pd.Timedelta(milliseconds=1)

        unique_ids = filtered_y_true["ID"].unique()
        global_precisions, global_recalls, global_f_scores = [], [], []
        global_sub_precisions, global_sub_recalls, global_sub_f_scores = [], [], []

        aid_channels_intervals = {}
        for aid in unique_ids:
            gt = filtered_y_true[filtered_y_true["ID"] == aid]
            channels_intervals = {}
            for c in all_channels:
                c_gt = gt[gt["Channel"] == c]
                channels_intervals[c] = P.Interval(*[P.closed(*row) for _, row in c_gt[["StartTime", "EndTime"]].iterrows()])
            aid_channels_intervals[aid] = channels_intervals

        for aid in unique_ids:
            channels_intervals = aid_channels_intervals[aid]
            full_interval = P.Interval(*list(channels_intervals.values()))
            tp = fp = fn = 0
            for c in all_channels:
                is_channel_affected = not channels_intervals[c].empty
                detection_interval = full_interval & events_pred_per_channel[c]
                is_channel_detected = not detection_interval.empty
                if is_channel_affected and is_channel_detected:
                    tp += 1
                elif is_channel_affected and not is_channel_detected:
                    fn += 1
                elif not is_channel_affected and is_channel_detected:
                    for id_, a_intervals in aid_channels_intervals.items():
                        if aid == id_ or a_intervals[c].empty:
                            continue
                        if not (detection_interval & a_intervals[c]).empty:
                            break
                    else:
                        fp += 1
            precision, recall, f_score = self.get_pr_re_f_score(tp, fp, fn)
            global_precisions.append(precision)
            global_recalls.append(recall)
            global_f_scores.append(f_score)

            if subsystems_mapping is not None:
                tp = fp = fn = 0
                for subsystem_channel_names in subsystems_mapping.values():
                    subsystem_channel_names = [c for c in subsystem_channel_names if c in all_channels]
                    if not subsystem_channel_names:
                        continue
                    subsystem_interval = P.Interval(*[channels_intervals[channel_name] for channel_name in subsystem_channel_names])
                    is_subsystem_affected = not subsystem_interval.empty
                    detected_intervals = {c: full_interval & events_pred_per_channel[c] for c in subsystem_channel_names}
                    is_subsystem_detected = np.any([not d_i.empty for d_i in detected_intervals.values()])
                    if is_subsystem_affected and is_subsystem_detected:
                        tp += 1
                    elif is_subsystem_affected and not is_subsystem_detected:
                        fn += 1
                    elif not is_subsystem_affected and is_subsystem_detected:
                        for id_, a_intervals in aid_channels_intervals.items():
                            if aid == id_:
                                continue
                            for d_c, d_interval in detected_intervals.items():
                                if a_intervals[d_c].empty:
                                    continue
                                if not (d_interval & a_intervals[d_c]).empty:
                                    detected_intervals[d_c] = P.empty()
                        if np.any([not d_i.empty for d_i in detected_intervals.values()]):
                            fp += 1
                precision, recall, f_score = self.get_pr_re_f_score(tp, fp, fn)
                global_sub_precisions.append(precision)
                global_sub_recalls.append(recall)
                global_sub_f_scores.append(f_score)

        result = {
            "precision": float(np.mean(global_precisions)) if global_precisions else 0.0,
            "recall": float(np.mean(global_recalls)) if global_recalls else 0.0,
            "F": float(np.mean(global_f_scores)) if global_f_scores else 0.0,
        }
        if global_sub_f_scores:
            result.update({
                "subsystem_precision": float(np.mean(global_sub_precisions)),
                "subsystem_recall": float(np.mean(global_sub_recalls)),
                "subsystem_F": float(np.mean(global_sub_f_scores)),
            })
        return result


def evaluate_official_metrics(
    labels_df,
    test_data_scores,
    subsystems_mapping,
    timestamps,
    channel_names,
    y_scores,
    binary_fraction=0.05,
    merge_gap_points=0,
    min_event_points=1,
):
    if pr_from_events is None:
        raise ModuleNotFoundError(
            "Official ESA affiliation metrics are not available. "
            f"Expected the official repository at {AFFILIATION_REPO}. "
            "Clone https://github.com/kplabs-pl/ESA-ADB.git as a sibling directory named "
            "'esa-adb-classical', or run with --skip-official-metrics."
        ) from AFFILIATION_IMPORT_ERROR

    timestamps = pd.to_datetime(timestamps)
    y_scores = scale_scores(y_scores)
    y_binary = topk_binarize(y_scores, fraction=binary_fraction)
    y_binary = postprocess_binary_events(
        y_binary,
        merge_gap_points=merge_gap_points,
        min_event_points=min_event_points,
    )
    if y_binary.ndim == 1:
        y_binary = np.expand_dims(y_binary, -1)

    channel_names = list(channel_names)
    filtered_labels = labels_df[labels_df["Channel"].isin(channel_names)].copy()
    results = {}

    global_binary = y_binary.max(axis=1).astype(np.uint8)
    global_ts = make_time_series(timestamps, global_binary)

    global_metrics = [
        ESAScoresLite(betas=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]}),
        ESAScoresLite(betas=0.5, select_labels={"Category": ["Anomaly"]}),
    ]
    for metric in global_metrics:
        score = metric.score(filtered_labels.drop(columns=["Channel"]), global_ts)
        for submetric, value in score.items():
            results[f"Global_{metric.name}_{submetric}"] = value

    for metric in [
        ADTQCLite(select_labels={"Category": ["Rare Event", "Anomaly"]}),
        ADTQCLite(select_labels={"Category": ["Anomaly"]}),
    ]:
        y_true_global = filtered_labels.copy()
        y_true_global["Channel"] = "global"
        score = metric.score(y_true_global, {"global": global_ts})
        for submetric, value in score.items():
            results[f"Global_{metric.name}_{submetric}"] = value

    if y_binary.shape[1] > 1:
        y_pred = {channel_name: make_time_series(timestamps, y_binary[:, i].astype(np.uint8)) for i, channel_name in enumerate(channel_names)}
        for metric in [
            ChannelAwareFScoreLite(beta=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]}),
            ChannelAwareFScoreLite(beta=0.5, select_labels={"Category": ["Anomaly"]}),
        ]:
            score = metric.score(filtered_labels, y_pred, subsystems_mapping)
            for submetric, value in score.items():
                results[f"{metric.name}_{submetric}"] = value

    return results
