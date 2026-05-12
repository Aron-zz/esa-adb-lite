"""Lightweight VUS metrics adapted from TimeEval's range VUS implementation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VUSConfig:
    max_buffer_size: int = 100
    max_threshold_samples: int = 50
    max_points: int = 200_000


class RangeAucMetric:
    def __init__(self, buffer_size=None, max_samples=50):
        self._buffer_size = buffer_size
        self._max_samples = max_samples

    @staticmethod
    def anomaly_bounds(y_true):
        labels = np.asarray(y_true) > 0
        labels = np.diff(np.r_[0, labels, 0])
        index = np.arange(labels.shape[0])
        return index[labels == 1], index[labels == -1]

    def _extend_anomaly_labels(self, y_true):
        starts, ends = self.anomaly_bounds(y_true)
        if starts.size == 0:
            return y_true.astype(np.float64), np.empty((0, 2), dtype=np.int64)

        buffer_size = self._buffer_size
        if buffer_size is None:
            buffer_size = int(np.median(ends - starts))

        if buffer_size <= 1:
            return y_true.astype(np.float64), np.array(list(zip(starts, ends)), dtype=np.int64)

        y_true_cont = y_true.astype(np.float64)
        slope_length = buffer_size // 2
        length = y_true_cont.shape[0]
        slope = np.linspace(1 / np.sqrt(2), 1, slope_length + 1)
        anomalies = np.empty((starts.shape[0], 2), dtype=np.int64)

        for i, (s, e) in enumerate(zip(starts, ends)):
            s0 = max(0, s - slope_length)
            s1 = s + 1
            y_true_cont[s0:s1] = np.maximum(slope[s0 - s1:], y_true_cont[s0:s1])

            e0 = e - 1
            e1 = min(length, e + slope_length)
            y_true_cont[e0:e1] = np.maximum(slope[e0 - e1:][::-1], y_true_cont[e0:e1])
            anomalies[i] = [s0, e1]

        return y_true_cont, anomalies

    def _uniform_threshold_sampling(self, y_score):
        n_samples = min(self._max_samples, y_score.shape[0])
        thresholds = np.sort(y_score)[::-1]
        return thresholds[np.linspace(0, thresholds.shape[0] - 1, n_samples, dtype=np.int64)]

    def _range_pr_roc_auc_support(self, y_true, y_score):
        y_true_cont, anomalies = self._extend_anomaly_labels(y_true)
        if anomalies.size == 0:
            return np.nan, np.nan

        thresholds = self._uniform_threshold_sampling(y_score)
        p = np.average([np.sum(y_true), np.sum(y_true_cont)])
        n = len(y_true) - p
        if p <= 0 or n <= 0:
            return np.nan, np.nan

        recalls = np.zeros(thresholds.shape[0] + 2)
        fprs = np.zeros(thresholds.shape[0] + 2)
        precisions = np.ones(thresholds.shape[0] + 1)

        for i, threshold in enumerate(thresholds):
            y_pred = y_score >= threshold
            pred_count = np.sum(y_pred)
            if pred_count == 0:
                continue

            product = y_true_cont * y_pred
            tp = np.sum(product)
            fp = pred_count - tp

            existence_reward = np.mean([np.sum(product[s:e + 1]) > 0 for s, e in anomalies])
            recalls[i + 1] = min(tp / p, 1.0) * existence_reward
            fprs[i + 1] = min(fp / n, 1.0)
            precisions[i + 1] = tp / pred_count

        recalls[-1] = 1.0
        fprs[-1] = 1.0

        pr_auc = np.sum((recalls[1:-1] - recalls[:-2]) * (precisions[1:] + precisions[:-1]) / 2)
        roc_auc = np.sum((fprs[1:] - fprs[:-1]) * (recalls[1:] + recalls[:-1]) / 2)
        return float(pr_auc), float(roc_auc)


def _downsample_max_bins(y_true, y_score, max_points):
    if max_points is None or max_points <= 0 or len(y_true) <= max_points:
        return y_true, y_score, 1

    stride = int(np.ceil(len(y_true) / max_points))
    n_bins = int(np.ceil(len(y_true) / stride))
    pad = n_bins * stride - len(y_true)

    if pad:
        y_true = np.pad(y_true, (0, pad), constant_values=0)
        y_score = np.pad(y_score, (0, pad), constant_values=np.nan)

    y_true_binned = y_true.reshape(n_bins, stride).max(axis=1)
    y_score_binned = np.nanmax(y_score.reshape(n_bins, stride), axis=1)
    return y_true_binned, y_score_binned, stride


def compute_vus_metrics(y_true, y_score, config=None):
    """Return global VUS-PR/VUS-ROC.

    The optional max_points reduction keeps large ESA test files practical. When
    reduction is used, each consecutive bin preserves the maximum label and score.
    """
    config = config or VUSConfig()
    y_true = np.asarray(y_true, dtype=np.uint8).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"Label/score length mismatch: {y_true.shape[0]} vs {y_score.shape[0]}")
    if len(np.unique(y_true)) < 2:
        return {"VUS_PR": np.nan, "VUS_ROC": np.nan}

    finite = np.isfinite(y_score)
    if not np.all(finite):
        y_true = y_true[finite]
        y_score = y_score[finite]

    y_true, y_score, stride = _downsample_max_bins(y_true, y_score, config.max_points)
    max_buffer = max(1, int(np.ceil(config.max_buffer_size / stride)))

    pr_values = []
    roc_values = []
    for buffer_size in range(max_buffer + 1):
        metric = RangeAucMetric(buffer_size=buffer_size, max_samples=config.max_threshold_samples)
        pr_auc, roc_auc = metric._range_pr_roc_auc_support(y_true, y_score)
        pr_values.append(pr_auc)
        roc_values.append(roc_auc)

    return {
        "VUS_PR": round(float(np.nanmean(pr_values)), 4),
        "VUS_ROC": round(float(np.nanmean(roc_values)), 4),
        "VUS_MaxBuffer": config.max_buffer_size,
        "VUS_MaxThresholds": config.max_threshold_samples,
        "VUS_MaxPoints": config.max_points,
    }
