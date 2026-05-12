"""评估入口：基础点级指标 + 官方 ESA 事件指标。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from official_metrics import evaluate_official_metrics
from vus_metrics import VUSConfig, compute_vus_metrics


def _to_2d(array, dtype=None):
    out = np.asarray(array, dtype=dtype)
    if out.ndim == 1:
        out = np.expand_dims(out, -1)
    return out


def _to_global_point_series(values):
    array = np.asarray(values)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.max(axis=1)
    raise ValueError(f"Expected 1D or 2D array, got shape {array.shape}")


def _basic_binary_metrics(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError(f"Label/score length mismatch: {y_true.shape[0]} vs {y_scores.shape[0]}")

    if len(np.unique(y_true)) < 2:
        return {"AUROC": np.nan, "AUPR": np.nan, "F1": np.nan}

    n_anom = max(1, int(len(y_true) * 0.05))
    thr = np.sort(y_scores)[-n_anom]
    y_pred = (y_scores >= thr).astype(np.uint8)
    auroc = round(float(roc_auc_score(y_true, y_scores)), 4)
    aupr = round(float(average_precision_score(y_true, y_scores)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "F1": f1,
        "Point_AUROC": auroc,
        "Point_AUPR": aupr,
        "Point_F1_Top5": f1,
    }


def evaluate(
    y_true,
    y_scores,
    timestamps=None,
    channel_names=None,
    labels_df=None,
    test_data_scores=None,
    subsystems_mapping=None,
    include_official=True,
    include_vus=False,
    vus_config=None,
):
    """
    统一评估：
    1. 基础全局点级 AUROC/AUPR/F1
    2. 若存在多通道输出，则附加通道均值 AUROC/AUPR/F1
    3. 若提供官方标签上下文，则附加官方 Mission1 事件指标
    """
    y_true_2d = _to_2d(y_true, dtype=np.uint8)
    y_true_2d = (y_true_2d > 0).astype(np.uint8, copy=False)
    y_scores_2d = _to_2d(y_scores, dtype=np.float64)

    results = {}

    global_true = _to_global_point_series(y_true_2d).astype(np.uint8, copy=False)
    global_scores = _to_global_point_series(y_scores_2d).astype(np.float64, copy=False)
    results.update(_basic_binary_metrics(global_true, global_scores))
    if include_vus:
        results.update(compute_vus_metrics(global_true, global_scores, vus_config or VUSConfig()))

    if y_true_2d.shape[1] > 1 and y_scores_2d.shape[1] > 1 and y_true_2d.shape[1] == y_scores_2d.shape[1]:
        per_channel = []
        for i in range(y_true_2d.shape[1]):
            metric_values = _basic_binary_metrics(y_true_2d[:, i], y_scores_2d[:, i])
            per_channel.append(metric_values)

        for key in ["AUROC", "AUPR", "F1", "Point_AUROC", "Point_AUPR", "Point_F1_Top5"]:
            values = [item[key] for item in per_channel if not np.isnan(item[key])]
            results[f"ChannelMean_{key}"] = round(float(np.mean(values)), 4) if values else np.nan

    if include_official and timestamps is not None and channel_names is not None and labels_df is not None and test_data_scores is not None:
        results.update(
            evaluate_official_metrics(
                labels_df=labels_df,
                test_data_scores=test_data_scores,
                subsystems_mapping=subsystems_mapping,
                timestamps=timestamps,
                channel_names=channel_names,
                y_scores=y_scores_2d,
            )
        )

    return results
