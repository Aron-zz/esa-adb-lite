"""评估指标：AUROC / AUPR / F1。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def evaluate(y_true, y_scores):
    """
    y_true: (n, d) uint8 — 逐点标签
    y_scores: (n, d) float — 异常分数（越高越异常）
    """
    y_true = y_true.ravel()
    y_scores = y_scores.ravel()

    if len(np.unique(y_true)) < 2:
        return {"AUROC": np.nan, "AUPR": np.nan, "F1": np.nan}

    # 二值化预测：top-5% 为异常
    n_anom = max(1, int(len(y_true) * 0.05))
    thr = np.sort(y_scores)[-n_anom]
    y_pred = (y_scores >= thr).astype(np.uint8)

    return {
        "AUROC": round(float(roc_auc_score(y_true, y_scores)), 4),
        "AUPR":  round(float(average_precision_score(y_true, y_scores)), 4),
        "F1":    round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
