"""6 种经典异常检测算法，纯函数，无 Docker 依赖。"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def _per_channel_scores(X, score_fn):
    """逐通道计算异常分数，返回 (n_samples, n_channels) 形状。"""
    scores = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[1]):
        col = X[:, i].reshape(-1, 1)
        scores[:, i] = score_fn(col).ravel()
    return scores


def pcc(X_train, X_test, random_state=42):
    """PCA 重构误差"""
    clf = PCA(n_components=2, random_state=random_state).fit(X_train)
    X_recon = clf.inverse_transform(clf.transform(X_test))
    return np.mean((X_test - X_recon) ** 2, axis=1, keepdims=True)


def hbos(X_train, X_test, n_bins=50, random_state=42):
    """Histogram-Based Outlier Score，逐通道。"""
    def _hbos(col):
        n = len(col)
        counts, edges = np.histogram(col, bins=n_bins)
        freqs = np.clip(counts / n, 1e-10, 1.0)
        indices = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1)
        return -np.log(freqs[indices])
    return _per_channel_scores(X_test, _hbos)


def std(X_train, X_test, tol=3.0, random_state=42):
    """统计阈值：偏离 mean ± tol*std 判定为异常。"""
    mean = np.mean(X_train, axis=0)
    stdv = np.std(X_train, axis=0)
    stdv = np.where(stdv == 0, 1.0, stdv)
    return ((X_test > mean + tol * stdv) | (X_test < mean - tol * stdv)).astype(np.float64)


def _train_stats_from_nominal(X_train, y_train=None):
    if y_train is None:
        mean = np.mean(X_train, axis=0)
        stdv = np.std(X_train, axis=0)
        return mean, np.where(stdv == 0, 1.0, stdv)

    y_train = np.asarray(y_train)
    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, -1)

    means = []
    stds = []
    for i in range(X_train.shape[1]):
        labels = y_train[:, min(i, y_train.shape[1] - 1)]
        nominal = X_train[:, i][labels == 0]
        if nominal.size == 0:
            nominal = X_train[:, i]
        means.append(np.mean(nominal))
        stds.append(np.std(nominal.astype(float)))
    stds = np.asarray(stds)
    return np.asarray(means), np.where(stds == 0, 1.0, stds)


def iforest(X_train, X_test, n_estimators=100, random_state=42):
    """Isolation Forest，逐通道。"""
    def _iforest(col):
        clf = IsolationForest(n_estimators=n_estimators, contamination="auto",
                              random_state=random_state, n_jobs=1)
        clf.fit(col)
        return -clf.decision_function(col)
    return _per_channel_scores(X_test, _iforest)


def knn(X_train, X_test, y_train=None, n_neighbors=5, random_state=42):
    """kNN 距离异常分数，近似官方版本：先按训练正常样本统计量标准化。"""
    mean, stdv = _train_stats_from_nominal(X_train, y_train)
    X_train_scaled = (X_train - mean) / stdv
    X_test_scaled = (X_test - mean) / stdv
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1).fit(X_train_scaled)
    dist, _ = nn.kneighbors(X_test_scaled)
    return np.mean(dist, axis=1, keepdims=True)


def _reverse_window_scores(scores_win, input_length, window_size):
    """将窗口分数按重叠均值回填为点分数。"""
    point_scores = np.zeros(input_length, dtype=np.float64)
    counts = np.zeros(input_length, dtype=np.float64)

    for offset in range(window_size):
        point_scores[offset:offset + len(scores_win)] += scores_win
        counts[offset:offset + len(scores_win)] += 1.0

    counts = np.where(counts == 0, 1.0, counts)
    return point_scores / counts


def subsequence_if(X_train, X_test, window_size=17, n_estimators=200, random_state=42):
    """滑动窗口 + IForest，逐通道。"""
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Channel count mismatch: {X_train.shape[1]} vs {X_test.shape[1]}")
    if X_train.shape[0] < window_size or X_test.shape[0] < window_size:
        raise ValueError(
            f"window_size={window_size} exceeds train/test length: {X_train.shape[0]}, {X_test.shape[0]}"
        )

    scores = np.empty((X_test.shape[0], X_test.shape[1]), dtype=np.float64)

    for i in range(X_test.shape[1]):
        train_col = X_train[:, i].astype(np.float64, copy=False)
        test_col = X_test[:, i].astype(np.float64, copy=False)

        train_win = np.lib.stride_tricks.sliding_window_view(train_col, window_size)
        test_win = np.lib.stride_tricks.sliding_window_view(test_col, window_size)

        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=1,
        )
        clf.fit(train_win)
        scores_win = -clf.decision_function(test_win)
        scores[:, i] = _reverse_window_scores(scores_win, len(test_col), window_size)

    return scores


ALGORITHMS = {
    "PCC": pcc,
    "HBOS": hbos,
    "STD": std,
    "iForest": iforest,
    "KNN": knn,
    "Subsequence_IF": subsequence_if,
}
