"""经典异常检测算法，纯函数，无 Docker 依赖。"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.decomposition import PCA


def _per_channel_scores(X, score_fn):
    """逐通道计算异常分数，返回 (n_samples, n_channels) 形状。"""
    scores = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[1]):
        col = X[:, i].reshape(-1, 1)
        scores[:, i] = score_fn(col).ravel()
    return scores


def _sample_rows(X, max_rows, random_state=42):
    """Deterministically sample rows for neighbor-based methods on large CSVs."""
    if max_rows is None or max_rows <= 0 or X.shape[0] <= max_rows:
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    idx.sort()
    return X[idx]


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


def std3(X_train, X_test, random_state=42):
    """Official Mission1 STD baseline with tolerance 3."""
    return std(X_train, X_test, tol=3.0, random_state=random_state)


def std5(X_train, X_test, random_state=42):
    """Official Mission1 STD baseline with tolerance 5."""
    return std(X_train, X_test, tol=5.0, random_state=random_state)


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


def lof(X_train, X_test, n_neighbors=20, max_train_samples=50_000, random_state=42):
    """Local Outlier Factor，逐通道 novelty 模式；大训练集默认抽样。"""
    def _lof(train_col, test_col):
        train_col = _sample_rows(train_col, max_train_samples, random_state)
        n = max(2, min(n_neighbors, len(train_col) - 1))
        clf = LocalOutlierFactor(n_neighbors=n, novelty=True, n_jobs=1)
        clf.fit(train_col)
        return -clf.decision_function(test_col)

    scores = np.empty_like(X_test, dtype=np.float64)
    for i in range(X_test.shape[1]):
        train_col = X_train[:, i].reshape(-1, 1)
        test_col = X_test[:, i].reshape(-1, 1)
        scores[:, i] = _lof(train_col, test_col).ravel()
    return scores


def copod(X_train, X_test, random_state=42):
    """轻量 COPOD 风格经验尾概率分数，逐通道。"""
    scores = np.empty_like(X_test, dtype=np.float64)
    eps = 1e-12
    for i in range(X_test.shape[1]):
        train_col = np.sort(X_train[:, i].astype(np.float64, copy=False))
        test_col = X_test[:, i].astype(np.float64, copy=False)
        n = len(train_col)
        left = np.searchsorted(train_col, test_col, side="right") / n
        right = (n - np.searchsorted(train_col, test_col, side="left")) / n
        tail = np.clip(2.0 * np.minimum(left, right), eps, 1.0)
        scores[:, i] = -np.log(tail)
    return scores


def robust_pca(X_train, X_test, n_components=5, random_state=42):
    """Median/IQR 标准化后的 PCA 重构误差，多变量全局分数。"""
    med = np.median(X_train, axis=0)
    q25 = np.percentile(X_train, 25, axis=0)
    q75 = np.percentile(X_train, 75, axis=0)
    iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)
    X_train_scaled = (X_train - med) / iqr
    X_test_scaled = (X_test - med) / iqr
    n = max(1, min(n_components, X_train_scaled.shape[1], X_train_scaled.shape[0]))
    clf = PCA(n_components=n, random_state=random_state).fit(X_train_scaled)
    X_recon = clf.inverse_transform(clf.transform(X_test_scaled))
    return np.mean((X_test_scaled - X_recon) ** 2, axis=1, keepdims=True)


def knn(X_train, X_test, y_train=None, n_neighbors=5, max_train_samples=100_000, random_state=42):
    """kNN 距离异常分数，近似官方版本：先按训练正常样本统计量标准化。"""
    mean, stdv = _train_stats_from_nominal(X_train, y_train)
    X_train_scaled = (X_train - mean) / stdv
    X_test_scaled = (X_test - mean) / stdv
    X_train_scaled = _sample_rows(X_train_scaled, max_train_samples, random_state)
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


def subsequence_knn(X_train, X_test, window_size=17, n_neighbors=5, max_train_windows=100_000, random_state=42):
    """滑动窗口 + kNN 距离，逐通道；大训练窗口默认抽样。"""
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
        train_win = _sample_rows(train_win, max_train_windows, random_state)
        n = max(1, min(n_neighbors, len(train_win)))
        nn = NearestNeighbors(n_neighbors=n, n_jobs=1).fit(train_win)
        dist, _ = nn.kneighbors(test_win)
        scores[:, i] = _reverse_window_scores(np.mean(dist, axis=1), len(test_col), window_size)
    return scores


ALGORITHMS = {
    "PCC": pcc,
    "HBOS": hbos,
    "STD": std3,
    "STD3": std3,
    "STD5": std5,
    "iForest": iforest,
    "LOF": lof,
    "COPOD": copod,
    "RobustPCA": robust_pca,
    "KNN": knn,
    "Subsequence_IF": subsequence_if,
    "Subsequence_KNN": subsequence_knn,
}
